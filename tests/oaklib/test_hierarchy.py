"""Entailment and walks over the strict Is a hierarchy, on a small CDM where it differs from concept_ancestor."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fixtures.cdm_builders import (
    ancestor,
    classified,
    concept,
    edge,
    relationship,
    relationship_class,
    sqlite_cdm,
    synonym,
)
from oaklib.datamodels.vocabulary import IS_A
from omop_alchemy.cdm.model.vocabulary.concept import Concept
from omop_alchemy.cdm.model.vocabulary.concept_ancestor import Concept_Ancestor
from omop_alchemy.cdm.model.vocabulary.concept_relationship import Concept_Relationship
from omop_alchemy.cdm.model.vocabulary.concept_synonym import Concept_Synonym
from omop_alchemy.cdm.model.vocabulary.relationship import Relationship
from omop_alchemy.cdm.model.vocabulary.vocabulary import Vocabulary

from omop_graph.extensions.omop_alchemy import (
    HierarchyDirection,
    PredicateKind,
    RelationshipClass,
    RelationshipMapping,
)
from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.oaklib_interface.codec import OMOPLanguage
from omop_graph.oaklib_interface.omop_implementation import OMOPAlchemyImplementation
from omop_graph.reasoning.concept_handlers import standardise_ids

(
    DISEASE,
    CANCER,
    SMOKING_RELATED,
    LUNG_CANCER,
    RETIRED_PARENT,
    COHORT,
    LUNG,
    THORAX_ORGAN,
    LOOP_A,
    LOOP_B,
) = range(200, 210)
OUTDATED, OUTER_ROOT, SOURCE, SOURCE_TWO, ORPHAN = range(210, 215)
SITE = "OMOP_REL:Has%20finding%20site"
MAPS_TO = "OMOP_REL:Maps%20to"
MAPPED_FROM = "OMOP_REL:Mapped%20from"
SUBSUMES = "OMOP_REL:Subsumes"


def curie(concept_id: int) -> str:
    return f"OMOP:{concept_id}"


@pytest.fixture(scope="module")
def adapter() -> Iterator[OMOPAlchemyImplementation]:
    rows = [
        *(concept(concept_id) for concept_id in range(DISEASE, LOOP_B + 1)),
        concept(OUTDATED, invalid_reason="U", standard_concept=None),
        concept(OUTER_ROOT),
        *(
            concept(concept_id, standard_concept=None)
            for concept_id in (SOURCE, SOURCE_TWO, ORPHAN)
        ),
        relationship("Is a", "Subsumes"),
        relationship("Subsumes", "Is a"),
        relationship("Has finding site", "Finding site of"),
        relationship("Maps to", "Mapped from"),
        relationship("Mapped from", "Maps to"),
        relationship_class(PredicateKind.IDENTITY, "Exact"),
        relationship_class(PredicateKind.HIERARCHY, "Taxonomic - up", HierarchyDirection.UP),
        relationship_class(PredicateKind.HIERARCHY, "Taxonomic - down", HierarchyDirection.DOWN),
        relationship_class(PredicateKind.ATTRIBUTE, "Site"),
        classified("Maps to", PredicateKind.IDENTITY, "Exact"),
        classified("Mapped from", PredicateKind.IDENTITY, "Exact"),
        classified("Subsumes", PredicateKind.HIERARCHY, "Taxonomic - down"),
        classified("Is a", PredicateKind.HIERARCHY, "Taxonomic - up", generalises=True),
        classified("Has finding site", PredicateKind.ATTRIBUTE, "Site", generalises=True),
        classified("Finding site of", PredicateKind.ATTRIBUTE, "Site"),
        edge(SOURCE, "Maps to", LUNG_CANCER),
        edge(SOURCE_TWO, "Maps to", LUNG_CANCER),
        edge(SOURCE_TWO, "Maps to", LUNG),
        edge(LUNG_CANCER, "Mapped from", SOURCE),
        edge(DISEASE, "Subsumes", CANCER),
        edge(CANCER, "Subsumes", LUNG_CANCER),
        edge(CANCER, "Is a", DISEASE),
        edge(SMOKING_RELATED, "Is a", DISEASE),
        edge(LUNG_CANCER, "Is a", CANCER),
        edge(LUNG_CANCER, "Is a", SMOKING_RELATED),
        edge(LUNG_CANCER, "Is a", RETIRED_PARENT, invalid_reason="D"),
        edge(CANCER, "Is a", OUTDATED),
        edge(OUTDATED, "Is a", OUTER_ROOT),
        edge(LUNG, "Is a", THORAX_ORGAN),
        edge(LUNG_CANCER, "Has finding site", LUNG),
        edge(LOOP_A, "Is a", LOOP_B),
        edge(LOOP_B, "Is a", LOOP_A),
        synonym(LUNG_CANCER, "Lung neoplasm", OMOPLanguage.ENGLISH),
        synonym(LUNG_CANCER, "Cancer de pulmon", OMOPLanguage.SPANISH),
        ancestor(DISEASE, LUNG_CANCER),
        ancestor(COHORT, LUNG_CANCER),
    ]
    models = (
        Concept,
        Relationship,
        Concept_Ancestor,
        Concept_Relationship,
        Concept_Synonym,
        Vocabulary,
        RelationshipClass,
        RelationshipMapping,
    )
    with sqlite_cdm(models, rows) as sessions:
        yield OMOPAlchemyImplementation(kg=KnowledgeGraph(sessions))


def test_closure_follows_active_is_a_edges_only(adapter) -> None:
    up = adapter.kg.closure([LUNG_CANCER, LUNG])

    assert up[LUNG_CANCER] == {CANCER, SMOKING_RELATED, DISEASE, OUTDATED, OUTER_ROOT}
    assert up[LUNG] == {THORAX_ORGAN}


def test_closure_reports_only_concepts_in_scope_but_walks_through_the_rest(adapter) -> None:
    scoped = adapter.kg.closure([LUNG_CANCER], concept_filter=adapter.concept_filter)

    assert scoped[LUNG_CANCER] == {CANCER, SMOKING_RELATED, DISEASE, OUTER_ROOT}


def test_closure_down_and_cycles_never_reach_the_start(adapter) -> None:
    assert adapter.kg.closure([DISEASE], HierarchyDirection.DOWN)[DISEASE] == {
        CANCER,
        SMOKING_RELATED,
        LUNG_CANCER,
    }
    assert adapter.kg.closure([LOOP_A])[LOOP_A] == {LOOP_B}


def test_entailed_outgoing_generalises_the_fillers(adapter) -> None:
    assert set(adapter.entailed_outgoing_relationships(curie(LUNG_CANCER))) == {
        (IS_A, curie(CANCER)),
        (IS_A, curie(SMOKING_RELATED)),
        (IS_A, curie(DISEASE)),
        (IS_A, curie(OUTER_ROOT)),
        (SITE, curie(LUNG)),
        (SITE, curie(THORAX_ORGAN)),
        (MAPPED_FROM, curie(SOURCE)),
    }


def test_entailed_outgoing_respects_the_predicate_filter(adapter) -> None:
    assert set(adapter.entailed_outgoing_relationships(curie(LUNG_CANCER), [SITE])) == {
        (SITE, curie(LUNG)),
        (SITE, curie(THORAX_ORGAN)),
    }


def test_a_concept_with_no_site_on_it_or_above_it_has_none(adapter) -> None:
    assert list(adapter.entailed_outgoing_relationships(curie(CANCER), [SITE])) == []


def test_entailed_incoming_reaches_through_descendants(adapter) -> None:
    assert set(adapter.entailed_incoming_relationships(curie(THORAX_ORGAN))) == {
        (IS_A, curie(LUNG)),
        (SITE, curie(LUNG_CANCER)),
    }
    assert set(adapter.entailed_incoming_relationships(curie(DISEASE), [IS_A])) == {
        (IS_A, curie(CANCER)),
        (IS_A, curie(SMOKING_RELATED)),
        (IS_A, curie(LUNG_CANCER)),
    }


def test_entailed_incoming_is_the_mirror_of_outgoing(adapter) -> None:
    outgoing = {
        (curie(LUNG_CANCER), p, o)
        for p, o in adapter.entailed_outgoing_relationships(curie(LUNG_CANCER))
    }
    incoming = {
        (s, p, o)
        for o in (
            curie(c)
            for c in (CANCER, SMOKING_RELATED, DISEASE, OUTER_ROOT, LUNG, THORAX_ORGAN, SOURCE)
        )
        for p, s in adapter.entailed_incoming_relationships(o)
        if s == curie(LUNG_CANCER)
    }

    assert incoming == outgoing


def test_entailed_between(adapter) -> None:
    assert set(adapter.entailed_relationships_between(curie(LUNG_CANCER), curie(DISEASE))) == {IS_A}
    assert set(adapter.entailed_relationships_between(curie(LUNG_CANCER), curie(THORAX_ORGAN))) == {
        SITE
    }
    assert list(adapter.entailed_relationships_between(curie(DISEASE), curie(LUNG_CANCER))) == []
    assert list(adapter.entailed_relationships_between(curie(LUNG_CANCER), curie(OUTDATED))) == []
    assert list(adapter.entailed_relationships_between(curie(LUNG_CANCER), curie(COHORT))) == []


def test_by_curie_variants_delegate(adapter) -> None:
    assert set(adapter.entailed_outgoing_relationships_by_curie(curie(LUNG))) == {
        (IS_A, curie(THORAX_ORGAN))
    }
    assert set(adapter.entailed_incoming_relationships_by_curie(curie(LUNG))) == {
        (SITE, curie(LUNG_CANCER)),
        (MAPS_TO, curie(SOURCE_TWO)),
    }


def test_a_mapping_is_not_generalised_up_the_hierarchy(adapter) -> None:
    assert set(adapter.entailed_outgoing_relationships(curie(SOURCE_TWO), [MAPS_TO])) == {
        (MAPS_TO, curie(LUNG_CANCER)),
        (MAPS_TO, curie(LUNG)),
    }
    assert set(adapter.entailed_incoming_relationships(curie(THORAX_ORGAN), [MAPS_TO])) == set()


def test_a_relationship_that_does_not_generalise_is_stored_edges_only(adapter) -> None:
    assert (
        set(adapter.entailed_outgoing_relationships(curie(LUNG), ["OMOP_REL:Finding%20site%20of"]))
        == set()
    )


def test_a_named_down_relationship_is_stored_edges_only(adapter) -> None:
    stored = {
        (curie(DISEASE), SUBSUMES, curie(CANCER)),
        (curie(CANCER), SUBSUMES, curie(LUNG_CANCER)),
    }

    assert set(adapter.relationships(predicates=[SUBSUMES], include_entailed=True)) == stored
    assert (
        set(
            adapter.relationships(
                subjects=[curie(DISEASE), curie(CANCER)],
                predicates=[SUBSUMES],
                include_entailed=True,
            )
        )
        == stored
    )


def test_relationships_with_include_entailed_inverts(adapter) -> None:
    assert set(
        adapter.relationships(
            subjects=[curie(DISEASE)], predicates=[IS_A], include_entailed=True, invert=True
        )
    ) == {
        (curie(DISEASE), IS_A, curie(CANCER)),
        (curie(DISEASE), IS_A, curie(SMOKING_RELATED)),
        (curie(DISEASE), IS_A, curie(LUNG_CANCER)),
    }


def test_every_entailed_relationship_is_streamed_once(adapter) -> None:
    streamed = list(adapter._all_entailed_relationships())

    assert len(streamed) == len(set(streamed))
    assert (curie(LUNG_CANCER), IS_A, curie(DISEASE)) in streamed
    assert (curie(LUNG_CANCER), SITE, curie(THORAX_ORGAN)) in streamed
    assert not any(s == curie(COHORT) for s, _, _ in streamed)
    assert (curie(CANCER), IS_A, curie(OUTDATED)) in streamed
    assert (curie(LUNG_CANCER), IS_A, curie(OUTDATED)) not in streamed
    assert not any(s == curie(OUTDATED) for s, _, _ in streamed)
    assert set(adapter.relationships(include_entailed=True, predicates=[SITE])) == {
        (curie(LUNG_CANCER), SITE, curie(LUNG)),
        (curie(LUNG_CANCER), SITE, curie(THORAX_ORGAN)),
    }


def test_the_graph_requires_subjects_or_objects(adapter) -> None:
    with pytest.raises(ValueError):
        list(adapter.kg.entailed_relationships(None, None))


def test_ancestors_default_to_the_strict_hierarchy_within_scope(adapter) -> None:
    assert set(adapter.ancestors(curie(LUNG_CANCER))) == {
        curie(c) for c in (LUNG_CANCER, CANCER, SMOKING_RELATED, DISEASE, OUTER_ROOT)
    }
    assert set(adapter.ancestors(curie(LUNG_CANCER), reflexive=False)) == {
        curie(c) for c in (CANCER, SMOKING_RELATED, DISEASE, OUTER_ROOT)
    }


def test_ancestors_follow_several_predicates_at_once(adapter) -> None:
    assert set(adapter.ancestors(curie(LUNG_CANCER), [IS_A, SITE], reflexive=False)) == {
        curie(c) for c in (CANCER, SMOKING_RELATED, DISEASE, OUTER_ROOT, LUNG, THORAX_ORGAN)
    }
    assert set(adapter.ancestors(curie(LUNG_CANCER), [SITE], reflexive=False)) == {curie(LUNG)}


def test_ancestors_take_many_starts_and_are_not_reflexive_over_each_other(adapter) -> None:
    starts = [curie(CANCER), curie(LUNG_CANCER)]

    assert set(adapter.ancestors(starts, reflexive=False)) == {
        curie(c) for c in (SMOKING_RELATED, DISEASE, OUTER_ROOT)
    }


def test_ancestors_with_entailment_agree_with_walking(adapter) -> None:
    from oaklib.interfaces.obograph_interface import GraphTraversalMethod

    assert set(
        adapter.ancestors(curie(LUNG_CANCER), [IS_A], method=GraphTraversalMethod.ENTAILMENT)
    ) == set(adapter.ancestors(curie(LUNG_CANCER), [IS_A], reflexive=False))


def test_descendants_and_their_count(adapter) -> None:
    assert set(adapter.descendants(curie(DISEASE), reflexive=False)) == {
        curie(c) for c in (CANCER, SMOKING_RELATED, LUNG_CANCER)
    }
    assert adapter.descendant_count(curie(DISEASE), reflexive=False) == 3
    assert set(adapter.descendants(curie(OUTER_ROOT), reflexive=False)) == {
        curie(CANCER),
        curie(LUNG_CANCER),
    }


def test_cycles_end_and_unknown_starts_are_kept_only_when_reflexive(adapter) -> None:
    assert set(adapter.ancestors(curie(LOOP_A), reflexive=False)) == {curie(LOOP_B)}
    assert set(adapter.ancestors("OMOP:999999")) == {"OMOP:999999"}
    assert list(adapter.ancestors("OMOP:999999", reflexive=False)) == []


def test_the_ancestor_graph_stays_connected_through_concepts_outside_the_scope(adapter) -> None:
    graph = adapter.ancestor_graph(curie(LUNG_CANCER))

    assert {node.id for node in graph.nodes if node.type == "CLASS"} == {
        curie(c) for c in (LUNG_CANCER, CANCER, SMOKING_RELATED, DISEASE, OUTDATED, OUTER_ROOT)
    }
    assert {(e.sub, e.pred, e.obj) for e in graph.edges} == {
        (curie(LUNG_CANCER), IS_A, curie(CANCER)),
        (curie(LUNG_CANCER), IS_A, curie(SMOKING_RELATED)),
        (curie(CANCER), IS_A, curie(DISEASE)),
        (curie(CANCER), IS_A, curie(OUTDATED)),
        (curie(SMOKING_RELATED), IS_A, curie(DISEASE)),
        (curie(OUTDATED), IS_A, curie(OUTER_ROOT)),
    }
    assert {node.id: node.lbl for node in graph.nodes}[curie(CANCER)] == f"Concept {CANCER}"


def test_the_descendant_graph_and_a_start_without_edges(adapter) -> None:
    graph = adapter.descendant_graph([curie(DISEASE)])

    assert {e.sub for e in graph.edges} == {
        curie(c) for c in (CANCER, SMOKING_RELATED, LUNG_CANCER)
    }
    assert {n.id for n in adapter.ancestor_graph(curie(COHORT)).nodes} == {curie(COHORT)}


def test_paths_run_through_the_ancestor_graph(adapter) -> None:
    paths = set(adapter.paths([curie(LUNG_CANCER)], [curie(DISEASE)]))

    assert paths
    assert {s for s, _, _ in paths} == {curie(LUNG_CANCER)}
    assert {o for _, o, _ in paths} == {curie(DISEASE)}


def test_a_concept_node(adapter) -> None:
    node = adapter.node(curie(LUNG_CANCER), include_metadata=True)

    assert (node.id, node.lbl, node.type) == (
        curie(LUNG_CANCER),
        f"Concept {LUNG_CANCER}",
        "CLASS",
    )
    assert {(s.val, s.lang) for s in node.meta.synonyms} == {
        ("Lung neoplasm", "en"),
        ("Cancer de pulmon", "es"),
    }
    assert [x.val for x in node.meta.xrefs] == [f"SNOMED:C{LUNG_CANCER}"]
    assert not node.meta.deprecated


def test_a_deprecated_and_a_predicate_node(adapter) -> None:
    assert adapter.node(curie(OUTDATED), include_metadata=True).meta.deprecated is True
    predicate = adapter.node(SITE)
    assert (predicate.lbl, predicate.type) == ("Has finding site", "PROPERTY")
    assert adapter.node("OMOP:999999") is None
    with pytest.raises(ValueError, match="does not exist"):
        adapter.node("OMOP:999999", strict=True)
    assert adapter.node(curie(CANCER), expand_curies=True).id != curie(CANCER)


def test_synonyms_of_many_concepts(adapter) -> None:
    found = list(adapter.synonym_property_values([curie(LUNG_CANCER), curie(CANCER)]))

    assert {c for c, _ in found} == {curie(LUNG_CANCER)}
    assert {spv.pred for _, spv in found} == {"hasExactSynonym"}


@pytest.fixture(scope="module")
def standardising(adapter) -> OMOPAlchemyImplementation:
    return OMOPAlchemyImplementation(kg=adapter.kg, standardise=True)


def test_a_concept_stands_for_its_standard_concepts(adapter) -> None:
    assert adapter.kg.standardise([LUNG_CANCER, SOURCE, SOURCE_TWO, ORPHAN, 999999]) == {
        LUNG_CANCER: (LUNG_CANCER,),
        SOURCE: (LUNG_CANCER,),
        SOURCE_TWO: (LUNG_CANCER, LUNG),
        ORPHAN: (),
        999999: (),
    }


def test_standardise_ids_picks_the_lowest_standard_concept(adapter) -> None:
    assert standardise_ids({SOURCE_TWO, ORPHAN, CANCER}, adapter.kg) == {
        SOURCE_TWO: LUNG_CANCER,
        ORPHAN: ORPHAN,
        CANCER: CANCER,
    }


def test_a_non_standard_concept_has_no_ancestors_by_default(adapter) -> None:
    assert list(adapter.ancestors(curie(SOURCE), reflexive=False)) == []


def test_a_non_standard_concept_takes_the_ancestors_of_its_standard_concepts(
    standardising,
) -> None:
    assert set(standardising.ancestors(curie(SOURCE), reflexive=False)) == {
        curie(c) for c in (LUNG_CANCER, CANCER, SMOKING_RELATED, DISEASE, OUTER_ROOT)
    }
    assert set(standardising.ancestors(curie(SOURCE_TWO), reflexive=False)) >= {
        curie(THORAX_ORGAN),
        curie(DISEASE),
    }
    assert list(standardising.ancestors(curie(ORPHAN), reflexive=False)) == []


def test_a_standard_concept_walks_as_before_with_standardise(adapter, standardising) -> None:
    assert set(standardising.ancestors(curie(LUNG_CANCER))) == set(
        adapter.ancestors(curie(LUNG_CANCER))
    )


def test_a_non_standard_concept_takes_the_entailed_relationships_of_its_standard_concepts(
    adapter, standardising
) -> None:
    own = set(adapter.entailed_outgoing_relationships(curie(SOURCE), [IS_A, SITE]))
    assert own == set()
    assert set(standardising.entailed_outgoing_relationships(curie(SOURCE), [IS_A, SITE])) == {
        (IS_A, curie(CANCER)),
        (IS_A, curie(DISEASE)),
        (IS_A, curie(SMOKING_RELATED)),
        (IS_A, curie(OUTER_ROOT)),
        (SITE, curie(LUNG)),
        (SITE, curie(THORAX_ORGAN)),
    }


def test_a_stand_in_does_not_lend_its_identity_relationships(adapter, standardising) -> None:
    triples = set(standardising.relationships(subjects=[curie(SOURCE)], include_entailed=True))

    assert triples >= {(curie(SOURCE), MAPS_TO, curie(LUNG_CANCER))}
    assert not {t for t in triples if t[1] == MAPPED_FROM}
    assert not {t for t in triples if t[0] == t[2]}


def test_entailed_relationships_are_reported_for_the_concept_that_stands_in(standardising) -> None:
    triples = set(
        standardising.relationships(
            subjects=[curie(SOURCE)], predicates=[SITE], include_entailed=True
        )
    )

    assert triples == {
        (curie(SOURCE), SITE, curie(LUNG)),
        (curie(SOURCE), SITE, curie(THORAX_ORGAN)),
    }
