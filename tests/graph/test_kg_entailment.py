from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest
from fixtures.chain_cdm import (
    FILLER,
    FILLER_MIDDLE,
    FILLER_TOP,
    HOLDER_LEAF,
    HOLDER_MIDDLE,
    chain_cdm,
)
from fixtures.chain_cdm import HOLDER as CHAIN_HOLDER
from fixtures.filler_cdm import HOLDER, NON_STANDARD_FILLER, STANDARD_PARENT, filler_cdm
from fixtures.hierarchy_cdm import (
    CANCER,
    DISEASE,
    LUNG,
    LUNG_CANCER,
    ORPHAN,
    OUTER_ROOT,
    SMOKING_RELATED,
    SOURCE,
    SOURCE_LUNG,
    SOURCE_TWO,
    THORAX_ORGAN,
    hierarchy_cdm,
)
from omop_alchemy.cdm.query import ConceptFilter

from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.graph.model.edges import EdgeFilter
from omop_graph.graph.queries import DEFAULT_CONCEPT_FILTER

ENTAILING = EdgeFilter(predicate_ids=frozenset({"Is a", "Has finding site"}))


@pytest.fixture(scope="module")
def kg() -> Iterator[KnowledgeGraph]:
    with hierarchy_cdm() as sessions:
        yield KnowledgeGraph(sessions)


def test_the_relationships_that_generalise_are_the_flagged_ones(kg) -> None:
    assert kg.entailing_relationship_ids == {"Is a", "Has finding site"}


def test_the_default_relationships_leave_out_the_down_direction_of_a_hierarchy(kg) -> None:
    assert kg.default_relationship_ids == {
        "Is a",
        "Has finding site",
        "Finding site of",
        "Maps to",
        "Mapped from",
    }


def test_entailed_relationships_need_subjects_or_objects(kg) -> None:
    with pytest.raises(ValueError, match="subjects or objects"):
        list(kg.entailed_relationships(None, None))


def test_a_relationship_that_generalises_is_entailed_and_the_others_are_stored(kg) -> None:
    triples = list(kg.entailed_relationships((LUNG_CANCER,), None))

    assert (LUNG_CANCER, "Has finding site", THORAX_ORGAN) in triples
    assert (LUNG_CANCER, "Is a", DISEASE) in triples
    assert (LUNG_CANCER, "Mapped from", SOURCE) in triples
    assert len(triples) == len(set(triples))
    assert not {t for t in triples if t[1] == "Mapped from" and t[2] != SOURCE}


def test_the_stored_relationships_come_before_the_entailed_ones(kg) -> None:
    triples = list(kg.entailed_relationships((LUNG_CANCER,), None))
    stored = [
        i for i, (_, relationship_id, _) in enumerate(triples) if relationship_id == "Mapped from"
    ]

    assert stored == [0]


def test_a_filter_selecting_no_relationship_gives_nothing(kg) -> None:
    nothing = EdgeFilter(predicate_ids=frozenset())

    assert list(kg.entailed_relationships((LUNG_CANCER,), None, nothing)) == []


def test_entailed_relationships_into_objects_reach_the_subtypes(kg) -> None:
    triples = set(kg.entailed_relationships(None, (THORAX_ORGAN,), ENTAILING))

    assert (LUNG_CANCER, "Has finding site", THORAX_ORGAN) in triples
    assert (LUNG, "Is a", THORAX_ORGAN) in triples


def test_a_non_standard_concept_has_no_entailed_relationships_of_its_own(kg) -> None:
    assert list(kg.entailed_relationships((SOURCE,), None, ENTAILING)) == []


def test_standardise_lends_the_entailed_relationships_of_the_standard_concepts(kg) -> None:
    triples = set(
        kg.entailed_relationships(
            (SOURCE,), None, ENTAILING, DEFAULT_CONCEPT_FILTER, standardise=True
        )
    )

    assert triples == {
        (SOURCE, "Is a", CANCER),
        (SOURCE, "Is a", DISEASE),
        (SOURCE, "Is a", SMOKING_RELATED),
        (SOURCE, "Is a", OUTER_ROOT),
        (SOURCE, "Has finding site", LUNG),
        (SOURCE, "Has finding site", THORAX_ORGAN),
    }


def test_standardise_reports_incoming_relationships_for_the_concept_standing_in(kg) -> None:
    triples = set(kg.entailed_relationships(None, (SOURCE_LUNG,), ENTAILING, standardise=True))

    assert (LUNG_CANCER, "Has finding site", SOURCE_LUNG) in triples
    assert {t[2] for t in triples} == {SOURCE_LUNG}


def test_a_concept_with_several_standard_concepts_cannot_be_standardised_for_entailment(kg) -> None:
    with pytest.raises(ValueError, match="several standard concepts"):
        list(kg.entailed_relationships((SOURCE_TWO,), None, ENTAILING, standardise=True))
    with pytest.raises(ValueError, match="several standard concepts"):
        list(kg.entailed_relationships(None, (SOURCE_TWO,), ENTAILING, standardise=True))


def test_standardise_does_not_lend_stored_relationships(kg) -> None:
    triples = set(kg.entailed_relationships((SOURCE,), None, standardise=True))

    assert {t for t in triples if t[1] in {"Mapped from", "Maps to"}} == {
        (SOURCE, "Maps to", LUNG_CANCER)
    }


def test_a_concept_with_no_standard_concept_has_nothing_to_lend(kg) -> None:
    assert list(kg.entailed_relationships((ORPHAN,), None, ENTAILING, standardise=True)) == []


def test_streaming_every_concept_gives_what_asking_for_all_of_them_gives(kg) -> None:
    concept_ids = tuple(kg.entities(DEFAULT_CONCEPT_FILTER))

    assert set(kg.all_entailed_relationships(ENTAILING)) == set(
        kg.entailed_relationships(concept_ids, None, ENTAILING, DEFAULT_CONCEPT_FILTER)
    )


def test_streaming_with_standardise_reports_for_each_concept_in_scope(kg) -> None:
    streamed = set(kg.all_entailed_relationships(ENTAILING, standardise=True))

    assert (SOURCE, "Has finding site", LUNG) not in streamed


@pytest.fixture(scope="module")
def filler_kg() -> Iterator[KnowledgeGraph]:
    with filler_cdm() as sessions:
        yield KnowledgeGraph(sessions)


WITHOUT_PARENT = ConceptFilter(concept_ids=(HOLDER, NON_STANDARD_FILLER))
"""A scope that excludes STANDARD_PARENT, which the filler of HOLDER is a kind of."""

ENTAILED_TO_PARENT = (HOLDER, "Has finding site", STANDARD_PARENT)


def test_a_requested_object_the_scope_excludes_is_not_entailed_from_either_end(filler_kg) -> None:
    from_subject = set(
        filler_kg.entailed_relationships(
            (HOLDER,), (STANDARD_PARENT,), concept_filter=WITHOUT_PARENT
        )
    )
    into_object = set(
        filler_kg.entailed_relationships(None, (STANDARD_PARENT,), concept_filter=WITHOUT_PARENT)
    )

    assert from_subject == set()
    assert into_object == {(NON_STANDARD_FILLER, "Is a", STANDARD_PARENT)}


def test_a_requested_object_the_scope_includes_is_entailed_from_either_end(filler_kg) -> None:
    everything = ConceptFilter()

    from_subject = set(
        filler_kg.entailed_relationships((HOLDER,), (STANDARD_PARENT,), concept_filter=everything)
    )
    into_object = set(
        filler_kg.entailed_relationships(None, (STANDARD_PARENT,), concept_filter=everything)
    )

    assert ENTAILED_TO_PARENT in from_subject
    assert ENTAILED_TO_PARENT in into_object


def test_a_relationship_asserted_to_an_excluded_object_is_still_reported(filler_kg) -> None:
    scope = ConceptFilter(concept_ids=(HOLDER,))
    asserted = (HOLDER, "Has finding site", NON_STANDARD_FILLER)

    from_subject = set(
        filler_kg.entailed_relationships((HOLDER,), (NON_STANDARD_FILLER,), concept_filter=scope)
    )
    into_object = set(
        filler_kg.entailed_relationships(None, (NON_STANDARD_FILLER,), concept_filter=scope)
    )

    assert asserted in from_subject
    assert asserted in into_object


def test_the_requested_objects_a_scope_excludes_are_named_in_one_warning(
    filler_kg, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        list(
            filler_kg.entailed_relationships(
                None, (STANDARD_PARENT, NON_STANDARD_FILLER), concept_filter=WITHOUT_PARENT
            )
        )

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert str(STANDARD_PARENT) in warnings[0].getMessage()
    assert str(NON_STANDARD_FILLER) not in warnings[0].getMessage()


def test_nothing_is_warned_when_the_scope_keeps_every_requested_object(
    filler_kg, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        list(
            filler_kg.entailed_relationships(
                None, (STANDARD_PARENT,), concept_filter=ConceptFilter()
            )
        )
        list(filler_kg.entailed_relationships((HOLDER,), None, concept_filter=WITHOUT_PARENT))

    assert not caplog.records


def test_a_concept_the_walk_finds_and_the_scope_removes_is_silent(
    filler_kg, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        triples = set(
            filler_kg.entailed_relationships((HOLDER,), None, concept_filter=WITHOUT_PARENT)
        )

    assert ENTAILED_TO_PARENT not in triples
    assert not caplog.records


@pytest.fixture(scope="module")
def chain_kg() -> Iterator[KnowledgeGraph]:
    with chain_cdm() as sessions:
        yield KnowledgeGraph(sessions)


def test_the_walk_up_from_a_filler_passes_through_a_concept_the_scope_excludes(chain_kg) -> None:
    triples = set(
        chain_kg.entailed_relationships((CHAIN_HOLDER,), None, ENTAILING, DEFAULT_CONCEPT_FILTER)
    )

    assert (CHAIN_HOLDER, "Has finding site", FILLER) in triples
    assert (CHAIN_HOLDER, "Has finding site", FILLER_TOP) in triples
    assert (CHAIN_HOLDER, "Has finding site", FILLER_MIDDLE) not in triples


def test_the_walk_down_from_a_holder_passes_through_a_concept_the_scope_excludes(
    chain_kg,
) -> None:
    triples = set(
        chain_kg.entailed_relationships(None, (FILLER_TOP,), ENTAILING, DEFAULT_CONCEPT_FILTER)
    )

    assert (CHAIN_HOLDER, "Has finding site", FILLER_TOP) in triples
    assert (HOLDER_LEAF, "Has finding site", FILLER_TOP) in triples
    assert not {t for t in triples if HOLDER_MIDDLE in (t[0], t[2])}


def test_a_subtype_inherits_through_a_concept_the_scope_excludes(chain_kg) -> None:
    triples = set(
        chain_kg.entailed_relationships((HOLDER_LEAF,), None, ENTAILING, DEFAULT_CONCEPT_FILTER)
    )

    assert (HOLDER_LEAF, "Has finding site", FILLER) in triples
    assert (HOLDER_LEAF, "Has finding site", FILLER_TOP) in triples


def test_standardise_names_no_objects_in_a_warning_since_it_reports_for_the_stand_in(
    kg, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        triples = set(
            kg.entailed_relationships(
                None, (SOURCE_LUNG,), ENTAILING, DEFAULT_CONCEPT_FILTER, standardise=True
            )
        )

    assert (LUNG_CANCER, "Has finding site", SOURCE_LUNG) in triples
    assert not caplog.records


def test_a_relationship_down_the_hierarchy_is_reported_as_stored(kg) -> None:
    subsumes = EdgeFilter(predicate_ids=frozenset({"Subsumes"}))

    assert list(kg.entailed_relationships((DISEASE,), None, subsumes)) == [
        (DISEASE, "Subsumes", CANCER)
    ]
    assert list(kg.entailed_relationships(None, (CANCER,), subsumes)) == [
        (DISEASE, "Subsumes", CANCER)
    ]
