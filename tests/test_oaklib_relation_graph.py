"""Entailment over the strict Is a hierarchy, on a world where it differs from concept_ancestor."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from typing import cast

import pytest
import sqlalchemy as sa
from oaklib.datamodels.vocabulary import IS_A
from omop_alchemy.cdm.model.vocabulary.concept import Concept
from omop_alchemy.cdm.model.vocabulary.concept_ancestor import Concept_Ancestor
from omop_alchemy.cdm.model.vocabulary.concept_relationship import Concept_Relationship
from omop_alchemy.cdm.model.vocabulary.relationship import Relationship
from orm_loader.helpers import Base
from sqlalchemy.orm import sessionmaker

from omop_graph.extensions.omop_alchemy import (
    HierarchyDirection,
    PredicateKind,
    RelationshipClass,
    RelationshipMapping,
)
from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.oaklib_interface.omop_implementation import OMOPAlchemyImplementation

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
OUTDATED, OUTER_ROOT = 210, 211
SITE = "OMOP_REL:Has%20finding%20site"
START, END = date(2020, 1, 1), date(2099, 12, 31)


def curie(concept_id: int) -> str:
    return f"OMOP:{concept_id}"


@pytest.fixture(scope="module")
def adapter() -> Iterator[OMOPAlchemyImplementation]:
    engine = sa.create_engine("sqlite+pysqlite:///:memory:").execution_options(
        schema_translate_map={"primary": None, "vocab": None}
    )
    tables = [
        cast(sa.Table, model.__table__)
        for model in (
            Concept,
            Relationship,
            Concept_Ancestor,
            Concept_Relationship,
            RelationshipClass,
            RelationshipMapping,
        )
    ]
    Base.metadata.create_all(engine, tables=tables)

    def concept(concept_id: int, **fields) -> Concept:
        return Concept(
            concept_id=concept_id,
            concept_name=f"Concept {concept_id}",
            concept_code=f"C{concept_id}",
            domain_id="Condition",
            vocabulary_id="SNOMED",
            concept_class_id="Clinical Finding",
            valid_start_date=START,
            valid_end_date=END,
            **{"standard_concept": "S", **fields},
        )

    def relationship(relationship_id: str, reverse: str) -> Relationship:
        return Relationship(
            relationship_id=relationship_id,
            relationship_name=relationship_id,
            is_hierarchical="0",
            defines_ancestry="0",
            reverse_relationship_id=reverse,
            relationship_concept_id=0,
        )

    def edge(
        subject: int, relationship_id: str, object_: int, invalid_reason: str | None = None
    ) -> Concept_Relationship:
        return Concept_Relationship(
            concept_id_1=subject,
            relationship_id=relationship_id,
            concept_id_2=object_,
            valid_start_date=START,
            valid_end_date=END,
            invalid_reason=invalid_reason,
        )

    def ancestor(ancestor_id: int, descendant_id: int) -> Concept_Ancestor:
        return Concept_Ancestor(
            ancestor_concept_id=ancestor_id,
            descendant_concept_id=descendant_id,
            min_levels_of_separation=1,
            max_levels_of_separation=1,
        )

    with sessionmaker(bind=engine)() as session:
        session.add_all(
            [
                *(concept(concept_id) for concept_id in range(DISEASE, LOOP_B + 1)),
                concept(OUTDATED, invalid_reason="U", standard_concept=None),
                concept(OUTER_ROOT),
                relationship("Is a", "Subsumes"),
                relationship("Subsumes", "Is a"),
                relationship("Has finding site", "Finding site of"),
                RelationshipClass(
                    predicate_kind=PredicateKind.HIERARCHY,
                    predicate_subkind="Taxonomic - up",
                    direction=HierarchyDirection.UP,
                    description="up",
                    semantics="up",
                    inference="none",
                ),
                RelationshipClass(
                    predicate_kind=PredicateKind.ATTRIBUTE,
                    predicate_subkind="Site",
                    description="site",
                    semantics="site",
                    inference="none",
                ),
                RelationshipMapping(
                    relationship_id="Is a",
                    predicate_kind=PredicateKind.HIERARCHY,
                    predicate_subkind="Taxonomic - up",
                ),
                RelationshipMapping(
                    relationship_id="Has finding site",
                    predicate_kind=PredicateKind.ATTRIBUTE,
                    predicate_subkind="Site",
                ),
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
                ancestor(DISEASE, LUNG_CANCER),
                ancestor(COHORT, LUNG_CANCER),
            ]
        )
        session.commit()
    try:
        yield OMOPAlchemyImplementation(kg=KnowledgeGraph(sessionmaker(bind=engine)))
    finally:
        engine.dispose()


def test_closure_follows_active_is_a_edges_only(adapter) -> None:
    up = adapter.kg.is_a_closure([LUNG_CANCER, LUNG])

    assert up[LUNG_CANCER] == {CANCER, SMOKING_RELATED, DISEASE, OUTDATED, OUTER_ROOT}
    assert up[LUNG] == {THORAX_ORGAN}


def test_closure_reports_only_concepts_in_scope_but_walks_through_the_rest(adapter) -> None:
    scoped = adapter.kg.is_a_closure([LUNG_CANCER], concept_filter=adapter.concept_filter)

    assert scoped[LUNG_CANCER] == {CANCER, SMOKING_RELATED, DISEASE, OUTER_ROOT}


def test_closure_down_and_cycles_never_reach_the_start(adapter) -> None:
    assert adapter.kg.is_a_closure([DISEASE], HierarchyDirection.DOWN)[DISEASE] == {
        CANCER,
        SMOKING_RELATED,
        LUNG_CANCER,
    }
    assert adapter.kg.is_a_closure([LOOP_A])[LOOP_A] == {LOOP_B}


def test_entailed_outgoing_generalises_the_fillers(adapter) -> None:
    assert set(adapter.entailed_outgoing_relationships(curie(LUNG_CANCER))) == {
        (IS_A, curie(CANCER)),
        (IS_A, curie(SMOKING_RELATED)),
        (IS_A, curie(DISEASE)),
        (IS_A, curie(OUTER_ROOT)),
        (SITE, curie(LUNG)),
        (SITE, curie(THORAX_ORGAN)),
    }


def test_entailed_outgoing_respects_the_predicate_filter(adapter) -> None:
    assert set(adapter.entailed_outgoing_relationships(curie(LUNG_CANCER), [SITE])) == {
        (SITE, curie(LUNG)),
        (SITE, curie(THORAX_ORGAN)),
    }


def test_entailed_outgoing_does_not_inherit_from_ancestors(adapter) -> None:
    assert list(adapter.entailed_outgoing_relationships(curie(CANCER), [SITE])) == []


def test_entailed_incoming_reaches_through_descendants(adapter) -> None:
    assert set(adapter.entailed_incoming_relationships(curie(THORAX_ORGAN))) == {
        (IS_A, curie(LUNG)),
        (SITE, curie(LUNG_CANCER)),
    }
    assert set(adapter.entailed_incoming_relationships(curie(DISEASE))) == {
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
            curie(c) for c in (CANCER, SMOKING_RELATED, DISEASE, OUTER_ROOT, LUNG, THORAX_ORGAN)
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
        (SITE, curie(LUNG_CANCER))
    }


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
