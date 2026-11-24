"""Mappings served as SSSOM, on a small world with every kind of mapping edge."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from typing import cast

import pytest
import sqlalchemy as sa
from oaklib.datamodels.vocabulary import (
    HAS_DBXREF,
    SKOS_CLOSE_MATCH,
    SKOS_EXACT_MATCH,
    SKOS_RELATED_MATCH,
)
from omop_alchemy.cdm.model.vocabulary.concept import Concept
from omop_alchemy.cdm.model.vocabulary.concept_relationship import Concept_Relationship
from omop_alchemy.cdm.model.vocabulary.concept_synonym import Concept_Synonym
from omop_alchemy.cdm.model.vocabulary.relationship import Relationship
from omop_alchemy.cdm.model.vocabulary.vocabulary import Vocabulary
from orm_loader.helpers import Base
from sqlalchemy.orm import sessionmaker
from sssom.constants import SEMAPV

from omop_graph.extensions.omop_alchemy import (
    PredicateKind,
    RelationshipClass,
    RelationshipMapping,
)
from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.oaklib_interface.omop_implementation import OMOPAlchemyImplementation
from omop_graph.oaklib_interface.unsupported import OMOPUnsupportedError

STANDARD, SOURCE_ICD, SOURCE_READ, BRAND, INGREDIENT, REPLACED, ICDO, SPACED = range(300, 308)
STARTED = date(2020, 1, 1)
PAIRS = (
    ("Maps to", "Mapped from", PredicateKind.IDENTITY, "Exact equivalence"),
    ("Brand name of", "Has brand name", PredicateKind.IDENTITY, "Near equivalence"),
    ("ICDO - SNOMED", "SNOMED - ICDO", PredicateKind.ASSOCIATION, "Mapping"),
    ("Concept replaced by", "Concept replaces", PredicateKind.IDENTITY, "Version lineage"),
)


def curie(concept_id: int) -> str:
    return f"OMOP:{concept_id}"


@pytest.fixture(scope="module")
def adapter() -> Iterator[OMOPAlchemyImplementation]:
    engine = sa.create_engine("sqlite+pysqlite:///:memory:").execution_options(
        schema_translate_map={"primary": None, "vocab": None}
    )
    Base.metadata.create_all(
        engine,
        tables=[
            cast(sa.Table, model.__table__)
            for model in (
                Concept,
                Relationship,
                Concept_Relationship,
                Concept_Synonym,
                Vocabulary,
                RelationshipClass,
                RelationshipMapping,
            )
        ],
    )

    def concept(concept_id: int, vocabulary: str, standard: str | None, **fields) -> Concept:
        return Concept(
            concept_id=concept_id,
            concept_name=f"Concept {concept_id}",
            concept_code=f"C{concept_id}",
            domain_id="Condition",
            vocabulary_id=vocabulary,
            concept_class_id="Clinical Finding",
            standard_concept=standard,
            valid_start_date=STARTED,
            valid_end_date=date(2099, 12, 31),
            **fields,
        )

    def edges(
        subject: int, relationship_id: str, reverse: str, object_: int, invalid: str | None = None
    ) -> list[Concept_Relationship]:
        def edge(s: int, r: str, o: int) -> Concept_Relationship:
            return Concept_Relationship(
                concept_id_1=s,
                relationship_id=r,
                concept_id_2=o,
                valid_start_date=STARTED,
                valid_end_date=date(2099, 12, 31),
                invalid_reason=invalid,
            )

        return [edge(subject, relationship_id, object_), edge(object_, reverse, subject)]

    with sessionmaker(bind=engine)() as session:
        session.add_all(
            [
                concept(STANDARD, "SNOMED", "S"),
                concept(SOURCE_ICD, "ICD10CM", None),
                concept(SOURCE_READ, "Read", None),
                concept(BRAND, "RxNorm", None),
                concept(INGREDIENT, "RxNorm", "S"),
                concept(REPLACED, "SNOMED", None, invalid_reason="U"),
                concept(ICDO, "ICDO3", None),
                concept(SPACED, "Nebraska Lexicon", None),
                *(
                    Vocabulary(
                        vocabulary_id=vocabulary,
                        vocabulary_name=vocabulary,
                        vocabulary_concept_id=index,
                    )
                    for index, vocabulary in enumerate(
                        ("SNOMED", "ICD10CM", "Read", "RxNorm", "ICDO3", "Nebraska Lexicon")
                    )
                ),
                *(
                    Relationship(
                        relationship_id=r,
                        relationship_name=r,
                        is_hierarchical="0",
                        defines_ancestry="0",
                        reverse_relationship_id=reverse,
                        relationship_concept_id=0,
                    )
                    for pair in PAIRS
                    for r, reverse in ((pair[0], pair[1]), (pair[1], pair[0]))
                ),
                *(
                    RelationshipClass(
                        predicate_kind=kind,
                        predicate_subkind=subkind,
                        description=subkind,
                        semantics=subkind,
                        inference="none",
                    )
                    for kind, subkind in {(p[2], p[3]) for p in PAIRS}
                ),
                *(
                    RelationshipMapping(
                        relationship_id=r, predicate_kind=kind, predicate_subkind=subkind
                    )
                    for forward, backward, kind, subkind in PAIRS
                    for r in (forward, backward)
                ),
                *edges(SOURCE_ICD, "Maps to", "Mapped from", STANDARD),
                *edges(SOURCE_READ, "Maps to", "Mapped from", STANDARD),
                *edges(SOURCE_READ, "Maps to", "Mapped from", REPLACED, invalid="D"),
                *edges(STANDARD, "Maps to", "Mapped from", STANDARD),
                *edges(BRAND, "Brand name of", "Has brand name", INGREDIENT),
                *edges(ICDO, "ICDO - SNOMED", "SNOMED - ICDO", STANDARD),
                *edges(REPLACED, "Concept replaced by", "Concept replaces", STANDARD),
                *edges(SPACED, "Maps to", "Mapped from", STANDARD),
            ]
        )
        session.commit()
    try:
        yield OMOPAlchemyImplementation(kg=KnowledgeGraph(sessionmaker(bind=engine)))
    finally:
        engine.dispose()


def summary(mappings) -> set[tuple[str, str, str]]:
    return {(m.subject_id, m.predicate_id, m.object_id) for m in mappings}


def test_a_source_concept_maps_to_its_standard_concept(adapter) -> None:
    mappings = list(adapter.sssom_mappings([curie(SOURCE_ICD)]))

    assert summary(mappings) == {
        (curie(SOURCE_ICD), SKOS_EXACT_MATCH, curie(STANDARD)),
        (curie(SOURCE_ICD), HAS_DBXREF, f"ICD10CM:C{SOURCE_ICD}"),
    }
    mapping = next(m for m in mappings if m.predicate_id == SKOS_EXACT_MATCH)
    assert (mapping.subject_source, mapping.object_source) == ("ICD10CM", "SNOMED")
    assert mapping.mapping_justification == SEMAPV.UnspecifiedMatching.value
    assert mapping.mapping_date == STARTED.isoformat()
    assert mapping.comment == "OMOP relationship Maps to"


def test_a_standard_concept_is_the_subject_of_every_mapping_to_it(adapter) -> None:
    assert summary(adapter.sssom_mappings([curie(STANDARD)])) == {
        (curie(STANDARD), SKOS_EXACT_MATCH, curie(SOURCE_ICD)),
        (curie(STANDARD), SKOS_EXACT_MATCH, curie(SOURCE_READ)),
        (curie(STANDARD), SKOS_EXACT_MATCH, curie(SPACED)),
        (curie(STANDARD), SKOS_RELATED_MATCH, curie(ICDO)),
        (curie(STANDARD), HAS_DBXREF, f"SNOMED:C{STANDARD}"),
    }


def test_each_kind_of_relationship_has_its_own_predicate(adapter) -> None:
    assert summary(adapter.sssom_mappings([curie(BRAND), curie(ICDO)])) >= {
        (curie(BRAND), SKOS_CLOSE_MATCH, curie(INGREDIENT)),
        (curie(ICDO), SKOS_RELATED_MATCH, curie(STANDARD)),
    }


def test_version_lineage_self_maps_and_deprecated_edges_are_not_mappings(adapter) -> None:
    every = summary(adapter.sssom_mappings())

    assert not any(curie(REPLACED) in (s, o) for s, _, o in every if _ != HAS_DBXREF)
    assert (curie(STANDARD), SKOS_EXACT_MATCH, curie(STANDARD)) not in every
    assert (curie(SOURCE_READ), SKOS_EXACT_MATCH, curie(REPLACED)) not in every


def test_a_vocabulary_code_gives_only_the_cross_reference(adapter) -> None:
    assert summary(adapter.sssom_mappings([f"ICD10CM:C{SOURCE_ICD}"])) == {
        (curie(SOURCE_ICD), HAS_DBXREF, f"ICD10CM:C{SOURCE_ICD}")
    }


def test_a_vocabulary_that_is_not_a_curie_prefix_gets_a_safe_one(adapter) -> None:
    mappings = list(adapter.sssom_mappings([curie(SPACED)]))

    assert (curie(SPACED), HAS_DBXREF, f"Nebraska_Lexicon:C{SPACED}") in summary(mappings)
    assert {m.subject_source for m in mappings} == {"Nebraska_Lexicon"}
    assert summary(adapter.sssom_mappings([f"Nebraska_Lexicon:C{SPACED}"])) == {
        (curie(SPACED), HAS_DBXREF, f"Nebraska_Lexicon:C{SPACED}")
    }
    assert summary(adapter.sssom_mappings([f"Nebraska Lexicon:C{SPACED}"])) == {
        (curie(SPACED), HAS_DBXREF, f"Nebraska_Lexicon:C{SPACED}")
    }
    for source in ("Nebraska_Lexicon", "Nebraska Lexicon"):
        assert summary(adapter.sssom_mappings([curie(STANDARD)], source=source)) == {
            (curie(STANDARD), SKOS_EXACT_MATCH, curie(SPACED))
        }


def test_mappings_can_be_limited_to_a_vocabulary(adapter) -> None:
    assert summary(adapter.sssom_mappings([curie(STANDARD)], source="Read")) == {
        (curie(STANDARD), SKOS_EXACT_MATCH, curie(SOURCE_READ))
    }


def test_every_mapping_is_listed_in_both_directions_with_the_cross_references_in_scope(
    adapter,
) -> None:
    every = summary(adapter.sssom_mappings())

    assert (curie(SOURCE_ICD), SKOS_EXACT_MATCH, curie(STANDARD)) in every
    assert (curie(STANDARD), SKOS_EXACT_MATCH, curie(SOURCE_ICD)) in every
    assert (curie(STANDARD), HAS_DBXREF, f"SNOMED:C{STANDARD}") in every
    assert (curie(SOURCE_ICD), HAS_DBXREF, f"ICD10CM:C{SOURCE_ICD}") not in every


def test_mappings_by_source(adapter) -> None:
    assert summary(adapter.sssom_mappings_by_source("RxNorm")) == {
        (curie(BRAND), SKOS_CLOSE_MATCH, curie(INGREDIENT)),
        (curie(INGREDIENT), SKOS_CLOSE_MATCH, curie(BRAND)),
        (curie(INGREDIENT), HAS_DBXREF, f"RxNorm:C{INGREDIENT}"),
    }


def test_simple_mappings(adapter) -> None:
    assert set(adapter.simple_mappings_by_curie(curie(SOURCE_ICD))) == {
        (SKOS_EXACT_MATCH, curie(STANDARD)),
        (HAS_DBXREF, f"ICD10CM:C{SOURCE_ICD}"),
    }
    assert (curie(BRAND), SKOS_CLOSE_MATCH, curie(INGREDIENT)) in set(
        adapter.simple_mappings([curie(BRAND)])
    )


def test_the_cross_reference_is_not_an_alias(adapter) -> None:
    assert HAS_DBXREF not in adapter.entity_alias_map(curie(STANDARD))


def test_transitive_mappings_follow_exact_equivalence_in_both_directions(adapter) -> None:
    mappings = list(adapter.get_transitive_mappings_by_curie(curie(SOURCE_ICD)))

    assert summary(mappings) == {
        (curie(SOURCE_ICD), SKOS_EXACT_MATCH, curie(STANDARD)),
        (curie(SOURCE_ICD), SKOS_EXACT_MATCH, curie(SOURCE_READ)),
        (curie(SOURCE_ICD), SKOS_EXACT_MATCH, curie(SPACED)),
    }
    assert {m.mapping_justification for m in mappings} == {SEMAPV.MappingChaining.value}


def test_clusters_are_unsupported(adapter) -> None:
    with pytest.raises(OMOPUnsupportedError, match="get_mapping_clusters"):
        adapter.get_mapping_clusters()
