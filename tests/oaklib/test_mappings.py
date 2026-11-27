"""Mappings served as SSSOM, on a small CDM with every kind of mapping edge."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fixtures.cdm_builders import START
from fixtures.helpers import curie
from fixtures.mappings_cdm import (
    BRAND,
    ICDO,
    INGREDIENT,
    OTHER_SOURCE,
    REPLACED,
    SOURCE_ICD,
    SOURCE_READ,
    SPACED,
    SPLIT_A,
    SPLIT_B,
    SPLIT_SOURCE,
    STANDARD,
    UNIQUE_SOURCE,
    UNIQUE_STANDARD,
    mappings_cdm,
)
from oaklib.datamodels.vocabulary import (
    HAS_DBXREF,
    SKOS_CLOSE_MATCH,
    SKOS_EXACT_MATCH,
    SKOS_RELATED_MATCH,
)
from omop_alchemy.cdm.query import ConceptFilter
from sssom.constants import SEMAPV

from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.oaklib_interface.omop_implementation import OMOPAlchemyImplementation
from omop_graph.oaklib_interface.unsupported import OMOPUnsupportedError


@pytest.fixture(scope="module")
def adapter() -> Iterator[OMOPAlchemyImplementation]:
    with mappings_cdm() as sessions:
        yield OMOPAlchemyImplementation(kg=KnowledgeGraph(sessions))


def summary(mappings) -> set[tuple[str, str, str]]:
    return {(m.subject_id, m.predicate_id, m.object_id) for m in mappings}


def test_a_one_to_one_mapping_is_an_exact_match(adapter) -> None:
    mappings = list(adapter.sssom_mappings([curie(UNIQUE_SOURCE)]))

    assert summary(mappings) == {
        (curie(UNIQUE_SOURCE), SKOS_EXACT_MATCH, curie(UNIQUE_STANDARD)),
        (curie(UNIQUE_SOURCE), HAS_DBXREF, f"ICD10CM:C{UNIQUE_SOURCE}"),
    }
    mapping = next(m for m in mappings if m.predicate_id == SKOS_EXACT_MATCH)
    assert (mapping.subject_source, mapping.object_source) == ("ICD10CM", "SNOMED")
    assert mapping.mapping_justification == SEMAPV.UnspecifiedMatching.value
    assert mapping.mapping_date == START.isoformat()
    assert str(mapping.mapping_cardinality) == "1:1"
    assert mapping.comment == "OMOP relationship Maps to"


def test_a_mapping_that_is_not_one_to_one_is_a_close_match_with_its_cardinality(adapter) -> None:
    def cardinalities(subject: int) -> dict[str, tuple[str, str]]:
        return {
            m.object_id: (m.predicate_id, str(m.mapping_cardinality))
            for m in adapter.sssom_mappings([curie(subject)])
            if m.predicate_id != HAS_DBXREF
        }

    assert cardinalities(SPLIT_SOURCE) == {
        curie(SPLIT_A): (SKOS_CLOSE_MATCH, "n:n"),
        curie(SPLIT_B): (SKOS_CLOSE_MATCH, "1:n"),
    }
    assert cardinalities(OTHER_SOURCE) == {curie(SPLIT_A): (SKOS_CLOSE_MATCH, "n:1")}
    assert cardinalities(SPLIT_B) == {curie(SPLIT_SOURCE): (SKOS_CLOSE_MATCH, "n:1")}
    assert cardinalities(SOURCE_ICD) == {curie(STANDARD): (SKOS_CLOSE_MATCH, "n:1")}


def test_a_standard_concept_is_the_subject_of_every_mapping_to_it(adapter) -> None:
    assert summary(adapter.sssom_mappings([curie(STANDARD)])) == {
        (curie(STANDARD), SKOS_CLOSE_MATCH, curie(SOURCE_ICD)),
        (curie(STANDARD), SKOS_CLOSE_MATCH, curie(SOURCE_READ)),
        (curie(STANDARD), SKOS_CLOSE_MATCH, curie(SPACED)),
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


def test_a_vocabulary_code_gives_the_mappings_of_its_concept(adapter) -> None:
    by_code = summary(adapter.sssom_mappings([f"ICD10CM:C{SOURCE_ICD}"]))

    assert by_code == summary(adapter.sssom_mappings([curie(SOURCE_ICD)]))
    assert (curie(SOURCE_ICD), SKOS_CLOSE_MATCH, curie(STANDARD)) in by_code


def test_a_vocabulary_that_is_not_a_curie_prefix_gets_a_safe_one(adapter) -> None:
    mappings = list(adapter.sssom_mappings([curie(SPACED)]))

    assert (curie(SPACED), HAS_DBXREF, f"Nebraska_Lexicon:C{SPACED}") in summary(mappings)
    assert {m.subject_source for m in mappings} == {"Nebraska_Lexicon"}
    by_curie = summary(mappings)
    assert summary(adapter.sssom_mappings([f"Nebraska_Lexicon:C{SPACED}"])) == by_curie
    assert summary(adapter.sssom_mappings([f"Nebraska Lexicon:C{SPACED}"])) == by_curie
    for source in ("Nebraska_Lexicon", "Nebraska Lexicon"):
        assert summary(adapter.sssom_mappings([curie(STANDARD)], source=source)) == {
            (curie(STANDARD), SKOS_CLOSE_MATCH, curie(SPACED))
        }


def test_mappings_can_be_limited_to_a_vocabulary(adapter) -> None:
    assert summary(adapter.sssom_mappings([curie(STANDARD)], source="Read")) == {
        (curie(STANDARD), SKOS_CLOSE_MATCH, curie(SOURCE_READ))
    }


def test_every_mapping_is_listed_in_both_directions_with_the_cross_references_in_scope(
    adapter,
) -> None:
    every = summary(adapter.sssom_mappings())

    assert (curie(SOURCE_ICD), SKOS_CLOSE_MATCH, curie(STANDARD)) in every
    assert (curie(STANDARD), SKOS_CLOSE_MATCH, curie(SOURCE_ICD)) in every
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
        (SKOS_CLOSE_MATCH, curie(STANDARD)),
        (HAS_DBXREF, f"ICD10CM:C{SOURCE_ICD}"),
    }
    assert (curie(BRAND), SKOS_CLOSE_MATCH, curie(INGREDIENT)) in set(
        adapter.simple_mappings([curie(BRAND)])
    )


def test_the_cross_reference_is_not_an_alias(adapter) -> None:
    assert HAS_DBXREF not in adapter.entity_alias_map(curie(STANDARD))


def test_transitive_mappings_go_from_a_source_concept_to_its_standard_concepts(adapter) -> None:
    mappings = list(adapter.get_transitive_mappings_by_curie(curie(UNIQUE_SOURCE)))

    assert summary(mappings) == {(curie(UNIQUE_SOURCE), SKOS_EXACT_MATCH, curie(UNIQUE_STANDARD))}
    assert {m.subject_source for m in mappings} == {"ICD10CM"}
    assert {m.object_source for m in mappings} == {"SNOMED"}
    assert {str(m.mapping_cardinality) for m in mappings} == {"1:1"}


def test_a_transitive_mapping_that_is_not_one_to_one_is_a_close_match(adapter) -> None:
    mappings = list(adapter.get_transitive_mappings_by_curie(curie(SPLIT_SOURCE)))

    assert summary(mappings) == {
        (curie(SPLIT_SOURCE), SKOS_CLOSE_MATCH, curie(SPLIT_A)),
        (curie(SPLIT_SOURCE), SKOS_CLOSE_MATCH, curie(SPLIT_B)),
    }
    assert {m.object_id: str(m.mapping_cardinality) for m in mappings} == {
        curie(SPLIT_A): "n:n",
        curie(SPLIT_B): "1:n",
    }


def test_transitive_mappings_do_not_join_the_concepts_sharing_a_target(adapter) -> None:
    assert list(adapter.get_transitive_mappings_by_curie(curie(STANDARD))) == []


def test_clusters_are_unsupported(adapter) -> None:
    with pytest.raises(OMOPUnsupportedError, match="get_mapping_clusters"):
        adapter.get_mapping_clusters()


def test_a_source_does_not_widen_the_vocabularies_of_the_adapter(adapter) -> None:
    scoped = OMOPAlchemyImplementation(
        kg=adapter.kg, concept_filter=ConceptFilter(vocabularies=("ICD10CM",), require_active=True)
    )

    cross_references = {
        mapping
        for mapping in summary(scoped.sssom_mappings(None, "SNOMED"))
        if mapping[1] == HAS_DBXREF
    }

    assert cross_references == set()
