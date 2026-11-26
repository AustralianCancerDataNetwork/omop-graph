from __future__ import annotations

from collections.abc import Iterator

import pytest
from fixtures.hierarchy_cdm import (
    CANCER,
    COHORT,
    DISEASE,
    LUNG_CANCER,
    OUTDATED,
    SMOKING_RELATED,
    hierarchy_cdm,
)
from fixtures.mappings_cdm import (
    BRAND,
    ICDO,
    INGREDIENT,
    REPLACED,
    SOURCE_ICD,
    SOURCE_READ,
    SPACED,
    STANDARD,
    mappings_cdm,
)

from omop_graph.extensions.omop_alchemy import PredicateKind, PredicateSubkind
from omop_graph.graph.capabilities.concepts import UnknownConceptError
from omop_graph.graph.edges import MappingStrength
from omop_graph.graph.kg import KnowledgeGraph


@pytest.fixture(scope="module")
def kg() -> Iterator[KnowledgeGraph]:
    with mappings_cdm() as sessions:
        yield KnowledgeGraph(sessions)


@pytest.fixture(scope="module")
def hierarchy_kg() -> Iterator[KnowledgeGraph]:
    with hierarchy_cdm() as sessions:
        yield KnowledgeGraph(sessions)


@pytest.mark.parametrize(
    ("kind", "name", "strength"),
    [
        (PredicateKind.IDENTITY, "Exact equivalence", MappingStrength.EXACT),
        (PredicateKind.IDENTITY, "Near equivalence", MappingStrength.CLOSE),
        (PredicateKind.ASSOCIATION, "Mapping", MappingStrength.RELATED),
        (PredicateKind.IDENTITY, "Version lineage", None),
        (PredicateKind.ATTRIBUTE, "Mapping", None),
    ],
)
def test_a_subkind_is_a_mapping_of_one_strength_or_none(kind, name, strength) -> None:
    assert MappingStrength.of(PredicateSubkind(kind, name)) is strength


def test_the_predicate_reports_its_mapping_strength(kg) -> None:
    assert kg.predicate("Maps to").mapping_strength is MappingStrength.EXACT
    assert kg.predicate("Concept replaced by").mapping_strength is None


def test_the_mapping_predicates_are_those_classified_as_mappings(kg) -> None:
    assert kg.mapping_predicates() == {
        "Maps to": MappingStrength.EXACT,
        "Mapped from": MappingStrength.EXACT,
        "Brand name of": MappingStrength.CLOSE,
        "Has brand name": MappingStrength.CLOSE,
        "ICDO - SNOMED": MappingStrength.RELATED,
        "SNOMED - ICDO": MappingStrength.RELATED,
    }


def test_a_non_standard_concept_maps_to_the_standard_concepts_by_exact_mappings(kg) -> None:
    assert kg.standard_mappings([SOURCE_ICD, SOURCE_READ, SPACED]) == {
        SOURCE_ICD: (STANDARD,),
        SOURCE_READ: (STANDARD,),
        SPACED: (STANDARD,),
    }


def test_a_standard_concept_and_an_unmapped_one_map_to_nothing(kg) -> None:
    assert kg.standard_mappings([STANDARD, INGREDIENT]) == {STANDARD: (), INGREDIENT: ()}


def test_mappings_are_not_worked_out_for_a_concept_that_does_not_exist(kg) -> None:
    with pytest.raises(UnknownConceptError) as raised:
        kg.standard_mappings([STANDARD, 999999, 999998])

    assert raised.value.concept_ids == (999999, 999998)


def test_the_strength_chooses_which_mappings_are_followed(kg) -> None:
    assert kg.standard_mappings([BRAND]) == {BRAND: ()}
    assert kg.standard_mappings([BRAND], MappingStrength.CLOSE) == {BRAND: (INGREDIENT,)}
    assert kg.standard_mappings([ICDO], MappingStrength.RELATED) == {ICDO: (STANDARD,)}


def test_a_concept_has_the_concepts_that_replace_it(kg) -> None:
    assert kg.replacements([REPLACED, STANDARD, 999999]) == {
        REPLACED: (STANDARD,),
        STANDARD: (),
        999999: (),
    }


def test_no_concepts_have_no_replacements(kg) -> None:
    assert kg.replacements([]) == {}


def test_parents_are_the_concept_ancestor_levels_by_default(hierarchy_kg) -> None:
    assert set(hierarchy_kg.parents(LUNG_CANCER)) == {DISEASE, COHORT}


def test_the_is_a_parents_are_the_targets_of_active_is_a_edges(hierarchy_kg) -> None:
    assert set(hierarchy_kg.parents(LUNG_CANCER, isa_only=True)) == {CANCER, SMOKING_RELATED}
    assert set(hierarchy_kg.parents(CANCER, isa_only=True)) == {DISEASE, OUTDATED}
