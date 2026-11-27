"""Common parents of seeds, standardised through ``Maps to`` and counted over the ``Is a`` hierarchy."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fixtures.cdm_builders import (
    ancestor,
    classified,
    concept,
    edge,
    edge_pair,
    relationship,
    relationship_class,
    sqlite_cdm,
)
from fixtures.hierarchy_cdm import (
    CANCER,
    COHORT,
    DISEASE,
    LUNG_CANCER,
    SMOKING_RELATED,
    SOURCE,
    hierarchy_cdm,
)

from omop_graph.extensions.relationship_classification import (
    HierarchyDirection,
    PredicateKind,
)
from omop_graph.graph.capabilities.concepts import UnknownConceptError
from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.reasoning.phenotypes.phenotype_simplifier import (
    find_common_parents,
)

(
    SEED_ONE,
    SEED_TWO,
    SOURCE_SEED,
    PARENT,
    STANDARD_OF_SEED,
    TARGET_ONE,
    TARGET_TWO,
    UNMAPPED_PARENT,
    AMBIGUOUS,
    SEED_THREE,
    SEED_FOUR,
    OTHER_CHILD,
    BELOW_SEED,
) = range(600, 613)


@pytest.fixture(scope="module")
def kg() -> Iterator[KnowledgeGraph]:
    """Seeds under parents that map to one, to no and to two standard concepts."""
    rows = [
        concept(SEED_ONE),
        concept(SEED_TWO),
        concept(SOURCE_SEED, standard_concept=None),
        concept(PARENT, standard_concept=None),
        concept(STANDARD_OF_SEED),
        concept(TARGET_ONE),
        concept(TARGET_TWO),
        concept(UNMAPPED_PARENT, standard_concept=None),
        concept(AMBIGUOUS, standard_concept=None),
        concept(SEED_THREE),
        concept(SEED_FOUR),
        concept(OTHER_CHILD),
        concept(BELOW_SEED),
        relationship("Is a", "Subsumes"),
        relationship("Subsumes", "Is a"),
        relationship("Maps to", "Mapped from"),
        relationship("Mapped from", "Maps to"),
        relationship_class(PredicateKind.IDENTITY, "Exact"),
        relationship_class(PredicateKind.HIERARCHY, "Taxonomic - up"),
        relationship_class(PredicateKind.HIERARCHY, "Taxonomic - down"),
        classified(
            "Is a",
            PredicateKind.HIERARCHY,
            "Taxonomic - up",
            generalises=True,
            direction=HierarchyDirection.UP,
        ),
        classified(
            "Subsumes",
            PredicateKind.HIERARCHY,
            "Taxonomic - down",
            direction=HierarchyDirection.DOWN,
        ),
        classified("Maps to", PredicateKind.IDENTITY, "Exact"),
        classified("Mapped from", PredicateKind.IDENTITY, "Exact"),
        *edge_pair(PARENT, "Maps to", "Mapped from", TARGET_ONE),
        *edge_pair(AMBIGUOUS, "Maps to", "Mapped from", TARGET_ONE),
        *edge_pair(AMBIGUOUS, "Maps to", "Mapped from", TARGET_TWO),
        *edge_pair(SOURCE_SEED, "Maps to", "Mapped from", STANDARD_OF_SEED),
        edge(STANDARD_OF_SEED, "Is a", TARGET_ONE),
        edge(SEED_ONE, "Is a", TARGET_ONE),
        edge(OTHER_CHILD, "Is a", TARGET_ONE),
        edge(BELOW_SEED, "Is a", SEED_ONE),
        ancestor(PARENT, SEED_ONE),
        ancestor(PARENT, SEED_TWO),
        ancestor(PARENT, SOURCE_SEED),
        ancestor(UNMAPPED_PARENT, SEED_ONE),
        ancestor(UNMAPPED_PARENT, SEED_TWO),
        ancestor(AMBIGUOUS, SEED_THREE),
        ancestor(AMBIGUOUS, SEED_FOUR),
    ]
    with sqlite_cdm(rows) as sessions:
        yield KnowledgeGraph(sessions)


def test_the_standard_concept_of_a_non_standard_seed_and_what_lies_below_a_seed_do_not_pollute(
    kg,
) -> None:
    parents = find_common_parents([SEED_ONE, SOURCE_SEED], kg, min_coverage=1)

    assert parents[TARGET_ONE].descendants >= {STANDARD_OF_SEED, SEED_ONE, OTHER_CHILD, BELOW_SEED}
    assert parents[TARGET_ONE].pollution == 1


def test_a_parent_is_credited_to_its_one_standard_concept(kg) -> None:
    parents = find_common_parents([SEED_ONE, SEED_TWO], kg)

    assert parents[TARGET_ONE].found == {SEED_ONE, SEED_TWO}
    assert PARENT not in parents


def test_a_parent_without_a_standard_concept_stands_for_itself(kg) -> None:
    parents = find_common_parents([SEED_ONE, SEED_TWO], kg)

    assert parents[UNMAPPED_PARENT].found == {SEED_ONE, SEED_TWO}


def test_a_parent_with_two_standard_concepts_cannot_be_standardised(kg) -> None:
    with pytest.raises(ValueError, match="several standard concepts"):
        find_common_parents([SEED_THREE, SEED_FOUR], kg)


def test_a_seed_with_two_standard_concepts_cannot_be_standardised(kg) -> None:
    with pytest.raises(ValueError, match="several standard concepts"):
        find_common_parents([AMBIGUOUS, SEED_ONE], kg)


def test_a_seed_that_does_not_exist_is_reported(kg) -> None:
    with pytest.raises(UnknownConceptError):
        find_common_parents([SEED_ONE, 999_999], kg)


@pytest.fixture(scope="module")
def hierarchy_kg() -> Iterator[KnowledgeGraph]:
    with hierarchy_cdm() as sessions:
        yield KnowledgeGraph(sessions)


def test_a_parent_is_counted_by_the_seeds_below_it_and_the_others_it_would_add(
    hierarchy_kg,
) -> None:
    parents = find_common_parents([LUNG_CANCER, SOURCE], hierarchy_kg, min_coverage=1)

    assert set(parents) == {DISEASE, COHORT}
    assert parents[DISEASE].found == {LUNG_CANCER}
    assert parents[DISEASE].descendants == {CANCER, SMOKING_RELATED, LUNG_CANCER}
    assert parents[DISEASE].coverage == 1
    assert parents[DISEASE].pollution == 2
    assert parents[DISEASE].purity == pytest.approx(1 / 3)
    assert parents[DISEASE].completeness == 0.5
    assert parents[COHORT].pollution == 0
    assert parents[COHORT].purity == 1.0
