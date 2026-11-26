"""Common parents of seeds, and the descendants of a parent followed through ``Subsumes`` alone."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fixtures.cdm_builders import (
    ancestor,
    classified,
    concept,
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
    PredicateKind,
)
from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.reasoning.phenotypes import phenotype_simplifier
from omop_graph.reasoning.phenotypes.phenotype_simplifier import (
    descendants_exhaustive_subsumes,
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
) = range(600, 611)


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
        relationship("Maps to", "Mapped from"),
        relationship("Mapped from", "Maps to"),
        relationship_class(PredicateKind.IDENTITY, "Exact"),
        classified("Maps to", PredicateKind.IDENTITY, "Exact"),
        classified("Mapped from", PredicateKind.IDENTITY, "Exact"),
        *edge_pair(PARENT, "Maps to", "Mapped from", TARGET_ONE),
        *edge_pair(AMBIGUOUS, "Maps to", "Mapped from", TARGET_ONE),
        *edge_pair(AMBIGUOUS, "Maps to", "Mapped from", TARGET_TWO),
        *edge_pair(SOURCE_SEED, "Maps to", "Mapped from", STANDARD_OF_SEED),
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


@pytest.fixture
def below(monkeypatch: pytest.MonkeyPatch) -> list[set[int]]:
    """Replace the descendant search by one that lists the standard concept of a seed."""
    excluded: list[set[int]] = []

    def fake(kg, root_id, exclude_roots=None):
        excluded.append(set(exclude_roots or ()))
        return {STANDARD_OF_SEED, SEED_ONE, 999}

    monkeypatch.setattr(phenotype_simplifier, "descendants_exhaustive_subsumes", fake)
    return excluded


def test_the_standard_concept_of_a_non_standard_seed_is_excluded_not_counted(kg, below) -> None:
    parents = find_common_parents([SEED_ONE, SOURCE_SEED], kg, min_coverage=1)

    assert STANDARD_OF_SEED in below[0]
    assert parents[TARGET_ONE].pollution == 1


def test_a_parent_is_credited_to_its_one_standard_concept(kg, below) -> None:
    parents = find_common_parents([SEED_ONE, SEED_TWO], kg)

    assert parents[TARGET_ONE].found == {SEED_ONE, SEED_TWO}
    assert PARENT not in parents


def test_a_parent_without_a_standard_concept_stands_for_itself(kg, below) -> None:
    parents = find_common_parents([SEED_ONE, SEED_TWO], kg)

    assert parents[UNMAPPED_PARENT].found == {SEED_ONE, SEED_TWO}


def test_a_parent_with_two_standard_concepts_cannot_be_standardised(kg, below) -> None:
    with pytest.raises(ValueError, match="several standard concepts"):
        find_common_parents([SEED_THREE, SEED_FOUR], kg)


def test_a_seed_with_two_standard_concepts_cannot_be_standardised(kg, below) -> None:
    with pytest.raises(ValueError, match="several standard concepts"):
        find_common_parents([AMBIGUOUS, SEED_ONE], kg)


@pytest.fixture(scope="module")
def hierarchy_kg() -> Iterator[KnowledgeGraph]:
    with hierarchy_cdm() as sessions:
        yield KnowledgeGraph(sessions)


def test_the_descendants_follow_subsumes_and_no_other_relationship(hierarchy_kg) -> None:
    assert descendants_exhaustive_subsumes(hierarchy_kg, DISEASE) == {CANCER, LUNG_CANCER}
    assert SMOKING_RELATED not in descendants_exhaustive_subsumes(hierarchy_kg, DISEASE)


def test_an_excluded_root_is_found_but_not_followed(hierarchy_kg) -> None:
    assert descendants_exhaustive_subsumes(hierarchy_kg, DISEASE, {CANCER}) == {CANCER}
    assert descendants_exhaustive_subsumes(hierarchy_kg, DISEASE, {LUNG_CANCER}) == {
        CANCER,
        LUNG_CANCER,
    }


def test_a_concept_with_nothing_below_it_has_no_descendants(hierarchy_kg) -> None:
    assert descendants_exhaustive_subsumes(hierarchy_kg, LUNG_CANCER) == set()
    assert descendants_exhaustive_subsumes(hierarchy_kg, 999_999) == set()


def test_one_query_reads_each_level(hierarchy_kg, monkeypatch) -> None:
    reads: list[set[int]] = []
    relationships = hierarchy_kg.relationships

    def spy(subjects, **kwargs):
        reads.append(set(subjects))
        return relationships(subjects, **kwargs)

    monkeypatch.setattr(hierarchy_kg, "relationships", spy)

    descendants_exhaustive_subsumes(hierarchy_kg, DISEASE)

    assert reads == [{DISEASE}, {CANCER}, {LUNG_CANCER}]


def test_a_parent_is_counted_by_the_seeds_below_it_and_the_others_it_would_add(
    hierarchy_kg,
) -> None:
    parents = find_common_parents([LUNG_CANCER, SOURCE], hierarchy_kg, min_coverage=1)

    assert set(parents) == {DISEASE, COHORT}
    assert parents[DISEASE].found == {LUNG_CANCER}
    assert parents[DISEASE].descendants == {CANCER, LUNG_CANCER}
    assert parents[DISEASE].coverage == 1
    assert parents[DISEASE].pollution == 1
    assert parents[DISEASE].purity == 0.5
    assert parents[DISEASE].completeness == 0.5
    assert parents[COHORT].pollution == 0
    assert parents[COHORT].purity == 1.0
