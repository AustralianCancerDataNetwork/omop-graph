"""Common parents of seeds that are standardised through ``Maps to``."""

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
from omop_alchemy.cdm.model.vocabulary import (
    Concept,
    Concept_Ancestor,
    Concept_Relationship,
    Relationship,
    Vocabulary,
)

from omop_graph.extensions.omop_alchemy import (
    PredicateKind,
    RelationshipClass,
    RelationshipMapping,
)
from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.reasoning.phenotypes import phenotype_simplifier
from omop_graph.reasoning.phenotypes.phenotype_simplifier import find_common_parents

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
    models = (
        Concept,
        Relationship,
        Concept_Relationship,
        Concept_Ancestor,
        RelationshipClass,
        RelationshipMapping,
        Vocabulary,
    )
    with sqlite_cdm(models, rows) as sessions:
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
