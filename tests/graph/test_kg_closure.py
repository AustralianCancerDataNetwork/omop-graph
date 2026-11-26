from __future__ import annotations

from collections.abc import Iterator

import pytest
from fixtures.hierarchy_cdm import (
    CANCER,
    DISEASE,
    LUNG,
    LUNG_CANCER,
    ORPHAN,
    OUTDATED,
    OUTER_ROOT,
    SMOKING_RELATED,
    SOURCE,
    SOURCE_TWO,
    THORAX_ORGAN,
    hierarchy_cdm,
)

from omop_graph.extensions.omop_alchemy import HierarchyDirection
from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.graph.queries import DEFAULT_CONCEPT_FILTER


@pytest.fixture(scope="module")
def kg() -> Iterator[KnowledgeGraph]:
    with hierarchy_cdm() as sessions:
        yield KnowledgeGraph(sessions)


def test_a_non_standard_concept_reaches_nothing_unless_it_stands_in(kg) -> None:
    assert kg.closure([SOURCE])[SOURCE] == frozenset()


def test_a_non_standard_concept_reaches_its_standard_concepts_and_their_ancestors(kg) -> None:
    reached = kg.closure([SOURCE, SOURCE_TWO, ORPHAN], standardise=True)

    assert reached[SOURCE] == {LUNG_CANCER, CANCER, SMOKING_RELATED, DISEASE, OUTDATED, OUTER_ROOT}
    assert reached[SOURCE_TWO] == reached[SOURCE] | {LUNG, THORAX_ORGAN}
    assert reached[ORPHAN] == frozenset()


def test_a_standard_concept_reaches_the_same_with_and_without_standardise(kg) -> None:
    assert kg.closure([LUNG_CANCER, LUNG], standardise=True) == kg.closure([LUNG_CANCER, LUNG])


def test_standardise_applies_the_concept_filter_to_the_walk(kg) -> None:
    scoped = kg.closure([SOURCE], concept_filter=DEFAULT_CONCEPT_FILTER, standardise=True)

    assert OUTDATED not in scoped[SOURCE]
    assert scoped[SOURCE] == {LUNG_CANCER, CANCER, SMOKING_RELATED, DISEASE, OUTER_ROOT}


def test_standardise_walks_down_from_the_standard_concepts(kg) -> None:
    reached = kg.closure([SOURCE], HierarchyDirection.DOWN, standardise=True)

    assert reached[SOURCE] == {LUNG_CANCER}


def test_standardise_keeps_every_requested_concept_and_none_twice(kg) -> None:
    reached = kg.closure([SOURCE, SOURCE, 999999], standardise=True)

    assert set(reached) == {SOURCE, 999999}
    assert reached[999999] == frozenset()
