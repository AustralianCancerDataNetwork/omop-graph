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
    SOURCE_LUNG,
    SOURCE_TWO,
    THORAX_ORGAN,
    hierarchy_cdm,
)

from omop_graph.extensions.relationship_classification import HierarchyDirection
from omop_graph.graph.capabilities.concepts import UnknownConceptError
from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.graph.model.scope import DEFAULT_CONCEPT_FILTER


@pytest.fixture(scope="module")
def kg() -> Iterator[KnowledgeGraph]:
    with hierarchy_cdm() as sessions:
        yield KnowledgeGraph(sessions)


def test_a_non_standard_concept_reaches_nothing_unless_it_stands_in(kg) -> None:
    assert kg.closure([SOURCE])[SOURCE] == frozenset()


def test_a_non_standard_concept_reaches_its_standard_concept_and_its_ancestors(kg) -> None:
    reached = kg.closure([SOURCE, SOURCE_LUNG, ORPHAN], standardise=True)

    assert reached[SOURCE] == {LUNG_CANCER, CANCER, SMOKING_RELATED, DISEASE, OUTDATED, OUTER_ROOT}
    assert reached[SOURCE_LUNG] == {LUNG, THORAX_ORGAN}
    assert reached[ORPHAN] == frozenset()


def test_a_concept_with_several_standard_concepts_cannot_be_standardised(kg) -> None:
    with pytest.raises(ValueError, match="several standard concepts") as raised:
        kg.closure([SOURCE_TWO], standardise=True)

    assert str(SOURCE_TWO) in str(raised.value)


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
    reached = kg.closure([SOURCE, SOURCE, ORPHAN], standardise=True)

    assert set(reached) == {SOURCE, ORPHAN}
    assert reached[ORPHAN] == frozenset()


def test_a_walk_from_a_concept_that_does_not_exist_names_every_missing_concept(kg) -> None:
    with pytest.raises(UnknownConceptError) as raised:
        kg.closure([SOURCE, 999999, 999998], standardise=True)

    assert raised.value.concept_ids == (999999, 999998)
