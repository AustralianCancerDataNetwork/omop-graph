from __future__ import annotations

from collections.abc import Iterator

import pytest
from fixtures.hierarchy_cdm import CANCER, DISEASE, LUNG_CANCER, RETIRED_PARENT, hierarchy_cdm

from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.graph.model.edges import EdgeFilter


@pytest.fixture(scope="module")
def kg() -> Iterator[KnowledgeGraph]:
    with hierarchy_cdm() as sessions:
        yield KnowledgeGraph(sessions)


def _triples(edges) -> set[tuple[int, str, int]]:
    return {(edge.subject_id, edge.predicate_id, edge.object_id) for edge in edges}


def test_only_the_edges_between_the_two_sets_are_returned(kg) -> None:
    edges = kg.edges_between({LUNG_CANCER, CANCER}, {CANCER, DISEASE})

    assert _triples(edges) == {(LUNG_CANCER, "Is a", CANCER), (CANCER, "Is a", DISEASE)}


def test_the_edge_filter_decides_which_edges_count(kg) -> None:
    active = kg.edges_between({LUNG_CANCER}, {RETIRED_PARENT})
    every = kg.edges_between({LUNG_CANCER}, {RETIRED_PARENT}, EdgeFilter(active_only=False))

    assert active == ()
    assert [edge.invalid_reason for edge in every] == ["D"]


def test_no_subjects_or_no_objects_have_no_edges(kg) -> None:
    assert kg.edges_between((), {CANCER}) == ()
    assert kg.edges_between({LUNG_CANCER}, ()) == ()
