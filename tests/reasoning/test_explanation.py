from __future__ import annotations

from collections.abc import Iterator

import pytest
from fixtures.edge_list import EdgeList
from fixtures.hierarchy_cdm import CANCER, DISEASE, LUNG_CANCER, hierarchy_cdm

from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.graph.model.edges import IS_A_FILTER
from omop_graph.graph.paths import Node, PathStep, find_shortest_paths
from omop_graph.reasoning.explanation import trace_contains_step


@pytest.fixture(scope="module")
def kg() -> Iterator[KnowledgeGraph]:
    with hierarchy_cdm() as sessions:
        yield KnowledgeGraph(sessions)


def _step(subject: int, predicate: str, object_: int) -> PathStep:
    return PathStep(Node(subject, True), predicate, Node(object_, True))


def test_a_step_found_from_the_target_has_the_depth_it_was_read_at() -> None:
    edges = EdgeList([(1, 2), (1, 3), (2, 4), (3, 4), (4, 5)])
    paths, trace = find_shortest_paths(edges.graph, 1, 5, traced=True)
    assert trace is not None

    expansions = {
        (step.subject.concept_id, step.object.concept_id): trace_contains_step(trace, step)
        for path in paths
        for step in path.steps
    }
    depths = {edge: expansion.depth for edge, expansion in expansions.items() if expansion}

    assert len(depths) == len(expansions)
    assert depths == {(1, 2): 0, (1, 3): 0, (2, 4): 1, (3, 4): 1, (4, 5): 0}


def test_every_step_of_a_found_path_is_in_its_trace(kg) -> None:
    paths, trace = find_shortest_paths(
        kg, LUNG_CANCER, DISEASE, edge_filter=IS_A_FILTER, traced=True
    )
    assert trace is not None

    assert all(trace_contains_step(trace, step) for path in paths for step in path.steps)


def test_a_step_needs_its_subject_object_and_relationship_to_match(kg) -> None:
    _, trace = find_shortest_paths(kg, LUNG_CANCER, DISEASE, edge_filter=IS_A_FILTER, traced=True)
    assert trace is not None

    assert trace_contains_step(trace, _step(LUNG_CANCER, "Is a", CANCER)) is not None
    assert trace_contains_step(trace, _step(CANCER, "Is a", LUNG_CANCER)) is None
    assert trace_contains_step(trace, _step(LUNG_CANCER, "Subsumes", CANCER)) is None
    assert trace_contains_step(trace, _step(LUNG_CANCER, "Is a", DISEASE)) is None
