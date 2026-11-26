from __future__ import annotations

import random
from collections.abc import Iterable, Iterator
from types import SimpleNamespace
from typing import cast

import pytest
from fixtures.helpers import UP
from fixtures.hierarchy_cdm import (
    CANCER,
    DISEASE,
    LUNG,
    LUNG_CANCER,
    ORPHAN,
    OUTDATED,
    OUTER_ROOT,
    SMOKING_RELATED,
    hierarchy_cdm,
)

from omop_graph.graph import paths as paths_module
from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.graph.model.edges import EdgeFilter
from omop_graph.graph.paths import find_shortest_paths, find_shortest_paths_batch


@pytest.fixture(scope="module")
def kg() -> Iterator[KnowledgeGraph]:
    with hierarchy_cdm() as sessions:
        yield KnowledgeGraph(sessions)


class EdgeList:
    """A graph of the given edges that answers the calls a path search makes, and counts them."""

    def __init__(self, edges: Iterable[tuple[int, int]]) -> None:
        self.edge_list = tuple(edges)
        self.concepts = sorted({concept for edge in self.edge_list for concept in edge})
        self.read_from: list[int] = []

    def edges(self, concept_ids, direction, edge_filter=None):
        ids = {concept_ids} if isinstance(concept_ids, int) else set(concept_ids)
        self.read_from.extend(sorted(ids))
        return tuple(
            SimpleNamespace(subject_id=subject, object_id=object_, predicate_id="Is a")
            for subject, object_ in self.edge_list
            if (subject if direction == "out" else object_) in ids
        )

    def concept_views(self, concept_ids, sort=True):
        return tuple(
            SimpleNamespace(concept_id=concept_id, standard_concept=concept_id % 2 == 1)
            for concept_id in concept_ids
        )

    @property
    def graph(self) -> KnowledgeGraph:
        """This list as the graph a search takes, which only needs the calls it answers."""
        return cast(KnowledgeGraph, self)

    def shortest_routes(self, source: int, target: int) -> list[tuple[int, ...]]:
        """Every shortest route, by enumerating the routes breadth first."""
        found: list[tuple[int, ...]] = []
        routes = [(source,)]
        while routes and not found:
            routes = [
                route + (object_,)
                for route in routes
                for subject, object_ in self.edge_list
                if subject == route[-1] and object_ not in route
            ]
            found = [route for route in routes if route[-1] == target]
        return sorted(found)


@pytest.fixture(params=["bidirectional", "batched"])
def search_in(request):
    """Both searches, called alike on a graph and returning the paths."""
    searches = {
        "bidirectional": lambda graph, *args, **kwargs: find_shortest_paths(graph, *args, **kwargs)[
            0
        ],
        "batched": find_shortest_paths_batch,
    }
    return searches[request.param]


@pytest.fixture
def search(search_in, kg):
    """Both searches on the hierarchy CDM."""
    return lambda *args, **kwargs: search_in(kg, *args, **kwargs)


def _routes(paths) -> list[tuple[int, ...]]:
    return sorted(path.nodes() for path in paths)


def test_a_concept_is_its_own_empty_path(search) -> None:
    paths = search(LUNG_CANCER, LUNG_CANCER)

    assert len(paths) == 1
    assert len(paths[0]) == 0


def test_every_shortest_path_is_found(search) -> None:
    paths = search(LUNG_CANCER, DISEASE, edge_filter=UP)

    assert _routes(paths) == [
        (LUNG_CANCER, CANCER, DISEASE),
        (LUNG_CANCER, SMOKING_RELATED, DISEASE),
    ]
    assert {step.predicate for path in paths for step in path.steps} == {"Is a"}


def test_the_edge_filter_decides_which_edges_are_followed(search) -> None:
    only_sites = EdgeFilter(predicate_ids=frozenset({"Has finding site"}))

    assert _routes(search(LUNG_CANCER, LUNG, edge_filter=only_sites)) == [(LUNG_CANCER, LUNG)]
    assert search(LUNG_CANCER, DISEASE, edge_filter=only_sites) == []


def test_max_depth_bounds_the_search_from_each_end(search) -> None:
    assert search(LUNG_CANCER, DISEASE, edge_filter=UP, max_depth=0) == []
    assert search(LUNG_CANCER, OUTER_ROOT, edge_filter=UP, max_depth=1) == []
    assert _routes(search(LUNG_CANCER, OUTER_ROOT, edge_filter=UP, max_depth=2)) == [
        (LUNG_CANCER, CANCER, OUTDATED, OUTER_ROOT)
    ]


def test_max_paths_limits_the_result(search) -> None:
    assert len(search(LUNG_CANCER, DISEASE, edge_filter=UP, max_paths=1)) == 1


def test_unconnected_concepts_have_no_path(search) -> None:
    assert search(ORPHAN, DISEASE) == []


def test_a_traced_search_records_how_it_ended(kg) -> None:
    paths, trace = find_shortest_paths(kg, LUNG_CANCER, DISEASE, edge_filter=UP, traced=True)

    assert paths
    assert trace is not None
    assert trace.seeds
    assert trace.steps


def test_a_traced_search_of_a_concept_to_itself_says_so(kg) -> None:
    _, trace = find_shortest_paths(kg, DISEASE, DISEASE, traced=True)

    assert trace is not None
    assert trace.terminated_reason == "source_equals_target"


DIAMONDS = ((1, 2), (1, 3), (2, 4), (3, 4), (4, 5), (4, 6), (5, 7), (6, 7))


REPEATED = ((1, 4), (1, 5), (2, 5), (4, 6), (4, 7), (5, 3), (6, 1), (6, 2), (6, 4), (7, 1))
"""A graph where paths from 4 to 3 meet at more than one node: 4-6-1-5-3, 4-7-1-5-3 and 4-6-2-5-3."""


def test_a_path_through_several_meeting_nodes_is_found_once(search_in) -> None:
    edges = EdgeList(REPEATED)

    assert _routes(search_in(edges, 4, 3, max_depth=3)) == [
        (4, 6, 1, 5, 3),
        (4, 6, 2, 5, 3),
        (4, 7, 1, 5, 3),
    ]


def test_a_search_cut_by_max_depth_keeps_every_shortest_path(search_in) -> None:
    edges = EdgeList(REPEATED)

    assert _routes(search_in(edges, 4, 3, max_depth=2)) == edges.shortest_routes(4, 3)


@pytest.mark.parametrize(("steps", "max_depth"), [(2, 1), (3, 2), (4, 2), (5, 3), (6, 3)])
def test_max_depth_from_each_end_reaches_a_path_of_twice_the_steps(
    search_in, steps, max_depth
) -> None:
    chain = EdgeList((n, n + 1) for n in range(1, steps + 1))

    assert _routes(search_in(chain, 1, steps + 1, max_depth=max_depth)) == [
        tuple(range(1, steps + 2))
    ]
    assert search_in(chain, 1, steps + 1, max_depth=max_depth - 1) == []


def test_max_paths_cuts_a_meeting_node_that_has_more_paths(search_in) -> None:
    edges = EdgeList(DIAMONDS)

    assert len(edges.shortest_routes(1, 7)) == 4
    assert len(search_in(edges, 1, 7, max_paths=3)) == 3
    assert len(search_in(edges, 1, 7, max_paths=2)) == 2
    assert len(search_in(edges, 1, 7, max_paths=9)) == 4


def test_a_search_stops_reading_once_its_paths_are_found(search_in) -> None:
    edges = EdgeList([(1, 2), (2, 3), (1, 10), (10, 11), (11, 12), (12, 13), (3, 20), (20, 21)])

    assert _routes(search_in(edges, 1, 3)) == [(1, 2, 3)]
    assert {12, 13, 20, 21}.isdisjoint(edges.read_from)


def test_both_searches_find_every_shortest_path_of_random_graphs(search_in) -> None:
    generator = random.Random(7)
    for _ in range(200):
        size = generator.randint(4, 9)
        edges = EdgeList(
            sorted(
                {
                    (subject, object_)
                    for subject, object_ in (
                        (generator.randint(1, size), generator.randint(1, size))
                        for _ in range(generator.randint(size, 3 * size))
                    )
                    if subject != object_
                }
            )
        )
        source, target = generator.sample(edges.concepts, 2)
        max_depth = generator.choice([1, 2, 3, 6])
        expected = edges.shortest_routes(source, target)

        found = _routes(search_in(edges, source, target, max_depth=max_depth, max_paths=1000))

        assert len(found) == len(set(found))
        if expected and len(expected[0]) - 1 <= 2 * max_depth:
            assert found == expected, (edges.edge_list, source, target, max_depth)


def test_the_nodes_of_a_path_say_whether_they_are_standard(search_in) -> None:
    [path] = search_in(EdgeList([(1, 2), (2, 3)]), 1, 3)

    assert [(step.subject.is_standard, step.object.is_standard) for step in path.steps] == [
        (True, False),
        (False, True),
    ]


def test_the_paths_of_the_lowest_meeting_node_come_first(search_in) -> None:
    edges = EdgeList([(100, 1), (100, 8), (1, 200), (8, 200)])

    assert _routes(search_in(edges, 100, 200, max_paths=1)) == [(100, 1, 200)]


def test_paths_are_not_built_once_enough_are_found(search_in, monkeypatch) -> None:
    built = []
    reconstruct = paths_module.reconstruct_paths
    monkeypatch.setattr(
        paths_module,
        "reconstruct_paths",
        lambda *args: built.append(args[2]) or reconstruct(*args),
    )

    search_in(EdgeList(REPEATED), 4, 3, max_paths=1)

    assert len(built) == 1


def test_a_trace_lists_the_expansions_in_order() -> None:
    edges = EdgeList([(1, 2), (2, 3), (2, 9)])

    _, trace = find_shortest_paths(edges.graph, 1, 3, traced=True)

    assert trace is not None
    assert trace.terminated_reason == "shortest_paths_found"
    assert trace.seeds == (1,)
    assert [
        (step.depth, step.node, [(edge.subject_id, edge.object_id) for edge in step.expanded_edges])
        for step in trace.steps
    ] == [(0, 1, [(1, 2)]), (1, 2, [(2, 3), (2, 9)])]


def test_a_trace_of_a_search_without_a_path_says_so() -> None:
    paths, trace = find_shortest_paths(EdgeList([(1, 2), (2, 3)]).graph, 3, 1, traced=True)

    assert paths == []
    assert trace is not None
    assert trace.terminated_reason == "no_path"
    assert trace.seeds == (3,)
    assert [(step.depth, step.node) for step in trace.steps] == [(0, 3), (0, 1)]
