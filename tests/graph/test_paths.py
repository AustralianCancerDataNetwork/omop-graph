from __future__ import annotations

import random
from collections.abc import Iterator

import pytest
from fixtures.edge_list import EdgeList
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
from omop_graph.graph.model.edges import IS_A_FILTER, EdgeFilter
from omop_graph.graph.paths import find_shortest_paths
from omop_graph.graph.traverse import TerminationReason


@pytest.fixture(scope="module")
def kg() -> Iterator[KnowledgeGraph]:
    with hierarchy_cdm() as sessions:
        yield KnowledgeGraph(sessions)


@pytest.fixture
def search(kg):
    """The search on the hierarchy CDM, returning the paths."""
    return lambda *args, **kwargs: find_shortest_paths(kg, *args, **kwargs)[0]


def _routes(paths) -> list[tuple[int, ...]]:
    return sorted(path.nodes() for path in paths)


def test_a_concept_is_its_own_empty_path(search) -> None:
    paths = search(LUNG_CANCER, LUNG_CANCER)

    assert len(paths) == 1
    assert len(paths[0]) == 0


def test_every_shortest_path_is_found(search) -> None:
    paths = search(LUNG_CANCER, DISEASE, edge_filter=IS_A_FILTER)

    assert _routes(paths) == [
        (LUNG_CANCER, CANCER, DISEASE),
        (LUNG_CANCER, SMOKING_RELATED, DISEASE),
    ]
    assert {step.predicate for path in paths for step in path.steps} == {"Is a"}


def test_the_edge_filter_decides_which_edges_are_followed(search) -> None:
    only_sites = EdgeFilter(predicate_ids=frozenset({"Has finding site"}))

    assert _routes(search(LUNG_CANCER, LUNG, edge_filter=only_sites)) == [(LUNG_CANCER, LUNG)]
    assert search(LUNG_CANCER, DISEASE, edge_filter=only_sites) == []


def test_max_depth_bounds_the_length_of_a_path(search) -> None:
    assert search(LUNG_CANCER, DISEASE, edge_filter=IS_A_FILTER, max_depth=0) == []
    assert search(LUNG_CANCER, OUTER_ROOT, edge_filter=IS_A_FILTER, max_depth=2) == []
    assert _routes(search(LUNG_CANCER, OUTER_ROOT, edge_filter=IS_A_FILTER, max_depth=3)) == [
        (LUNG_CANCER, CANCER, OUTDATED, OUTER_ROOT)
    ]


def test_max_paths_limits_the_result(search) -> None:
    assert len(search(LUNG_CANCER, DISEASE, edge_filter=IS_A_FILTER, max_paths=1)) == 1


def test_unconnected_concepts_have_no_path(kg) -> None:
    paths, trace = find_shortest_paths(kg, ORPHAN, DISEASE, traced=True)

    assert paths == []
    assert trace is not None
    assert trace.terminated_reason is TerminationReason.NO_PATH


def test_a_traced_search_records_how_it_ended(kg) -> None:
    paths, trace = find_shortest_paths(
        kg, LUNG_CANCER, DISEASE, edge_filter=IS_A_FILTER, traced=True
    )

    assert paths
    assert trace is not None
    assert trace.terminated_reason is TerminationReason.SHORTEST_PATHS_FOUND
    assert trace.seeds == (LUNG_CANCER,)
    assert trace.steps


def test_a_traced_search_of_a_concept_to_itself_says_so(kg) -> None:
    _, trace = find_shortest_paths(kg, DISEASE, DISEASE, traced=True)

    assert trace is not None
    assert trace.terminated_reason is TerminationReason.SOURCE_EQUALS_TARGET


def test_an_untraced_search_returns_no_trace(kg) -> None:
    assert find_shortest_paths(kg, LUNG_CANCER, DISEASE)[1] is None
    assert find_shortest_paths(kg, ORPHAN, DISEASE)[1] is None
    assert find_shortest_paths(kg, DISEASE, DISEASE)[1] is None


def test_the_options_of_a_search_are_keywords(kg) -> None:
    with pytest.raises(TypeError):
        find_shortest_paths(kg, LUNG_CANCER, DISEASE, IS_A_FILTER)  # ty: ignore[too-many-positional-arguments]


DIAMONDS = ((1, 2), (1, 3), (2, 4), (3, 4), (4, 5), (4, 6), (5, 7), (6, 7))


REPEATED = ((1, 4), (1, 5), (2, 5), (4, 6), (4, 7), (5, 3), (6, 1), (6, 2), (6, 4), (7, 1))
"""A graph where paths from 4 to 3 meet at more than one node: 4-6-1-5-3, 4-7-1-5-3 and 4-6-2-5-3."""


def _find(edges: EdgeList, source: int, target: int, **kwargs):
    return find_shortest_paths(edges.graph, source, target, **kwargs)[0]


def test_a_path_through_several_meeting_nodes_is_found_once() -> None:
    edges = EdgeList(REPEATED)

    assert _routes(_find(edges, 4, 3, max_depth=4)) == [
        (4, 6, 1, 5, 3),
        (4, 6, 2, 5, 3),
        (4, 7, 1, 5, 3),
    ]


@pytest.mark.parametrize("steps", [1, 2, 3, 5, 6])
def test_max_depth_is_the_most_steps_a_path_may_have(steps) -> None:
    chain = EdgeList((n, n + 1) for n in range(1, steps + 1))

    assert _routes(_find(chain, 1, steps + 1, max_depth=steps)) == [tuple(range(1, steps + 2))]
    assert _find(chain, 1, steps + 1, max_depth=steps - 1) == []


def test_max_paths_cuts_a_meeting_node_that_has_more_paths() -> None:
    edges = EdgeList(DIAMONDS)

    assert len(edges.shortest_routes(1, 7)) == 4
    assert len(_find(edges, 1, 7, max_paths=3)) == 3
    assert len(_find(edges, 1, 7, max_paths=2)) == 2
    assert len(_find(edges, 1, 7, max_paths=9)) == 4


def test_a_search_reads_each_frontier_once_and_stops_when_its_paths_are_found() -> None:
    edges = EdgeList([(1, 2), (2, 3), (1, 10), (10, 11), (11, 12), (12, 13), (3, 20), (20, 21)])

    assert _routes(_find(edges, 1, 3)) == [(1, 2, 3)]
    assert edges.read_from == [1, 3]


def test_the_end_with_fewer_concepts_is_read_next() -> None:
    edges = EdgeList(
        [(1, 2), (1, 3), (2, 10), (3, 11), (6, 9), (7, 9), (8, 9), (5, 6), (5, 7), (4, 5)]
    )

    _find(edges, 1, 9)

    assert edges.read_from[:4] == [1, 9, 2, 3]


def test_a_search_stops_when_an_end_has_nothing_left_to_expand() -> None:
    edges = EdgeList([(1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7)])

    paths, trace = find_shortest_paths(edges.graph, 1, 99, traced=True)

    assert paths == []
    assert edges.read_from == [1, 99]
    assert trace is not None
    assert trace.terminated_reason is TerminationReason.NO_PATH
    assert [step.node for step in trace.steps] == [1, 99]


def test_a_search_cut_by_max_depth_reads_no_deeper_than_the_budget() -> None:
    edges = EdgeList((n, n + 1) for n in range(1, 10))

    assert _find(edges, 1, 10, max_depth=3) == []
    assert edges.read_from == [1, 2, 3]


def test_every_shortest_path_of_random_graphs_is_found_and_none_longer_than_max_depth() -> None:
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

        found = _routes(_find(edges, source, target, max_depth=max_depth, max_paths=1000))

        assert len(found) == len(set(found))
        if expected and len(expected[0]) - 1 <= max_depth:
            assert found == expected, (edges.edge_list, source, target, max_depth)
        else:
            assert found == [], (edges.edge_list, source, target, max_depth)


def test_two_relationships_between_a_pair_make_two_paths_on_either_side_of_the_meeting() -> None:
    edges = EdgeList(
        [(1, 2, "A"), (1, 2, "B"), (2, 3), (3, 4, "X"), (3, 4, "Y"), (1, 5), (1, 6), (1, 7)]
    )

    paths = _find(edges, 1, 4)

    assert sorted(tuple(step.predicate for step in path.steps) for path in paths) == [
        ("A", "Is a", "X"),
        ("A", "Is a", "Y"),
        ("B", "Is a", "X"),
        ("B", "Is a", "Y"),
    ]


def test_the_nodes_of_a_path_say_whether_they_are_standard() -> None:
    [path] = _find(EdgeList([(1, 2), (2, 3)]), 1, 3)

    assert [(step.subject.is_standard, step.object.is_standard) for step in path.steps] == [
        (True, False),
        (False, True),
    ]


def test_the_paths_of_the_lowest_meeting_node_come_first() -> None:
    edges = EdgeList([(100, 1), (100, 8), (1, 200), (8, 200)])

    assert _routes(_find(edges, 100, 200, max_paths=1)) == [(100, 1, 200)]


def test_paths_are_not_built_once_enough_are_found(monkeypatch) -> None:
    built = []
    reconstruct = paths_module.reconstruct_paths
    monkeypatch.setattr(
        paths_module,
        "reconstruct_paths",
        lambda *args: built.append(args[2]) or reconstruct(*args),
    )

    _find(EdgeList(REPEATED), 4, 3, max_depth=4, max_paths=1)

    assert len(built) == 1


@pytest.mark.parametrize("max_paths", [1, 2, 3, 20])
@pytest.mark.parametrize("graph", [DIAMONDS, REPEATED])
def test_the_result_does_not_depend_on_the_order_the_edges_are_read_in(graph, max_paths) -> None:
    source, target = (1, 7) if graph is DIAMONDS else (4, 3)
    shuffled = random.Random(3)
    results = []
    for _ in range(12):
        edge_list = list(graph)
        shuffled.shuffle(edge_list)
        paths, trace = find_shortest_paths(
            EdgeList(edge_list).graph, source, target, max_paths=max_paths, traced=True
        )
        assert trace is not None
        results.append(
            (
                paths,
                [
                    (
                        step.depth,
                        step.node,
                        [(e.subject_id, e.object_id) for e in step.expanded_edges],
                    )
                    for step in trace.steps
                ],
            )
        )

    assert all(result == results[0] for result in results)


def test_a_trace_lists_the_expansions_in_order() -> None:
    edges = EdgeList([(1, 2), (2, 3), (2, 9)])

    _, trace = find_shortest_paths(edges.graph, 1, 3, traced=True)

    assert trace is not None
    assert trace.terminated_reason is TerminationReason.SHORTEST_PATHS_FOUND
    assert trace.seeds == (1,)
    assert [
        (step.depth, step.node, [(edge.subject_id, edge.object_id) for edge in step.expanded_edges])
        for step in trace.steps
    ] == [(0, 1, [(1, 2)]), (1, 2, [(2, 3), (2, 9)])]


def test_a_trace_records_the_edges_read_towards_the_target_by_the_concept_they_end_at() -> None:
    edges = EdgeList([(1, 2), (1, 3), (2, 4), (3, 4), (4, 5)])

    _, trace = find_shortest_paths(edges.graph, 1, 5, traced=True)

    assert trace is not None
    assert [
        (step.depth, step.node, [(edge.subject_id, edge.object_id) for edge in step.expanded_edges])
        for step in trace.steps
    ] == [(0, 1, [(1, 2), (1, 3)]), (0, 5, [(4, 5)]), (1, 4, [(2, 4), (3, 4)])]


def test_a_trace_of_a_search_without_a_path_says_so() -> None:
    paths, trace = find_shortest_paths(EdgeList([(1, 2), (2, 3)]).graph, 3, 1, traced=True)

    assert paths == []
    assert trace is not None
    assert trace.terminated_reason is TerminationReason.NO_PATH
    assert trace.seeds == (3,)
    assert [(step.depth, step.node) for step in trace.steps] == [(0, 3)]
