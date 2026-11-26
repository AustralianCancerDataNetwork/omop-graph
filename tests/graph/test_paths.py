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
    hierarchy_cdm,
)

from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.graph.model.edges import EdgeFilter
from omop_graph.graph.paths import find_shortest_paths, find_shortest_paths_batch

UP = EdgeFilter(predicate_ids=frozenset({"Is a"}))


@pytest.fixture(scope="module")
def kg() -> Iterator[KnowledgeGraph]:
    with hierarchy_cdm() as sessions:
        yield KnowledgeGraph(sessions)


@pytest.fixture(params=["bidirectional", "batched"])
def search(request, kg):
    """Both searches, called alike and returning the paths."""
    if request.param == "bidirectional":
        return lambda *args, **kwargs: find_shortest_paths(kg, *args, **kwargs)[0]
    return lambda *args, **kwargs: find_shortest_paths_batch(kg, *args, **kwargs)


def _routes(paths) -> set[tuple[int, ...]]:
    return {path.nodes() for path in paths}


def test_a_concept_is_its_own_empty_path(search) -> None:
    paths = search(LUNG_CANCER, LUNG_CANCER)

    assert len(paths) == 1
    assert len(paths[0]) == 0


def test_every_shortest_path_is_found(search) -> None:
    paths = search(LUNG_CANCER, DISEASE, edge_filter=UP)

    assert _routes(paths) == {
        (LUNG_CANCER, CANCER, DISEASE),
        (LUNG_CANCER, SMOKING_RELATED, DISEASE),
    }
    assert {step.predicate for path in paths for step in path.steps} == {"Is a"}


def test_the_edge_filter_decides_which_edges_are_followed(search) -> None:
    only_sites = EdgeFilter(predicate_ids=frozenset({"Has finding site"}))

    assert _routes(search(LUNG_CANCER, LUNG, edge_filter=only_sites)) == {(LUNG_CANCER, LUNG)}
    assert search(LUNG_CANCER, DISEASE, edge_filter=only_sites) == []


def test_max_depth_bounds_the_search_from_each_end(search) -> None:
    assert search(LUNG_CANCER, DISEASE, edge_filter=UP, max_depth=0) == []
    assert search(LUNG_CANCER, OUTER_ROOT, edge_filter=UP, max_depth=1) == []
    assert _routes(search(LUNG_CANCER, OUTER_ROOT, edge_filter=UP, max_depth=2)) == {
        (LUNG_CANCER, CANCER, OUTDATED, OUTER_ROOT)
    }


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
