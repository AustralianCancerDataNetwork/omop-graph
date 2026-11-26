from __future__ import annotations

from collections.abc import Iterator

import pytest
from fixtures.hierarchy_cdm import CANCER, DISEASE, LUNG_CANCER, hierarchy_cdm

from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.graph.model.edges import EdgeFilter
from omop_graph.graph.model.matches import LabelMatchKind
from omop_graph.graph.paths import GraphPath, find_shortest_paths
from omop_graph.graph.traverse import GraphTrace, Subgraph, traverse
from omop_graph.reasoning.explanation import PathExplanation
from omop_graph.render import (
    bind_default_renderers,
    render_explained_path,
    render_path,
    render_subgraph,
    render_trace,
)

UP = EdgeFilter(predicate_ids=frozenset({"Is a"}))
NAMES = {
    LUNG_CANCER: f"Concept {LUNG_CANCER}",
    CANCER: f"Concept {CANCER}",
    DISEASE: f"Concept {DISEASE}",
}


@pytest.fixture(scope="module")
def kg() -> Iterator[KnowledgeGraph]:
    with hierarchy_cdm() as sessions:
        yield KnowledgeGraph(sessions)


@pytest.fixture(scope="module")
def path(kg) -> GraphPath:
    return find_shortest_paths(kg, LUNG_CANCER, CANCER, edge_filter=UP)[0][0]


@pytest.fixture(scope="module")
def subgraph_and_trace(kg) -> tuple[Subgraph, GraphTrace]:
    subgraph, trace = traverse(kg, [LUNG_CANCER], UP, max_depth=2, max_nodes=None, trace=True)
    assert trace is not None
    return subgraph, trace


@pytest.mark.parametrize("format", ["text", "html", "mmd"])
def test_a_path_names_both_ends_in_every_format(kg, path, format) -> None:
    rendered = render_path(kg, path, format=format)

    assert NAMES[LUNG_CANCER] in rendered
    assert NAMES[CANCER] in rendered
    assert "Is a" in rendered


def test_a_path_in_mermaid_is_a_left_to_right_graph(kg, path) -> None:
    assert render_path(kg, path, format="mmd").startswith("graph LR")


@pytest.mark.parametrize("format", ["text", "html"])
def test_a_subgraph_lists_its_concepts(kg, subgraph_and_trace, format) -> None:
    subgraph, _ = subgraph_and_trace
    rendered = render_subgraph(kg, subgraph, format=format)

    assert all(name in rendered for name in NAMES.values())


def test_a_subgraph_in_mermaid_has_a_node_per_concept(kg, subgraph_and_trace) -> None:
    subgraph, _ = subgraph_and_trace
    rendered = render_subgraph(kg, subgraph, format="mmd")

    assert rendered.startswith("graph TD")
    assert all(f"{concept_id}[" in rendered for concept_id in subgraph.nodes)


@pytest.mark.parametrize("format", ["text", "html"])
def test_a_trace_shows_each_expansion(kg, subgraph_and_trace, format) -> None:
    _, trace = subgraph_and_trace
    rendered = render_trace(kg, trace, format=format)

    assert NAMES[LUNG_CANCER] in rendered
    assert NAMES[CANCER] in rendered


@pytest.mark.parametrize("format", ["text", "html"])
def test_an_explained_path_gives_the_kind_of_each_step(
    kg, path, subgraph_and_trace, format
) -> None:
    _, trace = subgraph_and_trace
    explanation = PathExplanation.from_path(kg, path, trace, LabelMatchKind.EXACT)
    rendered = render_explained_path(kg, explanation, format=format)

    assert NAMES[LUNG_CANCER] in rendered
    assert "Hierarchy" in rendered


def test_the_format_follows_the_environment_when_not_given(kg, path) -> None:
    assert render_path(kg, path) == render_path(kg, path, format="text")


def test_binding_the_renderers_gives_notebook_displays(
    kg, path, subgraph_and_trace, monkeypatch
) -> None:
    subgraph, trace = subgraph_and_trace
    for displayed in (GraphPath, Subgraph, GraphTrace):
        monkeypatch.setattr(displayed, "_repr_html_", None, raising=False)
    bind_default_renderers(kg)

    assert NAMES[CANCER] in path._repr_html_()
    assert NAMES[LUNG_CANCER] in subgraph._repr_html_()
    assert NAMES[LUNG_CANCER] in trace._repr_html_()
