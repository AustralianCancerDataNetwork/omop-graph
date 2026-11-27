from __future__ import annotations

from collections.abc import Iterator

import pytest
from fixtures.hierarchy_cdm import (
    CANCER,
    DISEASE,
    LUNG,
    LUNG_CANCER,
    SOURCE,
    hierarchy_cdm,
)

from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.graph.model.edges import IS_A_FILTER
from omop_graph.graph.model.matches import LabelMatchKind
from omop_graph.graph.paths import GraphPath, Node, PathStep
from omop_graph.graph.traverse import (
    GraphTrace,
    Subgraph,
    TerminationReason,
    TraceStep,
    traverse,
)
from omop_graph.reasoning.explanation import PathExplanation
from omop_graph.render import (
    bind_default_renderers,
    render_explained_path,
    render_path,
    render_subgraph,
    render_trace,
)


def _path(*steps: tuple[int, bool, str, int, bool]) -> GraphPath:
    """A path of steps given as the subject, its standardness, the predicate and the object."""
    return GraphPath(
        tuple(
            PathStep(Node(subject, subject_standard), predicate, Node(object_, object_standard))
            for subject, subject_standard, predicate, object_, object_standard in steps
        )
    )


def _lines(*lines: str) -> str:
    return "\n".join(lines)


def _trace_from(kg: KnowledgeGraph, seed: int) -> GraphTrace:
    """The trace of a walk of one step from a concept, along any relationship."""
    trace = traverse(kg, [seed], edge_filter=None, max_depth=1, max_nodes=None, trace=True)[1]
    assert trace is not None
    return trace


@pytest.fixture(scope="module")
def kg() -> Iterator[KnowledgeGraph]:
    with hierarchy_cdm() as sessions:
        yield KnowledgeGraph(sessions)


@pytest.fixture(scope="module")
def path() -> GraphPath:
    return _path((LUNG_CANCER, True, "Is a", CANCER, True), (CANCER, True, "Is a", DISEASE, True))


@pytest.fixture(scope="module")
def subgraph_and_trace(kg) -> tuple[Subgraph, GraphTrace]:
    subgraph, trace = traverse(
        kg, [LUNG_CANCER], edge_filter=IS_A_FILTER, max_depth=2, max_nodes=None, trace=True
    )
    assert trace is not None
    return subgraph, trace


def _crowded_trace(kg, edge_count: int) -> GraphTrace:
    """A trace in which one expansion has that many edges of the same relationship."""
    edge = min(kg.edges(LUNG_CANCER, "out", IS_A_FILTER), key=lambda edge: edge.object_id)
    return GraphTrace(
        seeds=(LUNG_CANCER,),
        steps=[TraceStep(depth=0, node=LUNG_CANCER, expanded_edges=(edge,) * edge_count)],
        terminated_reason=TerminationReason.MAX_NODES,
    )


@pytest.fixture(scope="module")
def small_subgraph(kg) -> Subgraph:
    """The concept and the two it is directly a kind of, so its nodes and edges differ in number."""
    return traverse(
        kg, [LUNG_CANCER], edge_filter=IS_A_FILTER, max_depth=1, max_nodes=None, trace=False
    )[0]


def test_a_path_in_text_has_a_line_for_each_step(kg, path) -> None:
    assert render_path(kg, path, format="text") == (
        "Concept 203 --[Is a]--> Concept 201\nConcept 201 --[Is a]--> Concept 200"
    )


def test_a_path_in_mermaid_is_a_left_to_right_graph_of_its_steps(kg, path) -> None:
    assert render_path(kg, path, format="mmd") == (
        "graph LR\n"
        '203["Concept 203"] -->|Is a| 201["Concept 201"]\n'
        '201["Concept 201"] -->|Is a| 200["Concept 200"]'
    )


def test_a_path_in_html_has_a_line_for_each_step(kg, path) -> None:
    assert render_path(kg, path, format="html") == (
        "Concept 203 <b>--[Is a]--></b> Concept 201<br/>Concept 201 <b>--[Is a]--></b> Concept 200"
    )


def test_the_format_follows_the_environment_when_not_given(kg, path) -> None:
    assert render_path(kg, path) == render_path(kg, path, format="text")


def test_a_subgraph_in_text_counts_and_lists_its_concepts(kg, subgraph_and_trace) -> None:
    subgraph, _ = subgraph_and_trace

    assert render_subgraph(kg, subgraph, format="text") == _lines(
        "Subgraph:",
        "  Nodes: 5",
        "  Edges: 5",
        "",
        "- Concept 200 (SNOMED:C200)",
        "- Concept 201 (SNOMED:C201)",
        "- Concept 202 (SNOMED:C202)",
        "- Concept 203 (SNOMED:C203)",
        "- Concept 210 (SNOMED:C210)",
    )


def test_a_subgraph_in_text_counts_its_nodes_and_its_edges_apart(kg, small_subgraph) -> None:
    assert render_subgraph(kg, small_subgraph, format="text").startswith(
        "Subgraph:\n  Nodes: 3\n  Edges: 2\n"
    )


def test_a_subgraph_in_html_counts_its_nodes_and_its_edges_apart(kg, small_subgraph) -> None:
    assert "Nodes: 3 · Edges: 2" in render_subgraph(kg, small_subgraph, format="html")


def test_a_subgraph_in_mermaid_has_a_node_and_an_edge_line_each(kg, subgraph_and_trace) -> None:
    subgraph, _ = subgraph_and_trace
    header, *lines = render_subgraph(kg, subgraph, format="mmd").split("\n")

    assert header == "graph TD"
    assert sorted(lines) == sorted(
        [
            *(f'  {n}["Concept {n}\\nSNOMED:C{n}"]' for n in (200, 201, 202, 203, 210)),
            "  203 -->|Is a| 201",
            "  203 -->|Is a| 202",
            "  201 -->|Is a| 200",
            "  201 -->|Is a| 210",
            "  202 -->|Is a| 200",
        ]
    )


def test_a_subgraph_in_html_shows_a_card_for_each_concept(kg, subgraph_and_trace) -> None:
    subgraph, _ = subgraph_and_trace
    rendered = render_subgraph(kg, subgraph, format="html")

    assert "Nodes: 5 · Edges: 5" in rendered
    assert rendered.count("<b>Concept ") == 5
    assert rendered.count("✅") == 4
    assert rendered.count("❌") == 1
    assert "<b>Concept 210</b> ❌" in rendered
    assert "<code>SNOMED:C203</code>" in rendered


def test_a_trace_in_text_lists_the_expansions_by_relationship(kg, subgraph_and_trace) -> None:
    _, trace = subgraph_and_trace

    assert render_trace(kg, trace, format="text") == _lines(
        "[depth 0] Concept 203",
        "    └─ Is a",
        "        → Concept 201",
        "        → Concept 202",
        "[depth 1] Concept 201",
        "    └─ Is a",
        "        → Concept 200",
        "        → Concept 210",
        "[depth 1] Concept 202",
        "    └─ Is a",
        "        → Concept 200",
    )


def test_a_trace_in_text_says_why_it_ended(kg) -> None:
    assert render_trace(kg, _crowded_trace(kg, 1), format="text").endswith(
        "\n[terminated: max_nodes]"
    )


@pytest.mark.parametrize(("edges", "shown", "more"), [(6, 6, None), (7, 6, 1), (9, 6, 3)])
def test_a_trace_in_text_shows_six_edges_of_a_relationship_and_counts_the_rest(
    kg, edges, shown, more
) -> None:
    lines = render_trace(kg, _crowded_trace(kg, edges), format="text").split("\n")

    assert lines.count("        → Concept 201") == shown
    assert [line for line in lines if "more" in line] == (
        [] if more is None else [f"        … {more} more"]
    )


def test_a_trace_in_html_lists_the_expansions_in_order(kg, subgraph_and_trace) -> None:
    _, trace = subgraph_and_trace
    rendered = render_trace(kg, trace, format="html")

    ordered = [
        "[depth 0]",
        "Concept 203 ✅",
        "└─ Is a",
        "→ Concept 201",
        "→ Concept 202",
        "[depth 1]",
    ]
    positions = [rendered.index(text) for text in ordered]
    assert positions == sorted(positions)
    assert rendered.count("[depth 1]") == 2
    assert "<code>SNOMED:C203</code>" in rendered
    assert "terminated" not in rendered


def test_a_trace_in_html_says_why_it_ended(kg) -> None:
    rendered = render_trace(kg, _crowded_trace(kg, 1), format="html")

    assert "<div style='color:#a00;'>[terminated: max_nodes]</div>" in rendered


@pytest.mark.parametrize(("edges", "shown", "more"), [(5, 5, None), (6, 5, 1), (9, 5, 4)])
def test_a_trace_in_html_shows_five_edges_of_a_relationship_and_counts_the_rest(
    kg, edges, shown, more
) -> None:
    rendered = render_trace(kg, _crowded_trace(kg, edges), format="html")

    assert rendered.count("→ Concept 201") == shown
    assert rendered.count("more") == (0 if more is None else 1)
    assert more is None or f"… {more} more" in rendered


def test_an_explained_path_gives_the_kind_depth_and_standardness(kg) -> None:
    mapped = _path((SOURCE, False, "Maps to", LUNG_CANCER, True))
    trace = _trace_from(kg, SOURCE)
    explanation = PathExplanation.from_path(kg, mapped, trace, LabelMatchKind.EXACT)

    assert render_explained_path(kg, explanation, format="text") == _lines(
        "Original: Concept 212",
        "Resolved: Concept 203 (ID: 203, standard)",
        "Steps:",
        "  Concept 212 --[Maps to]--> Concept 203 (IDENTITY, depth=0) Identity",
    )


def test_an_explained_path_names_the_kind_of_the_relationship(kg) -> None:
    attribute = _path((LUNG_CANCER, True, "Has finding site", LUNG, True))
    trace = GraphTrace(seeds=(LUNG_CANCER,), steps=[])
    explanation = PathExplanation.from_path(kg, attribute, trace, LabelMatchKind.EXACT)

    assert render_explained_path(kg, explanation, format="text").endswith(
        "  Concept 203 --[Has finding site]--> Concept 206 (ATTRIBUTE, depth=?) Attribute"
    )


def test_an_explained_path_of_a_concept_to_itself_says_if_it_is_not_standard(kg) -> None:
    trace = GraphTrace(seeds=(SOURCE,), steps=[])
    explanation = PathExplanation.from_path(kg, GraphPath(()), trace, LabelMatchKind.EXACT)

    assert render_explained_path(kg, explanation, format="text") == _lines(
        "Original: Concept 212", "Resolved: Concept 212 (ID: 212, non-standard)", "Steps:"
    )


def test_an_explained_path_in_html_has_a_row_for_each_step(kg) -> None:
    mapped = _path((SOURCE, False, "Maps to", LUNG_CANCER, True))
    trace = _trace_from(kg, SOURCE)
    explanation = PathExplanation.from_path(kg, mapped, trace, LabelMatchKind.EXACT)
    rendered = " ".join(render_explained_path(kg, explanation, format="html").split())

    assert (
        "<b>Original:</b> Concept 212 &rarr; <b>Resolved:</b> Concept 203 &#10003; (ID: 203)"
        in rendered
    )
    assert (
        "<tr> <td>Concept 212</td> <td>Maps to</td> <td>Concept 203</td> <td>IDENTITY</td> "
        "<td>0</td> <td>Identity</td> </tr>"
    ) in rendered


def test_an_explained_path_in_html_marks_a_concept_that_is_not_standard(kg) -> None:
    trace = GraphTrace(seeds=(SOURCE,), steps=[])
    explanation = PathExplanation.from_path(kg, GraphPath(()), trace, LabelMatchKind.EXACT)

    rendered = " ".join(render_explained_path(kg, explanation, format="html").split())

    assert "<b>Resolved:</b> Concept 212 &#10007; (ID: 212)" in rendered


def test_binding_the_renderers_gives_notebook_displays(
    kg, path, subgraph_and_trace, monkeypatch
) -> None:
    subgraph, trace = subgraph_and_trace
    for displayed in (GraphPath, Subgraph, GraphTrace):
        monkeypatch.setattr(displayed, "_repr_html_", None, raising=False)
    bind_default_renderers(kg)

    assert path._repr_html_() == render_path(kg, path, format="html")
    assert subgraph._repr_html_() == render_subgraph(kg, subgraph, format="html")
    assert trace._repr_html_() == render_trace(kg, trace, format="html")
