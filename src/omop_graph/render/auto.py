from __future__ import annotations

from typing import Literal

from omop_graph.graph.paths import GraphPath, PathExplanation
from omop_graph.graph.traverse import GraphTrace, Subgraph

from .html import (
    explained_path_html,
    path_html,
    subgraph_html,
    trace_html_with_cards,
)
from .mmd import (
    path_mermaid,
    subgraph_mermaid,
)
from .text import (
    explained_path_text,
    path_text,
    subgraph_text,
    trace_text,
)

Format = Literal["auto", "html", "text", "mmd"]


def _in_notebook() -> bool:
    try:
        from IPython.core.getipython import get_ipython  # type: ignore[import-unresolved]

        ip = get_ipython()
        return ip is not None and "IPKernelApp" in ip.config
    except Exception:
        return False


def _resolve_format(fmt: Format) -> str:
    if fmt != "auto":
        return fmt
    return "html" if _in_notebook() else "text"


def render_subgraph(
    kg,
    sg: Subgraph,
    *,
    format: Format = "auto",
) -> str:
    fmt = _resolve_format(format)
    if fmt == "html":
        return subgraph_html(kg, sg)
    if fmt == "mmd":
        return subgraph_mermaid(kg, sg)
    return subgraph_text(kg, sg)


def render_trace(
    kg,
    trace: GraphTrace,
    *,
    format: Format = "auto",
) -> str:
    fmt = _resolve_format(format)
    if fmt == "html":
        return trace_html_with_cards(kg, trace)
    return trace_text(kg, trace)


def render_path(
    kg,
    path: GraphPath,
    *,
    format: Format = "auto",
) -> str:
    fmt = _resolve_format(format)
    if fmt == "html":
        return path_html(kg, path)
    if fmt == "mmd":
        return path_mermaid(kg, path)
    return path_text(kg, path)


def render_explained_path(
    kg,
    explanation: PathExplanation,
    *,
    format: Format = "auto",
) -> str:
    fmt = _resolve_format(format)
    if fmt == "html":
        return explained_path_html(kg, explanation)
    return explained_path_text(kg, explanation)


def bind_default_renderers(kg):
    from omop_graph.render import render_path, render_subgraph, render_trace

    Subgraph._repr_html_ = lambda self: render_subgraph(kg, self, format="html")  # type: ignore
    GraphTrace._repr_html_ = lambda self: render_trace(kg, self, format="html")  # type: ignore
    GraphPath._repr_html_ = lambda self: render_path(kg, self, format="html")  # type: ignore
