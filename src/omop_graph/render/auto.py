from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from omop_graph.graph.paths import GraphPath
from omop_graph.graph.traverse import GraphTrace, Subgraph
from omop_graph.reasoning.explanation import PathExplanation

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
    """Whether the code runs in a Jupyter notebook kernel."""
    try:
        from IPython.core.getipython import get_ipython

        ip = get_ipython()
        return ip is not None and "IPKernelApp" in ip.config
    except (ImportError, AttributeError):
        return False


def _resolve_format(fmt: Format) -> str:
    """The format to render in: ``html`` in a notebook and ``text`` elsewhere, when ``auto``."""
    if fmt != "auto":
        return fmt
    return "html" if _in_notebook() else "text"


def render_subgraph(
    kg,
    sg: Subgraph,
    *,
    format: Format = "auto",
) -> str:
    """
    Render a subgraph.

    Parameters
    ----------
    kg : KnowledgeGraph
        The graph to read the concept names from.
    sg : Subgraph
        The subgraph to render.
    format : {"auto", "html", "text", "mmd"}
        The output format. ``auto`` gives HTML in a notebook and text elsewhere. The HTML
        shows cards for the first 20 concepts.

    Returns
    -------
    str
        The node and edge counts and the concepts, or a Mermaid flowchart for ``mmd``.
    """
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
    """
    Render the trace of a traversal or a path search.

    Parameters
    ----------
    kg : KnowledgeGraph
        The graph to read the concept and relationship names from.
    trace : GraphTrace
        The trace to render.
    format : {"auto", "html", "text", "mmd"}
        The output format. ``auto`` gives HTML in a notebook and text elsewhere. ``mmd``
        is not supported and gives text.

    Returns
    -------
    str
        Each expanded concept with its depth, and the edges read from it grouped by
        relationship, followed by why the search ended. A relationship shows its first 6
        edges in text and its first 5 in HTML, and counts the rest.
    """
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
    """
    Render a path.

    Parameters
    ----------
    kg : KnowledgeGraph
        The graph to read the concept names from.
    path : GraphPath
        The path to render.
    format : {"auto", "html", "text", "mmd"}
        The output format. ``auto`` gives HTML in a notebook and text elsewhere.

    Returns
    -------
    str
        One line per step, or a left to right Mermaid flowchart for ``mmd``.
    """
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
    """
    Render the explanation of a path.

    Parameters
    ----------
    kg : KnowledgeGraph
        The graph to read the concept names from.
    explanation : PathExplanation
        The explanation to render.
    format : {"auto", "html", "text", "mmd"}
        The output format. ``auto`` gives HTML in a notebook and text elsewhere. ``mmd``
        is not supported and gives text.

    Returns
    -------
    str
        The concept the path starts at and the one it resolves to, then each step with
        the kind of its relationship and its depth in the search.
    """
    fmt = _resolve_format(format)
    if fmt == "html":
        return explained_path_html(kg, explanation)
    return explained_path_text(kg, explanation)


def _install_repr_html(cls: type[Any], render: Callable[[Any], str]) -> None:
    """Set render as the IPython rich-display hook ``_repr_html_`` of cls."""
    cls._repr_html_ = render


def bind_default_renderers(kg) -> None:
    """
    Make paths, subgraphs and traces display as HTML in a notebook.

    This sets ``_repr_html_`` on ``GraphPath``, ``Subgraph`` and ``GraphTrace``, so it applies
    to every instance, and a later call replaces the graph an earlier call bound.

    Parameters
    ----------
    kg : KnowledgeGraph
        The graph the renderers read concept names from.
    """
    from omop_graph.render import render_path, render_subgraph, render_trace

    _install_repr_html(Subgraph, lambda self: render_subgraph(kg, self, format="html"))
    _install_repr_html(GraphTrace, lambda self: render_trace(kg, self, format="html"))
    _install_repr_html(GraphPath, lambda self: render_path(kg, self, format="html"))
