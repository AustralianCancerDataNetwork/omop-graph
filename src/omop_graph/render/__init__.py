"""
Rendering utilities for KnowledgeGraph outputs.

Public API:
- render_subgraph(...)
- render_trace(...)
- render_path(...)
- render_explained_path(...)
- render_candidate_hits(...)
- render_grounding_review(...)

Renderers auto-select HTML / text / Mermaid depending on environment.
"""

from .auto import (
    render_candidate_hits,
    render_grounding_review,
    render_subgraph,
    render_trace,
    render_path,
    render_explained_path,
    bind_default_renderers
)

__all__ = [
    "render_candidate_hits",
    "render_grounding_review",
    "render_subgraph",
    "render_trace",
    "render_path",
    "render_explained_path",
    "bind_default_renderers",
]
