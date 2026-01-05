"""
Rendering utilities for KnowledgeGraph outputs.

Public API:
- render_subgraph(...)
- render_trace(...)
- render_path(...)
- render_explained_path(...)

Renderers auto-select HTML / text / Mermaid depending on environment.
"""

from .auto import (
    render_subgraph,
    render_trace,
    render_path,
    render_explained_path,
    bind_default_renderers
)

__all__ = [
    "render_subgraph",
    "render_trace",
    "render_path",
    "render_explained_path",
    "bind_default_renderers",
]
