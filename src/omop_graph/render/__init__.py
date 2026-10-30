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
    bind_default_renderers,
    render_explained_path,
    render_path,
    render_subgraph,
    render_trace,
)

__all__ = [
    "bind_default_renderers",
    "render_explained_path",
    "render_path",
    "render_subgraph",
    "render_trace",
]
