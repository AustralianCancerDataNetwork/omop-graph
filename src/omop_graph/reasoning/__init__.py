from __future__ import annotations

from importlib import import_module

__all__ = [
    "GroundingBatchResult",
    "GroundingBatchRunner",
    "GroundingConstraints",
    "ground_term",
]


def __getattr__(name: str):
    if name in {"GroundingConstraints", "ground_term"}:
        module = import_module("omop_graph.reasoning.grounding")
        return getattr(module, name)
    if name in {"GroundingBatchResult", "GroundingBatchRunner"}:
        module = import_module("omop_graph.reasoning.grounding_batch")
        return getattr(module, name)
    raise AttributeError(f"module 'omop_graph.reasoning' has no attribute {name!r}")
