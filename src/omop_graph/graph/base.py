from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import date
from typing import Iterable, Optional

from .edges import EdgeView, PredicateKind, Predicate
from .nodes import ConceptView


class GraphBackend(ABC):
    """
    Abstract graph interface.

    Algorithms (paths, traversal, scoring) depend ONLY on this interface.
    """

    @abstractmethod
    def concept_view(self, concept_id: int) -> ConceptView:
        ...

    @abstractmethod
    def predicate_kind(self, relationship_id: str) -> PredicateKind:
        ...

    @abstractmethod
    def predicate_name(self, relationship_id: str) -> str:
        ...

    @abstractmethod
    def reverse_predicate_id(self, relationship_id: str) -> Optional[str]:
        ...

    @abstractmethod
    def iter_edges(
        self, 
        concept_id: int, 
        *, 
        direction: str, 
        predicate: Predicate | str | None,
        predicate_kinds: set[PredicateKind] | None = None, 
        active_only: bool = True) -> Iterable[EdgeView]:
        ...

    def clear_caches(self) -> None:
        """Optional hook for cache invalidation."""
        return None
