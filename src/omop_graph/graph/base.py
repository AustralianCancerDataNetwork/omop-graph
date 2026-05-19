from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import date
from functools import lru_cache
from typing import Iterable, Optional, Literal
from sqlalchemy.orm import Session

from ..extensions.omop_alchemy import ClassIDEnum
from .edges import EdgeView
from .nodes import ConceptView


class GraphBackend(ABC):
    """
    Abstract graph interface.

    Algorithms (paths, traversal, scoring) depend ONLY on this interface.
    """

    @abstractmethod
    @lru_cache(maxsize=200_000)
    def concept_view(self, concept_id: int) -> ConceptView:
        ...

    @abstractmethod
    def predicate_kind(self, relationship_id: str) -> ClassIDEnum:
        ...

    @abstractmethod
    @lru_cache(maxsize=10_000)
    def predicate_name(self, relationship_id: str) -> str:
        ...

    @abstractmethod
    def reverse_predicate_id(self, relationship_id: str) -> Optional[str]:
        ...

    @abstractmethod
    def iter_edges(
        self, 
        session: Session,
        concept_ids: int | tuple[int, ...], 
        direction: Literal["in", "out"], 
        predicate_ids: frozenset[str] | None,
        predicate_kinds: Optional[frozenset[ClassIDEnum]] = None,
        active_only: bool = True,
        on: Optional[date] = None,
        within_domain: bool = True,
    ) -> Iterable[EdgeView]:
        """
        Iterate over edges for a concept with filtering.

        Parameters
        ----------
        session : Session
            Active session. This prevents the session from staying open if the generator
            is only partially consumed and the calling function determines how long the
            sessions stays active.
        concept_ids : int, tuple[int, ...]
            The source/target concept ID(s).
        direction : str
            'out' for outgoing, 'in' for incoming.
        predicate_ids : frozenset[str], optional
            Filter by specific relationship IDs.
        predicate_kinds : Set[ClassIDEnum], optional
            Filter by semantic kind of relationship.
        active_only : bool
            If True, return only valid/active edges.
        on : date, optional
            Check validity on a specific date.
        within_domain : bool
            If True, only return edges where source/target domains match.

        Yields
        -------
        EdgeView
            Edges matching criteria.
        """
        ...

    def clear_caches(self) -> None:
        """Optional hook for cache invalidation."""
        return None
