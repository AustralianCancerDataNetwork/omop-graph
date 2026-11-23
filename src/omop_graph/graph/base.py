from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Literal

from sqlalchemy.orm import Session

from ..extensions.omop_alchemy import PredicateKind
from .edges import EdgeFilter, EdgeView
from .nodes import ConceptView


class GraphBackend(ABC):
    """
    Abstract graph interface.

    Algorithms (paths, traversal, scoring) depend ONLY on this interface.
    """

    @abstractmethod
    def concept_view(self, concept_id: int) -> ConceptView: ...

    @abstractmethod
    def predicate_kind(self, relationship_id: str) -> PredicateKind: ...

    @abstractmethod
    def predicate_name(self, relationship_id: str) -> str: ...

    @abstractmethod
    def reverse_predicate_id(self, relationship_id: str) -> str | None: ...

    @abstractmethod
    def iter_edges(
        self,
        session: Session,
        concept_ids: int | tuple[int, ...],
        direction: Literal["in", "out"],
        edge_filter: EdgeFilter | None = None,
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
        edge_filter : EdgeFilter, optional
            Which edges to return. Defaults to ``EdgeFilter()``: active edges of any
            relationship, across domains.

        Yields
        -------
        EdgeView
            Edges matching criteria.
        """
        ...
