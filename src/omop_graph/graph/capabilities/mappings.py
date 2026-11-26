"""Mappings between concepts of different vocabularies."""

from __future__ import annotations

from collections.abc import Generator, Iterable

# Local Application Imports
from .._streaming import stream
from ..edges import (
    EdgeFilter,
)
from ..nodes import (
    MappingView,
)
from ..queries import (
    q_mappings,
)
from .edges import EdgesMixin


class MappingsMixin(EdgesMixin):
    """Mappings between concepts of different vocabularies."""

    def mappings(
        self,
        concept_ids: Iterable[int] | None = None,
        edge_filter: EdgeFilter | None = None,
        source: str | None = None,
    ) -> Generator[MappingView, None, None]:
        """
        Yield the edges starting from concepts, with the vocabularies of both ends.

        Edges from a concept to itself and edges without a classification are left out.
        Without *concept_ids* every edge is read through a server-side cursor.

        Parameters
        ----------
        concept_ids : Iterable[int] | None
            Concepts to start from, or None for all.
        edge_filter : EdgeFilter, optional
            Which edges to return, as in :meth:`relationships`.
        source : str, optional
            Only edges with this vocabulary on either side.
        """
        resolved = self._resolve_edge_filter(edge_filter)
        if resolved is None:
            return
        if concept_ids is None:
            statements = [stream(q_mappings(None, resolved, source))]
        else:
            ids = tuple(dict.fromkeys(concept_ids))
            statements = [q_mappings(batch, resolved, source) for batch in self.batches(ids)]
        with self.session_factory() as session:
            for stmt in statements:
                for row in session.execute(stmt):
                    if row.predicate_id in self._relationship_mapping:
                        yield MappingView(**row._mapping)
