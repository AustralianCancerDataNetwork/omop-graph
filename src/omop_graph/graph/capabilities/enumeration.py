"""Enumeration of concepts: all, roots, leaves, singletons and deprecated."""

from __future__ import annotations

from collections.abc import Callable, Generator

from omop_alchemy.cdm.query import ConceptFilter
from sqlalchemy.sql import Select

# Local Application Imports
from .._streaming import stream
from ..model.edges import (
    EdgeFilter,
)
from ..queries import (
    DEFAULT_CONCEPT_FILTER,
    q_entities,
    q_leaves,
    q_obsolete_concepts,
    q_roots,
    q_singletons,
)
from .edges import EdgesMixin


class EnumerationMixin(EdgesMixin):
    """Enumeration of concepts: all, roots, leaves, singletons and deprecated."""

    def entities(
        self, concept_filter: ConceptFilter = DEFAULT_CONCEPT_FILTER, limit: int | None = None
    ) -> Generator[int, None, None]:
        """
        Yield the ids of the concepts selected by concept_filter.

        Defaults to active standard and classification concepts. The ids are read through
        a server-side cursor, so they are never all held in memory. With limit, the lowest
        ids are yielded.

        Parameters
        ----------
        concept_filter : ConceptFilter
            The concepts to select.
        limit : int, optional
            The most ids to yield, the lowest first.

        Yields
        ------
        int
            The concept ids.
        """
        yield from self._stream_ids(q_entities(concept_filter, limit))

    def roots(
        self,
        concept_filter: ConceptFilter = DEFAULT_CONCEPT_FILTER,
        edge_filter: EdgeFilter | None = None,
        limit: int | None = None,
    ) -> Generator[int, None, None]:
        """
        Yield root concepts (no parents) among those selected by concept_filter.

        Without edge_filter the parents are the closure's level 1 ancestors. With it, a
        parent is the target of an outgoing edge it selects. With limit, the lowest ids are
        yielded.

        Parameters
        ----------
        concept_filter : ConceptFilter
            The concepts to select.
        edge_filter : EdgeFilter | None
            Which edges make a concept a parent or a child, the closure when None.
        limit : int, optional
            The most ids to yield, the lowest first.

        Yields
        ------
        int
            The concept ids.
        """
        return self._enumerate(q_roots, concept_filter, edge_filter, limit)

    def leaves(
        self,
        concept_filter: ConceptFilter = DEFAULT_CONCEPT_FILTER,
        edge_filter: EdgeFilter | None = None,
        limit: int | None = None,
    ) -> Generator[int, None, None]:
        """
        Yield leaf concepts (no children) among those selected by concept_filter.

        Without edge_filter the children are the closure's level 1 descendants. With it,
        a child is the source of an incoming edge it selects.

        Parameters
        ----------
        concept_filter : ConceptFilter
            The concepts to select.
        edge_filter : EdgeFilter | None
            Which edges make a concept a parent or a child, the closure when None.
        limit : int, optional
            The most ids to yield, the lowest first.

        Yields
        ------
        int
            The concept ids.
        """
        return self._enumerate(q_leaves, concept_filter, edge_filter, limit)

    def singletons(
        self,
        concept_filter: ConceptFilter = DEFAULT_CONCEPT_FILTER,
        edge_filter: EdgeFilter | None = None,
        limit: int | None = None,
    ) -> Generator[int, None, None]:
        """
        Yield singleton concepts (no parents and no children) among those selected by
        concept_filter. See :meth:`roots` for edge_filter.

        Parameters
        ----------
        concept_filter : ConceptFilter
            The concepts to select.
        edge_filter : EdgeFilter | None
            Which edges make a concept a parent or a child, the closure when None.
        limit : int, optional
            The most ids to yield, the lowest first.

        Yields
        ------
        int
            The concept ids.
        """
        return self._enumerate(q_singletons, concept_filter, edge_filter, limit)

    def _enumerate(
        self,
        build: Callable[[ConceptFilter, EdgeFilter | None, int | None], Select],
        concept_filter: ConceptFilter,
        edge_filter: EdgeFilter | None,
        limit: int | None,
    ) -> Generator[int, None, None]:
        """Stream build with edge_filter resolved, or every concept when it selects no edge."""
        if edge_filter is None:
            stmt = build(concept_filter, None, limit)
        else:
            resolved = self._resolve_edge_filter(edge_filter)
            stmt = (
                build(concept_filter, resolved, limit)
                if resolved
                else q_entities(concept_filter, limit)
            )
        return self._stream_ids(stmt)

    def _stream_ids(self, stmt: Select) -> Generator[int, None, None]:
        """Yield the first column of stmt through a server-side cursor."""
        with self.session_factory() as session:
            yield from session.execute(stream(stmt)).scalars()

    def obsolete_concepts(
        self,
        include_updated: bool = True,
        concept_filter: ConceptFilter | None = None,
        limit: int | None = None,
    ) -> Generator[int, None, None]:
        """
        Yield the ids of deprecated concepts.

        Parameters
        ----------
        include_updated : bool
            If True, include the concepts updated into others (``invalid_reason`` ``U``)
            besides the deleted ones (``D``).
        concept_filter : ConceptFilter, optional
            Narrows the concepts, for example to some vocabularies or domains.
        limit : int, optional
            The most ids to yield, the lowest first.

        Yields
        ------
        int
            The concept ids.
        """
        yield from self._stream_ids(q_obsolete_concepts(include_updated, concept_filter, limit))
