"""Relationships and edges between concepts."""

from __future__ import annotations

from collections.abc import Generator, Iterable
from dataclasses import replace
from itertools import product
from typing import Literal

from sqlalchemy import Row
from sqlalchemy.orm import Session
from sqlalchemy.sql import Select

# Local Application Imports
from ...extensions.omop_alchemy import (
    PredicateSubkind,
)
from .._streaming import stream
from ..edges import (
    DEFAULT_EDGE_FILTER,
    EdgeFilter,
    EdgeView,
)
from ..queries import (
    q_edges,
    q_relationships,
)
from .predicates import PredicatesMixin


class EdgesMixin(PredicatesMixin):
    """Relationships and edges between concepts."""

    def relationships(
        self,
        subjects: Iterable[int] | None,
        *,
        objects: Iterable[int] | None = None,
        edge_filter: EdgeFilter | None = DEFAULT_EDGE_FILTER,
        invert: bool = False,
    ) -> Generator[tuple[int, str, int], None, None]:
        """
        Query relationships between concepts.

        Any number of subjects and objects is allowed. With neither, every relationship
        the filter selects is read through a server-side cursor, so the table is never
        held in memory.

        Parameters
        ----------
        subjects : Iterable[int] | None
            Subject concept ids, or None for any.
        objects : Iterable[int] | None
            Object concept ids, or None for any.
        edge_filter : EdgeFilter, optional
            Which relationships to return. Defaults to ``DEFAULT_EDGE_FILTER``, which leaves
            out the metadata relationships. None selects every classified relationship.
            Relationships without a classification are skipped, as in :meth:`iter_edges`.
        invert : bool
            If True, swaps subjects and objects in the query and result.

        Yields
        ------
        Tuple[int, str, int]
            Triples of (subject_concept_id, relationship_id, object_concept_id).
            When ``invert=True``, the triple is (object_concept_id, relationship_id, subject_concept_id).
        """
        if invert:
            for s, p, o in self.relationships(
                subjects=objects,
                objects=subjects,
                edge_filter=edge_filter,
            ):
                yield o, p, s
            return

        resolved = self._resolve_edge_filter(edge_filter)
        if resolved is None:
            return

        if subjects is None and objects is None:
            statements = (stream(q_relationships(None, None, resolved)),)
        else:
            subject_batches = (None,) if subjects is None else tuple(self.batches(subjects))
            object_batches = (None,) if objects is None else tuple(self.batches(objects))
            statements = tuple(
                q_relationships(subject_batch, object_batch, resolved)
                for subject_batch, object_batch in product(subject_batches, object_batches)
            )
        yield from self._classified_triples(statements)

    def all_relationships(
        self, edge_filter: EdgeFilter | None = DEFAULT_EDGE_FILTER
    ) -> Generator[tuple[int, str, int], None, None]:
        """Yield every relationship selected by edge_filter, as :meth:`relationships` does."""
        yield from self.relationships(None, edge_filter=edge_filter)

    def _classified_triples(
        self, statements: Iterable[Select]
    ) -> Generator[tuple[int, str, int], None, None]:
        """Run the relationship triple queries and keep the triples of classified relationships."""
        with self.session_factory() as session:
            for stmt in statements:
                for s, p, o in session.execute(stmt):
                    if p in self._relationship_mapping:
                        yield s, p, o

    def edges(
        self,
        concept_ids: Iterable[int] | int,
        direction: Literal["in", "out"],
        edge_filter: EdgeFilter | None = DEFAULT_EDGE_FILTER,
    ) -> tuple[EdgeView, ...]:
        """
        Convenience method to retrieve all edges from one or multiple concepts.

        Parameters
        ----------
        concept_ids : int, Iterable[int]
            The source/target concept ID(s), however many.
        direction : str
            'out' for outgoing, 'in' for incoming.
        edge_filter : EdgeFilter, optional
            Which edges to return. Defaults to ``DEFAULT_EDGE_FILTER``, which leaves out the
            metadata relationships. None selects every classified relationship.
        """
        with self.session_factory() as session:
            edges = tuple(
                self.iter_edges(
                    session=session,
                    concept_ids=concept_ids,
                    direction=direction,
                    edge_filter=edge_filter,
                )
            )
        return edges

    def iter_edges(
        self,
        session: Session,
        concept_ids: int | Iterable[int],
        direction: Literal["in", "out"],
        edge_filter: EdgeFilter | None = DEFAULT_EDGE_FILTER,
        ordered: bool = False,
    ) -> Generator[EdgeView, None, None]:
        """Yield edges of concept_ids on session, classified from the relationship mapping.

        Edges whose relationship has no classification are skipped. If
        ``ordered=True``, edges are ordered by source concept, target concept,
        and relationship ID within each batch of ``LOOKUP_BATCH_SIZE`` concepts, making
        results reproducible. If ``ordered=False``, row order is unspecified and may
        differ between runs and backends.
        """
        resolved = self._resolve_edge_filter(edge_filter)
        if resolved is None:
            return

        ids = (concept_ids,) if isinstance(concept_ids, int) else concept_ids
        rows = self._rows_in_batches(
            session,
            lambda batch: q_edges(
                concept_ids=batch,
                direction=direction,
                edge_filter=resolved,
                ordered=ordered,
            ),
            ids,
        )
        for row in rows:
            if (edge := self._edge_view(row)) is not None:
                yield edge

    def edges_between(
        self,
        subject_ids: Iterable[int],
        object_ids: Iterable[int],
        edge_filter: EdgeFilter | None = DEFAULT_EDGE_FILTER,
    ) -> tuple[EdgeView, ...]:
        """
        Retrieve the edges that run from one of subject_ids to one of object_ids.

        Parameters
        ----------
        subject_ids : Iterable[int]
            The concepts the edges start from, however many.
        object_ids : Iterable[int]
            The concepts the edges end at, however many.
        edge_filter : EdgeFilter, optional
            Which edges to return. Defaults to ``DEFAULT_EDGE_FILTER``, which leaves out the
            metadata relationships. None selects every classified relationship.
        """
        resolved = self._resolve_edge_filter(edge_filter)
        if resolved is None:
            return ()
        subjects, objects = tuple(dict.fromkeys(subject_ids)), tuple(dict.fromkeys(object_ids))
        found: list[EdgeView] = []
        with self.session_factory() as session:
            for subject_batch in self.batches(subjects):
                for object_batch in self.batches(objects):
                    for row in session.execute(
                        q_edges(subject_batch, "out", resolved, others=object_batch)
                    ):
                        if (edge := self._edge_view(row)) is not None:
                            found.append(edge)
        return tuple(found)

    def _edge_view(self, row: Row) -> EdgeView | None:
        """The edge of a row, or None when its relationship has no classification."""
        mapping = self._relationship_mapping.get(row.predicate_id)
        if mapping is None:
            return None
        return EdgeView(
            **row._mapping,
            predicate_kind=mapping.predicate_kind,
            predicate_subkind=mapping.predicate_subkind,
        )

    def _resolve_edge_filter(self, edge_filter: EdgeFilter | None) -> EdgeFilter | None:
        """Fold the kind and subkind fields of edge_filter into ``predicate_ids``.

        Returns None when they select no relationship, so the caller has nothing
        to query. An empty set in any of the three fields selects none, where None selects all.

        Raises
        ------
        ValueError
            If ``predicate_subkinds`` names a subkind no relationship is classified under.
        """
        if edge_filter is None:
            edge_filter = EdgeFilter()
        if edge_filter.predicate_ids is not None and not edge_filter.predicate_ids:
            return None
        kinds, subkinds = edge_filter.predicate_kinds, edge_filter.predicate_subkinds
        if kinds is None and subkinds is None:
            return edge_filter

        if subkinds is not None:
            known = {
                PredicateSubkind(m.predicate_kind, m.predicate_subkind)
                for m in self._relationship_mapping.values()
            }
            if unknown := subkinds - known:
                raise ValueError(
                    f"Unknown predicate subkinds {sorted(map(str, unknown))}; "
                    f"valid subkinds are {sorted(map(str, known))}."
                )

        selected = frozenset(
            relationship_id
            for relationship_id, m in self._relationship_mapping.items()
            if (kinds is None or m.predicate_kind in kinds)
            and (
                subkinds is None
                or PredicateSubkind(m.predicate_kind, m.predicate_subkind) in subkinds
            )
        )
        if edge_filter.predicate_ids is not None:
            selected &= edge_filter.predicate_ids
        if not selected:
            return None
        return replace(
            edge_filter, predicate_ids=selected, predicate_kinds=None, predicate_subkinds=None
        )
