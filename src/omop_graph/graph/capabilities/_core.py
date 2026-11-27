"""The shared state and batched lookups every capability builds on."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from itertools import batched, product
from typing import ClassVar, overload

from sqlalchemy import Row
from sqlalchemy.orm import Session
from sqlalchemy.sql import Select

# Local Application Imports
from ...extensions.omop_alchemy import (
    RelationshipMappingElement,
)


class GraphCore:
    """The session factory, the relationship classification and batched lookups."""

    LOOKUP_BATCH_SIZE: ClassVar[int] = 5_000
    """Most concept ids sent in one ``IN`` list."""

    session_factory: Callable[[], Session]
    _relationship_mapping: dict[str, RelationshipMappingElement]

    @classmethod
    def batches[T](cls, ids: Iterable[T]) -> Iterator[tuple[T, ...]]:
        """
        Split ids into tuples of at most ``LOOKUP_BATCH_SIZE`` ids, in order.

        Every query that takes a list of ids runs once for each batch, which keeps the
        number of bind parameters of a statement under the limit of the database. The ids
        can be concept ids or the CURIEs of an adapter.
        """
        return batched(ids, cls.LOOKUP_BATCH_SIZE)

    @overload
    def _pair_batches(
        self, subjects: Iterable[int], objects: Iterable[int]
    ) -> Iterator[tuple[tuple[int, ...], tuple[int, ...]]]: ...

    @overload
    def _pair_batches(
        self, subjects: Iterable[int] | None, objects: Iterable[int] | None
    ) -> Iterator[tuple[tuple[int, ...] | None, tuple[int, ...] | None]]: ...

    def _pair_batches(
        self, subjects: Iterable[int] | None, objects: Iterable[int] | None
    ) -> Iterator[tuple[tuple[int, ...] | None, tuple[int, ...] | None]]:
        """
        Pair the batches of subjects with the batches of objects.

        A side that is None is not split and comes back as None in every pair. A side that
        is given but empty gives no pair, so a query over it runs zero times.
        """
        subject_batches = (None,) if subjects is None else tuple(self.batches(subjects))
        object_batches = (None,) if objects is None else tuple(self.batches(objects))
        return product(subject_batches, object_batches)

    def _rows_in_batches(
        self, session: Session, build: Callable[[tuple[int, ...]], Select], ids: Iterable[int]
    ) -> Iterator[Row]:
        """Yield the rows of ``build(batch)`` for each batch of ids, in the order of ids."""
        for batch in self.batches(ids):
            yield from session.execute(build(batch))
