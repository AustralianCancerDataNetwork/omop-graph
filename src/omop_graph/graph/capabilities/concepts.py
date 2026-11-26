"""Concept views and enumeration of concepts."""

from __future__ import annotations

from collections.abc import Iterable

# Local Application Imports
from ..nodes import (
    ConceptView,
)
from ..queries import (
    q_concept_view,
    q_concept_views,
    q_concepts_by_code,
)
from ._core import GraphCore


class ConceptsMixin(GraphCore):
    """Concept views and enumeration of concepts."""

    def concept_view(self, concept_id: int) -> ConceptView:
        """
        Retrieve a single concept view by ID.

        Parameters
        ----------
        concept_id : int
            The OMOP Concept ID.

        Returns
        -------
        ConceptView
            The immutable view of the concept.
        """
        with self.session_factory() as session:
            row = session.execute(q_concept_view(concept_id)).one()
        return ConceptView.from_row(row)

    def concept_views(
        self, concept_ids: Iterable[int], sort: bool = True
    ) -> tuple[ConceptView, ...]:
        """
        Retrieve the views of several concepts, however many.

        Parameters
        ----------
        concept_ids : Iterable[int]
            OMOP Concept IDs.
        sort : bool
            If True, the views of each batch of ``LOOKUP_BATCH_SIZE`` ids are ordered by
            match quality: standard concepts first, then active ones, then shorter names.

        Returns
        -------
        tuple[ConceptView, ...]
            The views of the concepts that exist, each once.
        """
        with self.session_factory() as session:
            return tuple(
                ConceptView.from_row(row)
                for row in self._rows_in_batches(
                    session,
                    lambda batch: q_concept_views(batch, sort=sort),
                    dict.fromkeys(concept_ids),
                )
            )

    def concepts_by_code(self, vocabulary_id: str, concept_code: str) -> dict[int, bool]:
        """
        Look up every concept with a vocabulary ID and concept code.

        OMOP does not keep a code unique within its vocabulary, so there can be several.

        Parameters
        ----------
        vocabulary_id : str
            The vocabulary ID (e.g., 'SNOMED', 'RxNorm').
        concept_code : str
            The source code within that vocabulary.

        Returns
        -------
        dict[int, bool]
            Whether each matching concept is valid, by concept id in ascending order.
            Empty when no concept has the code.
        """
        with self.session_factory() as session:
            rows = session.execute(q_concepts_by_code(vocabulary_id, concept_code))
            return {int(row.concept_id): bool(row.is_valid) for row in rows}
