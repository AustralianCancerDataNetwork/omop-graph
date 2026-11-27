"""Vocabularies, domains and synonyms."""

from __future__ import annotations

import functools
from collections.abc import Iterable

# Local Application Imports
from ..language import OMOPLanguage
from ..nodes import (
    SynonymView,
    VocabularyView,
)
from ..queries import (
    q_domains,
    q_synonym_language_ids,
    q_synonyms,
    q_vocabularies,
)
from ._core import GraphCore


class VocabularyMixin(GraphCore):
    """Vocabularies, domains and synonyms."""

    @functools.cached_property
    def _vocabularies(self) -> tuple[VocabularyView, ...]:
        """The vocabularies, read once for the life of the graph."""
        with self.session_factory() as session:
            return tuple(
                VocabularyView(**row._mapping) for row in session.execute(q_vocabularies())
            )

    def vocabularies(self) -> tuple[VocabularyView, ...]:
        """
        Return every vocabulary, read once for the life of the graph.

        Returns
        -------
        tuple[VocabularyView, ...]
            The vocabularies.
        """
        return self._vocabularies

    def domain_concept_ids(self) -> dict[str, int]:
        """
        Return the concept id of each domain, keyed by ``domain_id``.

        Returns
        -------
        dict[str, int]
            The concept id of each domain.
        """
        with self.session_factory() as session:
            return {row.domain_id: row.domain_concept_id for row in session.execute(q_domains())}

    def synonyms(self, concept_ids: Iterable[int]) -> tuple[SynonymView, ...]:
        """
        Return the synonyms of concept_ids, however many, ordered by concept and name.

        Parameters
        ----------
        concept_ids : Iterable[int]
            The concepts whose synonyms are read.

        Returns
        -------
        tuple[SynonymView, ...]
            The synonyms.
        """
        with self.session_factory() as session:
            return tuple(
                SynonymView(**row._mapping)
                for row in self._rows_in_batches(session, q_synonyms, sorted(set(concept_ids)))
            )

    def synonym_language_ids(self) -> frozenset[int]:
        """
        Return the language concept ids that at least one synonym uses.

        Returns
        -------
        frozenset[int]
            The language concept ids.
        """
        with self.session_factory() as session:
            return frozenset(session.execute(q_synonym_language_ids()).scalars())

    def synonym_languages(self) -> tuple[OMOPLanguage, ...]:
        """
        Return the languages, with a known code, that at least one synonym uses.

        Returns
        -------
        tuple[OMOPLanguage, ...]
            The languages.
        """
        used = self.synonym_language_ids()
        return tuple(language for language in OMOPLanguage if language.concept_id in used)

    def translations(self, concept_ids: Iterable[int], language: OMOPLanguage) -> dict[int, str]:
        """
        Return the first synonym of each concept in language.

        OMOP does not rank synonyms, so the first is the one that sorts first by name.
        Concepts without a synonym in the language are left out.

        Parameters
        ----------
        concept_ids : Iterable[int]
            The concepts to translate.
        language : OMOPLanguage
            The language of the synonyms.

        Returns
        -------
        dict[int, str]
            The first synonym of each concept that has one.
        """
        found: dict[int, str] = {}
        with self.session_factory() as session:
            for row in self._rows_in_batches(
                session,
                lambda batch: q_synonyms(batch, language.concept_id),
                sorted(set(concept_ids)),
            ):
                found.setdefault(row.concept_id, row.name)
        return found

    def synonyms_for_concept(self, concept_id: int) -> tuple[str, ...]:
        """
        Retrieve all synonyms for a concept.

        Parameters
        ----------
        concept_id : int
            The concept.

        Returns
        -------
        tuple[str, ...]
            The synonym names.
        """
        return tuple(synonym.name for synonym in self.synonyms((concept_id,)))
