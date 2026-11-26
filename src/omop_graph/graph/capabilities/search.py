"""Lookup of concepts by name and the validation of search constraints."""

from __future__ import annotations

import functools
import logging
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import replace

from omop_alchemy.backends import BackendNotSupportedError, FullTextError
from omop_alchemy.cdm.model.vocabulary import Concept
from omop_alchemy.cdm.query import ConceptFilter

# Local Application Imports
from ..nodes import (
    LabelMatch,
    LabelMatchKind,
)
from ..queries import (
    q_concept_domain_ids,
    q_concept_name_fulltext,
    q_concept_name_ilike,
    q_concept_name_match,
    q_concept_vocabulary_ids,
)
from ..scope import ALL_CONCEPTS
from .enumeration import EnumerationMixin

logger = logging.getLogger(__name__)


class SearchMixin(EnumerationMixin):
    """Lookup of concepts by name and the validation of search constraints."""

    def concept_lookup(
        self,
        query_term: str,
        match_kind: LabelMatchKind,
        synonym: bool = False,
        concept_filter: ConceptFilter | None = None,
        sort: bool = True,
        limit: int | None = None,
    ) -> tuple[LabelMatch, ...]:
        """
        Resolve a query to concept_id(s).

        Parameters
        ----------
        query_term : str
            The term to search for.
        match_kind : LabelMatchKind
            The kind of match to perform (exact, starts with, fulltext, partial).
        synonym : bool
            If True, searches in Concept_Synonym instead of Concept.
        concept_filter : ConceptFilter, optional
            Additional filters for domain/vocabulary.
        sort : bool
            Whether to order the matches by quality.
        limit : int, optional
            The most matches to read.
        """
        input_query_term = self._normalise_query_term(query_term)
        if not input_query_term:
            return ()

        with self.session_factory() as session:
            if match_kind == LabelMatchKind.EXACT:
                fn = q_concept_name_match
            elif match_kind == LabelMatchKind.PARTIAL:
                fn = q_concept_name_ilike
            elif match_kind == LabelMatchKind.STARTS_WITH:
                fn = functools.partial(q_concept_name_ilike, anchored=True)
            elif match_kind == LabelMatchKind.FTS:
                fn = functools.partial(q_concept_name_fulltext, engine=session.get_bind(Concept))
            else:
                raise ValueError(f"Unsupported search mode: {match_kind}")
            try:
                cn = fn(
                    input_query_term,
                    concept_filter=concept_filter,
                    synonym=synonym,
                    sort=sort,
                )
            except (FullTextError, BackendNotSupportedError) as e:
                if match_kind == LabelMatchKind.FTS:
                    logger.info(e)
                    return ()
                raise

            if limit is not None:
                cn = cn.limit(limit)
            matches = tuple(
                LabelMatch(
                    input_query=input_query_term,
                    matched_concept_label=name,
                    matched_concept_id=int(cid),
                    match_kind=match_kind,
                    is_standard=is_standard,
                    is_active=is_active,
                    synonym=synonym,
                )
                for cid, name, is_standard, is_active in session.execute(cn)
            )
        return matches

    def search(
        self,
        term: str,
        kinds: Sequence[LabelMatchKind] = (LabelMatchKind.EXACT,),
        *,
        synonyms: bool = False,
        concept_filter: ConceptFilter = ALL_CONCEPTS,
        limit: int | None = None,
        identifiers: Iterable[int] = (),
    ) -> Iterator[int]:
        """
        Yield the concepts a term matches, best matches first, each once.

        The concepts named as *identifiers* come first, when they are in scope. Then each
        kind of match is tried in turn, on concept names and then, with *synonyms*, on
        synonyms, as in :meth:`concept_lookup`.

        Parameters
        ----------
        term : str
            The term to search for.
        kinds : Sequence[LabelMatchKind]
            The kinds of match, in the order they rank.
        synonyms : bool
            If True, also match synonyms.
        concept_filter : ConceptFilter
            The concepts that can match. By default every concept.
        limit : int, optional
            The most concepts to yield. Zero or less yields none.
        identifiers : Iterable[int]
            Concepts the term names as an identifier, such as by code.

        Yields
        ------
        int
            Concept ids.
        """
        found: dict[int, None] = {}

        def full() -> bool:
            return limit is not None and len(found) >= limit

        named = tuple(identifiers)
        if named and not full():
            in_scope = tuple(self.entities(replace(concept_filter, concept_ids=named)))
            for concept_id in in_scope[:limit]:
                found[concept_id] = None
                yield concept_id
        for kind in kinds:
            for on_synonyms in (False, True) if synonyms else (False,):
                if full():
                    return
                matches = self.concept_lookup(
                    term,
                    match_kind=kind,
                    synonym=on_synonyms,
                    concept_filter=concept_filter,
                    limit=None if limit is None else limit + len(found),
                )
                for match in matches:
                    if match.matched_concept_id not in found:
                        found[match.matched_concept_id] = None
                        yield match.matched_concept_id
                        if full():
                            return

    def concept_ids_by_label(self, label: str) -> tuple[int, ...]:
        """
        Find concept IDs that match the label exactly (case-insensitive).
        """
        label = self._normalise_query_term(label)
        with self.session_factory() as session:
            rows = session.execute(q_concept_name_match(label)).scalars()
        return tuple(rows)

    def _normalise_query_term(self, query_term: str) -> str:
        """
        Normalize a string for lookup (lowercase, single spaces).
        """
        return re.sub(r"\s+", " ", query_term.strip().lower())

    def check_concept_filter(self, constraints: ConceptFilter) -> None:
        if constraints.domains is not None:
            invalid = [d for d in constraints.domains if d not in self._valid_domains]
            if invalid:
                raise ValueError(
                    f"Invalid domain constraint(s): {invalid}. "
                    f"Available domains: {sorted(self._valid_domains)}"
                )

        if constraints.vocabularies is not None:
            invalid = [v for v in constraints.vocabularies if v not in self._valid_vocabularies]
            if invalid:
                raise ValueError(
                    f"Invalid vocabulary constraint(s): {invalid}. "
                    f"Available vocabularies: {sorted(self._valid_vocabularies)}"
                )

    @functools.cached_property
    def _valid_domains(self) -> frozenset[str]:
        with self.session_factory() as session:
            rows = session.execute(q_concept_domain_ids()).all()
        return frozenset(row.domain_id for row in rows)

    @functools.cached_property
    def _valid_vocabularies(self) -> frozenset[str]:
        with self.session_factory() as session:
            rows = session.execute(q_concept_vocabulary_ids()).all()
        return frozenset(row.vocabulary_id for row in rows)
