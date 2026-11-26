"""Lookup of concepts by name and the validation of search constraints."""

from __future__ import annotations

import functools
import logging
import re

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
from ._core import GraphCore

logger = logging.getLogger(__name__)


class SearchMixin(GraphCore):
    """Lookup of concepts by name and the validation of search constraints."""

    def concept_lookup(
        self,
        query_term: str,
        match_kind: LabelMatchKind,
        synonym: bool = False,
        search_constraint: ConceptFilter | None = None,
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
        search_constraint : ConceptFilter, optional
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
                    search_constraint=search_constraint,
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

    def check_search_constraints(self, constraints: ConceptFilter) -> None:
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
