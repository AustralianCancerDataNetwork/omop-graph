"""
Resolver strategies for mapping text to OMOP Concepts.

This module defines a hierarchy of `CandidateResolver` classes. Each resolver implements
a specific strategy (Exact Match, Partial Match, Full-Text Search) to find OMOP concepts
that match a given input string.

They are designed to be used in a `ResolverPipeline` where high-confidence resolvers
(Exact) are tried before lower-confidence ones (Partial/Fuzzy).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from omop_alchemy.cdm.query import ConceptFilter

from omop_graph.graph.capabilities.embeddings import (
    HAS_OMOP_EMB,
    EmbeddingModelNotRegisteredError,
    MissingExtensionError,
)
from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.graph.model.matches import LabelMatch, LabelMatchKind
from omop_graph.graph.model.scope import ALL_CONCEPTS

if TYPE_CHECKING:
    from omop_graph.graph.kg import KnowledgeGraph

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CandidateHit:
    """
    A resolved candidate concept found by a resolver.

    Parameters
    ----------
    concept_id : int
        The OMOP Concept ID.
    match_kind : LabelMatchKind
        The kind of match of this hit.
    matched_concept_label : str
        The specific text in the database (name or synonym) that matched.
    synonym : bool
        Whether ``matched_concept_label`` came from a synonym rather than
        the concept's own name.
    """

    concept_id: int
    match_kind: LabelMatchKind
    matched_concept_label: str
    synonym: bool


class CandidateResolver:
    """
    Interface for resolving free text to OMOP concept_ids.

    Attributes
    ----------
    match_kind : LabelMatchKind
        The kind of match that this resolver produces.
    """

    def __init__(
        self,
        match_kind: LabelMatchKind,
        synonym: bool,
    ) -> None:
        super().__init__()

        self._match_kind = match_kind
        self._synonym = synonym

    @property
    def match_kind(self) -> LabelMatchKind:
        return self._match_kind

    @property
    def synonym(self) -> bool:
        return self._synonym

    def get_matches(
        self,
        kg: KnowledgeGraph,
        query: str,
        concept_filter: ConceptFilter | None = None,
        sort: bool = False,
        limit: int | None = None,
        **kwargs,
    ) -> tuple[LabelMatch, ...]:
        """
        Execute the search strategy against the Knowledge Graph.

        Parameters
        ----------
        kg : KnowledgeGraph
            The graph instance.
        query : str
            The input query to search for.
        concept_filter : ConceptFilter, optional
            Filters for domain/vocabulary. A scope that must not set ``limit``.
        sort : bool, default False
            Whether to sort LabelMatch results by their internal relevance ranking.
        limit : int, optional
            The most matches to return.

        Returns
        -------
        Tuple[LabelMatch, ...]
            A tuple of raw label matches.
        """
        return tuple(
            kg.concept_lookup(
                query_term=query,
                match_kind=self.match_kind,
                synonym=self.synonym,
                concept_filter=concept_filter,
                sort=sort,
                limit=limit,
            )
        )

    def resolve(
        self,
        kg: KnowledgeGraph,
        query: str,
        concept_filter: ConceptFilter | None = None,
        limit: int | None = None,
        **kwargs,
    ) -> Iterable[CandidateHit]:
        """
        Public API to find and format candidates.

        Parameters
        ----------
        kg : KnowledgeGraph
            The graph instance.
        query : str
            The input query.
        concept_filter : ConceptFilter, optional
            Filters for the concepts to consider in the search. A scope that must not set
            ``limit``.
        limit : int, optional
            The most candidates to return.

        Returns
        -------
        Iterable[CandidateHit]
            The formatted candidate hits.
        """
        matches = self.get_matches(kg, query, concept_filter=concept_filter, limit=limit, **kwargs)
        hits = [
            CandidateHit(
                concept_id=m.matched_concept_id,
                match_kind=self.match_kind,
                matched_concept_label=m.matched_concept_label,
                synonym=self.synonym,
            )
            for m in matches
        ]
        return hits


class ExactLabelResolver(CandidateResolver):
    """
    Strategy: Exact case-insensitive match on `Concept.concept_name`.
    """

    def __init__(self) -> None:
        super().__init__(match_kind=LabelMatchKind.EXACT, synonym=False)


class ExactSynonymResolver(CandidateResolver):
    """
    Strategy: Exact case-insensitive match on `Concept_Synonym.concept_synonym_name`.
    """

    def __init__(self) -> None:
        super().__init__(match_kind=LabelMatchKind.EXACT, synonym=True)


class FullTextResolver(CandidateResolver):
    """
    Strategy: Postgres Full-Text Search (tsvector) on `Concept.concept_name`.
    Matches irrespective of word order (e.g., "Kidney Cancer" -> "Cancer of Kidney").
    """

    def __init__(self) -> None:
        super().__init__(match_kind=LabelMatchKind.FTS, synonym=False)


class FullTextSynonymResolver(CandidateResolver):
    """
    Strategy: Postgres Full-Text Search (tsvector) on `Concept_Synonym.concept_synonym_name`.
    """

    def __init__(self) -> None:
        super().__init__(match_kind=LabelMatchKind.FTS, synonym=True)


class PartialLabelResolver(CandidateResolver):
    """
    Strategy: Substring match (ILIKE %term%) on `Concept.concept_name`.
    Ranked by similarity heuristics (starts_with, length diff).
    """

    def __init__(self) -> None:
        super().__init__(match_kind=LabelMatchKind.PARTIAL, synonym=False)


class PartialSynonymResolver(CandidateResolver):
    """
    Strategy: Substring match (ILIKE %term%) on `Concept_Synonym.concept_synonym_name`.
    Inherits ranking logic from PartialLabelResolver.
    """

    def __init__(self) -> None:
        super().__init__(match_kind=LabelMatchKind.PARTIAL, synonym=True)


class EmbeddingResolver(CandidateResolver):
    """
    Strategy: Retrieve nearest concepts from stored concept embeddings.
    Currently only for synonym=False as seamntic similarity should be preserved
    in the primary name. Could be extended to synonym=True if needed.

    Notes
    -----
    Only available if the omop_emb extension is installed and configured.
    """

    def __init__(self) -> None:
        super().__init__(match_kind=LabelMatchKind.EMBEDDING, synonym=False)
        if not HAS_OMOP_EMB:
            raise MissingExtensionError("EmbeddingResolver")

    def get_matches(
        self,
        kg: KnowledgeGraph,
        query: str,
        concept_filter: ConceptFilter | None = None,
        sort: bool = False,
        limit: int | None = None,
        query_embedding: np.ndarray | None = None,
        **kwargs,
    ) -> tuple[LabelMatch, ...]:
        if not HAS_OMOP_EMB:
            raise MissingExtensionError("EmbeddingResolver")

        if query_embedding is None:
            logger.warning(
                f"No text embedding provided for {self.__class__.__name__}. Returning no matches."
            )
            return ()
        assert query_embedding.shape[0] == 1, (
            "query_embedding should have shape (1, embedding_dim) for a single query."
        )
        if not kg.has_embeddings:
            logger.info("The KG has no embedding store. Returning no matches.")
            return ()

        try:
            matches = kg.nearest_concepts(
                query_embedding,
                concept_filter=concept_filter if concept_filter is not None else ALL_CONCEPTS,
                k=limit,
            )
        except EmbeddingModelNotRegisteredError as error:
            logger.info(error)
            return ()
        concept_views = kg.concept_views(
            concept_ids=tuple(m.concept_id for m in matches), sort=sort
        )
        label_matches = tuple(
            LabelMatch(
                input_query=query,
                matched_concept_label=cv.concept_name,
                matched_concept_id=int(cv.concept_id),
                match_kind=LabelMatchKind.EMBEDDING,
                is_standard=cv.standard_concept,
                is_active=cv.is_active,
                synonym=False,  # Embedding matches are based on the primary name, not synonyms
            )
            for cv in concept_views
        )
        return label_matches


# Default sequence of resolvers to be used in a pipeline
ALL_RESOLVERS: tuple[CandidateResolver, ...] = tuple(
    sorted(
        (
            ExactLabelResolver(),
            ExactSynonymResolver(),
            PartialLabelResolver(),
            PartialSynonymResolver(),
            FullTextResolver(),
            FullTextSynonymResolver(),
            *((EmbeddingResolver(),) if HAS_OMOP_EMB else ()),
        ),
        key=lambda resolver: (resolver.match_kind, resolver.synonym),
    )
)
