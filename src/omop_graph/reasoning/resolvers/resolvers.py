"""
Resolver strategies for mapping text to OMOP Concepts.

This module defines a hierarchy of `CandidateResolver` classes. Each resolver implements
a specific strategy (Exact Match, Partial Match, Full-Text Search) to find OMOP concepts
that match a given input string.

They are designed to be used in a `ResolverPipeline` where high-confidence resolvers
(Exact) are tried before lower-confidence ones (Partial/Fuzzy).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING, Iterable, Optional, Tuple

import numpy as np

from omop_graph.extensions.emb import MissingExtensionError
from omop_graph.graph.constraints import SearchConstraintConcept
from omop_graph.graph.nodes import LabelMatch, LabelMatchKind
from omop_graph.utils.types import ResolverConfidence

if TYPE_CHECKING:
    from omop_emb import EmbeddingService
    from omop_graph.graph.kg import KnowledgeGraph
    from omop_llm import LLMClient


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CandidateHit:
    """
    A resolved candidate concept found by a resolver.

    Parameters
    ----------
    concept_id : int
        The OMOP Concept ID.
    resolver_confidence : ResolverConfidence
        The confidence level of the strategy that found this hit.
    matched_label : str
        The specific text in the database (name or synonym) that matched.
    """

    concept_id: int
    resolver_confidence: ResolverConfidence
    matched_label: str


class CandidateResolver(ABC):
    """
    Abstract interface for resolving free text to OMOP concept_ids.

    Resolvers are 'recall-oriented' and 'constraint-agnostic' regarding the
    validity of the concept in a specific context (that is handled later by reasoning).

    Attributes
    ----------
    confidence : ResolverConfidence
        The static confidence level assigned to hits found by this resolver.
    """

    confidence: ResolverConfidence

    @abstractmethod
    def get_matches(
        self,
        kg: "KnowledgeGraph",
        text: str,
        constraints: Optional[SearchConstraintConcept] = None,
        text_embedding: Optional[np.ndarray] = None,
        text_embedding_model: Optional[str] = None,
        embedding_client: Optional["LLMClient"] = None,
        embedding_service: Optional["EmbeddingService"] = None,
    ) -> Tuple[LabelMatch, ...]:
        """
        Execute the search strategy against the Knowledge Graph.

        Parameters
        ----------
        kg : KnowledgeGraph
            The graph instance.
        text : str
            The input text to search for.
        constraints : SearchConstraintConcept, optional
            Filters for domain/vocabulary.

        Returns
        -------
        Tuple[LabelMatch, ...]
            A tuple of raw label matches.
        """
        ...

    def resolve(
        self,
        kg: "KnowledgeGraph",
        text: str,
        constraints: Optional[SearchConstraintConcept] = None,
        limit: Optional[int] = None,
        text_embedding: Optional[np.ndarray] = None,
        text_embedding_model: Optional[str] = None,
        embedding_client: Optional["LLMClient"] = None,
        embedding_service: Optional["EmbeddingService"] = None,
    ) -> Iterable[CandidateHit]:
        """
        Public API to find and format candidates.

        Parameters
        ----------
        kg : KnowledgeGraph
            The graph instance.
        text : str
            The input text.
        constraints : SearchConstraintConcept, optional
            Filters.
        limit : int, optional
            Max number of hits to return.

        Returns
        -------
        Iterable[CandidateHit]
            The formatted candidate hits.
        """
        matches = self.get_matches(
            kg,
            text,
            constraints=constraints,
            text_embedding=text_embedding,
            text_embedding_model=text_embedding_model,
            embedding_client=embedding_client,
            embedding_service=embedding_service,
        )
        hits = [
            CandidateHit(
                concept_id=m.concept_id,
                resolver_confidence=self.confidence,
                matched_label=m.matched_label,
            )
            for m in matches
        ]
        return hits[:limit] if limit else hits


class ExactLabelResolver(CandidateResolver):
    """
    Strategy: Exact case-insensitive match on `Concept.concept_name`.
    """

    confidence = ResolverConfidence.EXACT

    def get_matches(
        self,
        kg: "KnowledgeGraph",
        text: str,
        constraints: Optional[SearchConstraintConcept] = None,
        text_embedding: Optional[np.ndarray] = None,
        text_embedding_model: Optional[str] = None,
        embedding_client: Optional["LLMClient"] = None,
        embedding_service: Optional["EmbeddingService"] = None,
    ) -> Tuple[LabelMatch, ...]:
        return tuple(kg.label_lookup(text, search_constraint=constraints))


class ExactSynonymResolver(CandidateResolver):
    """
    Strategy: Exact case-insensitive match on `Concept_Synonym.concept_synonym_name`.
    """

    confidence = ResolverConfidence.EXACT_SYNONYM

    def get_matches(
        self,
        kg: "KnowledgeGraph",
        text: str,
        constraints: Optional[SearchConstraintConcept] = None,
        text_embedding: Optional[np.ndarray] = None,
        text_embedding_model: Optional[str] = None,
        embedding_client: Optional["LLMClient"] = None,
        embedding_service: Optional["EmbeddingService"] = None,
    ) -> Tuple[LabelMatch, ...]:
        return tuple(kg.synonym_lookup(text, search_constraint=constraints))


class FullTextResolver(CandidateResolver):
    """
    Strategy: Postgres Full-Text Search (tsvector) on `Concept.concept_name`.
    Matches irrespective of word order (e.g., "Kidney Cancer" -> "Cancer of Kidney").
    """

    confidence = ResolverConfidence.FULLTEXT

    def get_matches(
        self,
        kg: "KnowledgeGraph",
        text: str,
        constraints: Optional[SearchConstraintConcept] = None,
        text_embedding: Optional[np.ndarray] = None,
        text_embedding_model: Optional[str] = None,
        embedding_client: Optional["LLMClient"] = None,
        embedding_service: Optional["EmbeddingService"] = None,
    ) -> Tuple[LabelMatch, ...]:
        return tuple(
            kg.fulltext_lookup(text, search_constraint=constraints, fuzzy=False)
        )


class FullTextSynonymResolver(CandidateResolver):
    """
    Strategy: Postgres Full-Text Search (tsvector) on `Concept_Synonym.concept_synonym_name`.
    """

    confidence = ResolverConfidence.FULLTEXT_SYNONYM

    def get_matches(
        self,
        kg: "KnowledgeGraph",
        text: str,
        constraints: Optional[SearchConstraintConcept] = None,
        text_embedding: Optional[np.ndarray] = None,
        text_embedding_model: Optional[str] = None,
        embedding_client: Optional["LLMClient"] = None,
        embedding_service: Optional["EmbeddingService"] = None,
    ) -> Tuple[LabelMatch, ...]:
        return tuple(
            kg.fulltext_lookup(text, search_constraint=constraints, fuzzy=True)
        )


class PartialLabelResolver(CandidateResolver):
    """
    Strategy: Substring match (ILIKE %term%) on `Concept.concept_name`.
    Ranked by similarity heuristics (starts_with, length diff).
    """

    confidence = ResolverConfidence.PARTIAL

    def get_matches(
        self,
        kg: "KnowledgeGraph",
        text: str,
        constraints: Optional[SearchConstraintConcept] = None,
        text_embedding: Optional[np.ndarray] = None,
        text_embedding_model: Optional[str] = None,
        embedding_client: Optional["LLMClient"] = None,
        embedding_service: Optional["EmbeddingService"] = None,
    ) -> Tuple[LabelMatch, ...]:
        matches = kg.label_lookup(text, fuzzy=True, search_constraint=constraints)
        ranked = sorted(
            matches, key=lambda m: self._similarity_score(text, m.matched_label)
        )
        return tuple(ranked)

    @staticmethod
    def _similarity_score(query: str, label: str) -> tuple:
        q = query.lower()
        l = label.lower()
        return (
            not l.startswith(q),  # Prefer matches starting with query (False < True)
            l.count(" "),  # Prefer fewer words
            abs(len(l) - len(q)),  # Prefer closer length
        )


class PartialSynonymResolver(PartialLabelResolver):
    """
    Strategy: Substring match (ILIKE %term%) on `Concept_Synonym.concept_synonym_name`.
    Inherits ranking logic from PartialLabelResolver.
    """

    confidence = ResolverConfidence.PARTIAL_SYNONYM

    def get_matches(
        self,
        kg: "KnowledgeGraph",
        text: str,
        constraints: Optional[SearchConstraintConcept] = None,
        text_embedding: Optional[np.ndarray] = None,
        text_embedding_model: Optional[str] = None,
        embedding_client: Optional["LLMClient"] = None,
        embedding_service: Optional["EmbeddingService"] = None,
    ) -> Tuple[LabelMatch, ...]:
        matches = kg.synonym_lookup(text, fuzzy=True, search_constraint=constraints)
        ranked = sorted(
            matches, key=lambda m: self._similarity_score(text, m.matched_label)
        )
        return tuple(ranked)


class EmbeddingResolver(CandidateResolver):
    """
    Strategy: Retrieve nearest concepts from stored concept embeddings.

    This resolver is intentionally conservative:
    - it only uses embeddings already stored in the KG
    - it will only compute an embedding on the fly for the query text itself
    - if embedding infrastructure is configured but unusable, it logs a warning
      and yields no hits rather than silently pretending semantic retrieval ran
    """

    confidence = ResolverConfidence.EMBEDDING

    def __init__(self, candidate_limit: int = 50):
        self.candidate_limit = candidate_limit

    def get_matches(
        self,
        kg: "KnowledgeGraph",
        text: str,
        constraints: Optional[SearchConstraintConcept] = None,
        text_embedding: Optional[np.ndarray] = None,
        text_embedding_model: Optional[str] = None,
        embedding_client: Optional["LLMClient"] = None,
        embedding_service: Optional["EmbeddingService"] = None,
    ) -> Tuple[LabelMatch, ...]:
        if text_embedding_model is None:
            logger.debug("EmbeddingResolver skipped because no text_embedding_model was provided.")
            return ()

        try:
            with kg.session_factory() as session:
                if not kg.emb.is_model_registered(session=session, model_name=text_embedding_model):
                    logger.warning(
                        "Embedding resolver requested model '%s', but it is not registered in the KG. "
                        "Semantic retrieval will be skipped.",
                        text_embedding_model,
                    )
                    return ()

                if not kg.emb.has_any_embeddings(session=session, embedding_model_name=text_embedding_model):
                    logger.warning(
                        "Embedding resolver requested model '%s', but no concept embeddings are stored in the KG. "
                        "Semantic retrieval will be skipped.",
                        text_embedding_model,
                    )
                    return ()

                query_embedding = self._resolve_query_embedding(
                    kg=kg,
                    session=session,
                    text=text,
                    constraints=constraints,
                    text_embedding=text_embedding,
                    text_embedding_model=text_embedding_model,
                    embedding_client=embedding_client,
                    embedding_service=embedding_service,
                )
                if query_embedding is None:
                    logger.warning(
                        "Embedding resolver could not obtain a query embedding for '%s'. "
                        "Semantic retrieval will be skipped.",
                        text,
                    )
                    return ()

                nearest_stmt = kg.emb.get_nearest_concepts(
                    session=session,
                    embedding_model_name=text_embedding_model,
                    text_embedding=query_embedding,
                    domains=constraints.domains if constraints is not None else None,
                    vocabularies=constraints.vocabs if constraints is not None else None,
                    require_standard=constraints.require_standard if constraints is not None else False,
                    limit=self.candidate_limit,
                )

                if not nearest_stmt:
                    logger.warning(
                        "Embedding resolver found no stored concept embeddings matching the current constraints "
                        "for model '%s'. Semantic retrieval will be skipped for this query.",
                        text_embedding_model,
                    )
                    return ()

                return tuple(
                    LabelMatch(
                        input_label=text,
                        matched_label=row.concept_name,
                        concept_id=int(row.concept_id),
                        match_kind=LabelMatchKind.EMBEDDING,
                        is_standard=bool(row.is_standard),
                        is_active=bool(row.is_active),
                    )
                    for row in nearest_stmt
                )
        except MissingExtensionError:
            logger.info(
                "Embedding resolver not available. Install omop-graph[emb] for PostgreSQL-backed semantic retrieval "
                "or install omop-emb with the backend extra you need."
            )
            return ()

    def _resolve_query_embedding(
        self,
        kg: "KnowledgeGraph",
        session,
        text: str,
        constraints: Optional[SearchConstraintConcept],
        text_embedding: Optional[np.ndarray],
        text_embedding_model: str,
        embedding_client: Optional["LLMClient"],
        embedding_service: Optional["EmbeddingService"],
    ) -> Optional[list[float]]:
        if text_embedding is not None:
            assert text_embedding.shape[0] == 1 and text_embedding.ndim == 2, (
                "Text embedding must be a 2D vector with first dim = 1."
            )
            return text_embedding.tolist()[0]

        exact_matches = list(kg.label_lookup(text, search_constraint=constraints))
        exact_matches.extend(list(kg.synonym_lookup(text, search_constraint=constraints)))
        exact_concept_ids = tuple(dict.fromkeys(match.concept_id for match in exact_matches))
        service = embedding_service or kg.emb_service
        try:
            query_embedding = service.get_or_create_query_embedding(
                session=session,
                model_name=text_embedding_model,
                query_text=text,
                embedding_client=embedding_client,
                reusable_concept_ids=exact_concept_ids,
            )
        except RuntimeError:
            return None
        assert query_embedding.shape[0] == 1 and query_embedding.ndim == 2, (
            "Query embedding must be a 2D vector with first dim = 1."
        )
        return query_embedding.tolist()[0]


# Default sequence of resolvers to be used in a pipeline
ALL_RESOLVERS = (
    ExactLabelResolver(),
    ExactSynonymResolver(),
    PartialLabelResolver(),
    PartialSynonymResolver(),
    FullTextResolver(),
    FullTextSynonymResolver(),
    EmbeddingResolver(),
)
