"""
Resolver strategies for mapping text to OMOP Concepts.

This module defines a hierarchy of `CandidateResolver` classes. Each resolver implements
a specific strategy (Exact Match, Partial Match, Full-Text Search) to find OMOP concepts
that match a given input string.

They are designed to be used in a `ResolverPipeline` where high-confidence resolvers
(Exact) are tried before lower-confidence ones (Partial/Fuzzy).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Optional, Tuple
import logging
import numpy as np

from omop_graph.graph.constraints import SearchConstraintConcept
from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.graph.nodes import LabelMatch, LabelMatchKind
from omop_graph.extensions.emb import (
    HAS_OMOP_EMB,
    EmbeddingIndexType,
    EmbeddingMetricType,
    get_neareast_concepts,
)

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
    matched_label : str
        The specific text in the database (name or synonym) that matched.
    """

    concept_id: int
    match_kind: LabelMatchKind
    matched_label: str
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
        text: str,
        constraints: Optional[SearchConstraintConcept] = None,
        sort: bool = False,
        **kwargs
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
        sort : bool, default False
            Whether to sort LabelMatch results by their internal relevance ranking.

        Returns
        -------
        Tuple[LabelMatch, ...]
            A tuple of raw label matches.
        """
        return tuple(
            kg.concept_lookup(
                label=text, 
                match_kind=self.match_kind, 
                synonym=self.synonym, 
                search_constraint=constraints,
                sort=sort
            )
        )

    def resolve(
        self,
        kg: KnowledgeGraph,
        text: str,
        constraints: Optional[SearchConstraintConcept] = None,
        **kwargs
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
        matches = self.get_matches(kg, text, constraints=constraints, **kwargs)
        hits = [
            CandidateHit(
                concept_id=m.concept_id,
                match_kind=self.match_kind,
                matched_label=m.matched_label,
                synonym=self.synonym
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
    """

    def __init__(self) -> None:
        super().__init__(match_kind=LabelMatchKind.EMBEDDING, synonym=False)

    
    def get_matches(
        self,
        kg: KnowledgeGraph,
        text: str,
        constraints: Optional[SearchConstraintConcept] = None,
        text_embedding: Optional[np.ndarray] = None,
        text_embedding_model: Optional[str] = None,
        metric_type: Optional[EmbeddingMetricType] = None,
        index_type: Optional[EmbeddingIndexType] = None,
        sort: bool = False,
    ) -> Tuple[LabelMatch, ...]:
        
        with kg.session_factory() as session:
            matches = get_neareast_concepts(
                session=session,
                kg=kg,
                text_embedding=text_embedding,
                text_embedding_model=text_embedding_model,
                concept_filter=constraints,
                metric_type=metric_type,
                index_type=index_type,
            )
            if matches is None:
                return ()
            if text_embedding is not None:
                assert text_embedding.shape[0] == 1, "text_embedding should have shape (1, embedding_dim) for a single query."
            assert len(matches) == 1, "Expected get_neareast_concepts to return a single dictionary given the text_embedding shape (1, embedding_dim)."
            matches = matches[0]  # Unpack the single dictionary from the tuple
            concept_views = kg.concept_views(
                concept_ids=tuple(matches.keys()),
                sort=sort
            )
            label_matches = tuple(
                LabelMatch(
                    input_label=text,
                    matched_label=cv.concept_name,
                    concept_id=int(cv.concept_id),
                    match_kind=LabelMatchKind.EMBEDDING,
                    is_standard=bool(cv.standard_concept),
                    is_active=cv.invalid_reason is None,
                )
                for cv in concept_views
            )
            return label_matches


# Default sequence of resolvers to be used in a pipeline
ALL_RESOLVERS: Tuple[CandidateResolver, ...] = (
    ExactLabelResolver(),
    ExactSynonymResolver(),
    PartialLabelResolver(),
    PartialSynonymResolver(),
    FullTextResolver(),
    FullTextSynonymResolver(),
)

if HAS_OMOP_EMB:
    ALL_RESOLVERS = (*ALL_RESOLVERS, EmbeddingResolver())