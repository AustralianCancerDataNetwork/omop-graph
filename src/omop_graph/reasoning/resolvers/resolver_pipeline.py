"""
Pipeline for executing multiple candidate resolvers in sequence.

This module defines the `ResolverPipeline`, which orchestrates the execution
of various search strategies (exact match, synonym match, etc.) to find
candidate OMOP concepts for a given text. It supports early stopping based on
confidence thresholds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generator, Optional, Tuple

# Local Application Imports
from omop_graph.graph.constraints import SearchConstraintConcept
from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.reasoning.resolvers.resolvers import (
    ALL_RESOLVERS,
    CandidateHit,
    CandidateResolver,
    EmbeddingResolver,
    ResolverConfidence,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from omop_emb import EmbeddingService


@dataclass
class ResolverPipeline:
    """
    A sequence of resolvers to be executed in order of confidence.

    The pipeline runs high-confidence resolvers first (e.g., exact matches).
    If a match is found and the confidence is sufficient, it can stop early,
    preventing the execution of lower-confidence (and potentially slower/noisier)
    resolvers.

    Parameters
    ----------
    resolvers : tuple[CandidateResolver, ...]
        The sequence of resolvers to execute. Automatically sorted by confidence.
    stop_after_confidence : ResolverConfidence, optional
        If set, the pipeline stops executing further resolvers once a match
        has been found by a resolver with this confidence level or better (lower value).
    """

    resolvers: Tuple[CandidateResolver, ...]
    stop_after_confidence: Optional[ResolverConfidence]

    def __init__(
        self,
        resolvers: Tuple[CandidateResolver, ...],
        *,
        stop_after_confidence: Optional[ResolverConfidence] = None,
    ):
        # Sort the resolvers by confidence (lower value = higher confidence)
        object.__setattr__(
            self,
            "resolvers",
            tuple(sorted(resolvers, key=lambda r: r.confidence.value)),
        )
        object.__setattr__(self, "stop_after_confidence", stop_after_confidence)

    @classmethod
    def with_all_resolvers(
        cls,
        stop_after_confidence: Optional[ResolverConfidence] = None,
        *,
        include_embedding_resolver: bool = True,
        embedding_candidate_limit: int = 50,
    ) -> "ResolverPipeline":
        """
        Create a pipeline configured with all available resolvers.

        Parameters
        ----------
        stop_after_confidence : ResolverConfidence, optional
            The confidence threshold for early stopping.

        Returns
        -------
        ResolverPipeline
            A fully configured pipeline instance.
        """
        resolvers = tuple(
            resolver
            for resolver in ALL_RESOLVERS
            if not isinstance(resolver, EmbeddingResolver)
        )
        if include_embedding_resolver:
            resolvers = resolvers + (EmbeddingResolver(candidate_limit=embedding_candidate_limit),)

        return cls(resolvers=resolvers, stop_after_confidence=stop_after_confidence)

    def resolve(
        self,
        kg: KnowledgeGraph,
        text: str,
        limit_per_resolver: Optional[int] = None,
        constraints: Optional[SearchConstraintConcept] = None,
        text_embedding=None,
        text_embedding_model: Optional[str] = None,
        embedding_client=None,
        embedding_service: Optional["EmbeddingService"] = None,
    ) -> Generator[CandidateHit, None, None]:
        """
        Execute the pipeline to find candidate concepts for the input text.

        Parameters
        ----------
        kg : KnowledgeGraph
            The graph instance used for lookups.
        text : str
            The input text to resolve.
        limit_per_resolver : int, optional
            Maximum number of hits to return per resolver strategy.
        constraints : SearchConstraintConcept, optional
            Domain or vocabulary restrictions to apply to the search.

        Yields
        -------
        CandidateHit
            Candidate concepts found by the resolvers.
        """
        seen = set()

        for resolver in self.resolvers:
            # Early stopping check
            if (
                len(seen) > 0
                and self.stop_after_confidence is not None
                and resolver.confidence.value > self.stop_after_confidence.value
            ):
                break

            hits = resolver.resolve(
                kg,
                text,
                limit=limit_per_resolver,
                constraints=constraints,
                text_embedding=text_embedding,
                text_embedding_model=text_embedding_model,
                embedding_client=embedding_client,
                embedding_service=embedding_service,
            )

            for hit in hits:
                if hit.concept_id not in seen:
                    seen.add(hit.concept_id)
                    yield hit
