"""
Pipeline for executing multiple candidate resolvers in sequence.

This module defines the `ResolverPipeline`, which orchestrates the execution
of various search strategies defined by CandidateResolver. It supports early stopping based on
the type of the resolver.
"""

from __future__ import annotations

import logging
from typing import Generator, Optional, Tuple, Type

# Local Application Imports
from omop_graph.graph.constraints import SearchConstraintConcept
from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.reasoning.resolvers.resolvers import (
    ALL_RESOLVERS,
    CandidateHit,
    CandidateResolver,
)

logger = logging.getLogger(__name__)


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
        The sequence of resolvers to execute.
    stop_after_resolver : str, optional
        If specified, the pipeline will stop after executing the resolver with this name.

    Notes
    -----
    TODO: Have it stop after a confidence score on a match. That requries some sort of callback into the resolve step to
    stop getting hits from the resolver and to get the confidence score of the match. 
    """
    def __init__(
        self,
        resolvers: Tuple[CandidateResolver, ...],
        stop_after_resolver: Optional[str | Type[CandidateResolver]] = None
    ):  
        self._resolver_map = {type(r): r for r in resolvers}
    
        if len(self._resolver_map) != len(resolvers):
            raise ValueError("Duplicate resolver types detected in pipeline.")

        self._resolvers = resolvers
        self._stop_at = None
        if stop_after_resolver:
            # If they passed a string name, find the matching type
            if isinstance(stop_after_resolver, str):
                match = next((t for t in self._resolver_map if t.__name__ == stop_after_resolver), None)
                if not match:
                    raise ValueError(f"{stop_after_resolver} not in pipeline.")
                self._stop_at = match
            else:
                # If they passed the class itself
                if stop_after_resolver not in self._resolver_map:
                    raise ValueError(f"{stop_after_resolver.__name__} not in pipeline.")
                self._stop_at = stop_after_resolver

    def __repr__(self) -> str:
        resolver_names = [type(r).__name__ for r in self.resolvers]
        return f"ResolverPipeline(resolvers={resolver_names}, stop_after_resolver={self.stop_at.__name__ if self.stop_at else None})"
    

    @property
    def resolvers(self) -> Tuple[CandidateResolver, ...]:
        return self._resolvers
    
    @property
    def stop_at(self) -> Optional[Type[CandidateResolver]]:
        """
        The class type of the resolver where the pipeline is configured to truncate.
        Returns None if the pipeline is configured to run to completion.
        """
        return self._stop_at

        
    @classmethod
    def with_all_resolvers(cls) -> "ResolverPipeline":
        """
        Create a pipeline configured with all available resolvers.

        Returns
        -------
        ResolverPipeline
            A fully configured pipeline instance.
        """
        # All resolvers
        return cls(resolvers=ALL_RESOLVERS, stop_after_resolver=None)

    def resolve(
        self,
        kg: KnowledgeGraph,
        text: str,
        constraints: Optional[SearchConstraintConcept] = None,
        **kwargs
    ) -> Generator[CandidateHit, None, None]:
        """
        Execute the pipeline to find candidate concepts for the input text.

        Parameters
        ----------
        kg : KnowledgeGraph
            The graph instance used for lookups.
        text : str
            The input text to resolve.
        constraints : SearchConstraintConcept, optional
            Domain or vocabulary restrictions to apply to the search.
            Determines also the number of candidates returned for each resolver using the `limit` field. If None, no additional filtering is applied.

        Yields
        -------
        CandidateHit
            Candidate concepts found by the resolvers.
        """
        seen = set()

        for resolver in self.resolvers:
            hits = resolver.resolve(
                kg,
                text,
                constraints=constraints,
                **kwargs
            )

            for hit in hits:
                if hit.concept_id not in seen:
                    seen.add(hit.concept_id)
                    yield hit

            # Early stopping
            if type(resolver) is self._stop_at:
                logger.info(f"Stopping pipeline after resolver {type(resolver).__name__} as configured.")
                break