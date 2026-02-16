from dataclasses import dataclass
from omop_graph.graph.kg import KnowledgeGraph

from dataclasses import dataclass
from typing import Optional, Generator
from .resolvers import CandidateResolver, ResolverConfidence, CandidateHit, ALL_RESOLVERS
from ...graph.kg import KnowledgeGraph
from ...graph.constraints import SearchConstraintConcept

import logging
logger = logging.getLogger(__name__)


@dataclass
class ResolverPipeline:
    resolvers: tuple[CandidateResolver, ...]

    def __init__(
        self,
        resolvers: tuple[CandidateResolver, ...],
        *,
        stop_after_confidence: ResolverConfidence | None = None,
    ):
        # Sort the resolvers by confidence so the stop logic works correctly
        self.resolvers = tuple(sorted(resolvers, key=lambda r: r.confidence.value))
        self.stop_after_confidence = stop_after_confidence

    @classmethod
    def with_all_resolvers(cls, stop_after_confidence: ResolverConfidence | None = None) -> "ResolverPipeline":
        return cls(
            resolvers=ALL_RESOLVERS,
            stop_after_confidence=stop_after_confidence
        )

    def resolve(
        self,
        kg: KnowledgeGraph,
        text: str,
        limit_per_resolver: int | None = None,
        constraints: Optional[SearchConstraintConcept] = None,
    ) -> Generator[CandidateHit, None, None]:
        seen = set()

        for resolver in self.resolvers:
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
            )
            for hit in hits:
                if hit.concept_id not in seen:
                    seen.add(hit.concept_id)
                    yield hit
    
    