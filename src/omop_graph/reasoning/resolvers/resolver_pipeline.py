from dataclasses import dataclass

from omop_graph.graph import KnowledgeGraph
from .base import CandidateResolver, CandidateHit, ResolverConfidence


@dataclass
class ResolverPipeline:
    resolvers: tuple[CandidateResolver, ...]

    def __init__(
        self,
        resolvers: tuple[CandidateResolver, ...],
        *,
        stop_after_confidence: ResolverConfidence | None = None,
    ):
        self.resolvers = resolvers
        self.stop_after_confidence = stop_after_confidence

    def resolve(
        self,
        kg: KnowledgeGraph,
        text: str,
        *,
        limit_per_resolver: int | None = None,
    ) -> list[CandidateHit]:
        seen = set()
        results: list[CandidateHit] = []

        for resolver in self.resolvers:
            if (
                len(results) > 0 
                and self.stop_after_confidence is not None
                and resolver.confidence.value > self.stop_after_confidence.value
            ):
                break

            hits = resolver.resolve(
                kg,
                text,
                limit=limit_per_resolver,
            )
            for hit in hits:
                if hit.concept_id not in seen:
                    seen.add(hit.concept_id)
                    results.append(hit)

        return results
