from dataclasses import dataclass
from omop_graph.graph.kg import KnowledgeGraph

from dataclasses import dataclass
from typing import Optional, Iterable, Generator
from .resolvers import CandidateResolver, ResolverConfidence, CandidateHit
from ...graph.paths import GraphPath, find_shortest_paths, find_shortest_paths_dijkstra, find_shortest_paths_batch
from ...graph.kg import KnowledgeGraph
from ...graph.edges import PredicateKind
from ...graph.scoring import PathProfile, get_best_path_profile

import logging
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class GroundingCandidate:
    concept_id: int
    is_standard: bool
    label: str
    reasons: tuple[str, ...]
    confidence: ResolverConfidence

    best_path_profile: Optional[PathProfile]
    paths: Optional[tuple[GraphPath, ...]]

    @property
    def score(self) -> float:
        # Higher Score is better

        if self.best_path_profile is not None:
            path_profile_score =  self.best_path_profile.score
        else:
            raise NotImplementedError
            path_profile_score = rank = (self.confidence, not self.is_standard)
        
        assert path_profile_score is not None, "Path profile score should not be None"
        return path_profile_score

@dataclass(frozen=True)
class GroundingConstraints:
    parent_ids: Optional[tuple[int, ...]]
    allowed_domains: Optional[tuple[str, ...]]
    allowed_vocabularies: Optional[tuple[str, ...]] = None
    require_standard: bool = False
    max_depth: int = 6

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

    def resolve(
        self,
        kg: KnowledgeGraph,
        text: str,
        *,
        limit_per_resolver: int | None = None,
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
            )
            for hit in hits:
                if hit.concept_id not in seen:
                    seen.add(hit.concept_id)
                    yield hit
    
    def ground_term(
        self,
        kg: KnowledgeGraph,
        text: str,
        *,
        constraints: GroundingConstraints,
    ) -> list[GroundingCandidate]:

        results: list[GroundingCandidate] = []

        resolved = self.resolve(kg, text)
        for hit in resolved:
            ok, reasons = self._passes_constraints(kg, hit.concept_id, constraints)
            if not ok:
                continue
            
            if constraints.parent_ids is not None:
                paths = self._find_hierarchy_paths(
                    kg,
                    hit.concept_id,
                    constraints.parent_ids,
                    max_depth=constraints.max_depth,
                )
                if not paths:
                    continue  # fails hierarchy constraint
                best_path_profile = get_best_path_profile(path_profiles=[PathProfile.from_path(kg, p, search_term=text, confidence=hit.resolver_confidence) for p in paths])
                paths = tuple(paths)
            else:
                # Placeholder with dummy values?
                best_path_profile = None
                paths = None

            c = kg.concept_view(hit.concept_id)

            results.append(
                GroundingCandidate(
                    concept_id=hit.concept_id,
                    label=c.concept_name,
                    best_path_profile=best_path_profile,
                    reasons=tuple(reasons),
                    paths=paths,
                    confidence=hit.resolver_confidence,
                    is_standard=c.standard_concept,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)  # Sort descending
        return results
    
    @staticmethod
    def _find_hierarchy_paths(
        kg: KnowledgeGraph,
        concept_id: int,
        parent_ids: tuple[int, ...],
        *,
        max_depth: int,
        max_paths: int = 3,
    ) -> list[GraphPath]:
        paths = []

        for parent in parent_ids:
            # found, trace = find_shortest_paths(
            found = find_shortest_paths_batch(
                kg,
                source=concept_id,
                target=parent,
                predicate_kinds={PredicateKind.ONTOLOGICAL},
                max_depth=max_depth,
                max_paths=max_paths,
            )
            paths.extend(found)

        return paths
    
    @staticmethod
    def _passes_constraints(
        kg: KnowledgeGraph,
        concept_id: int,
        constraints: GroundingConstraints,
    ) -> tuple[bool, list[str]]:
        reasons = []

        c = kg.concept_view(concept_id)

        # domain constraint
        if constraints.allowed_domains is not None:
            if c.domain_id not in constraints.allowed_domains:
                return False, [
                    f"domain {c.domain_id} not in {constraints.allowed_domains}"
                ]

        # vocabulary constraint
        if constraints.allowed_vocabularies:
            if c.vocabulary_id not in constraints.allowed_vocabularies:
                return False, [
                    f"vocabulary {c.vocabulary_id} not allowed"
                ]

        # standardness
        if constraints.require_standard and not c.standard_concept:
            return False, ["concept is non-standard"]

        return True, reasons