
from dataclasses import dataclass
from typing import Optional, Iterable
from ..graph.paths import GraphPath, find_shortest_paths
from ..graph.kg import KnowledgeGraph
from ..graph.edges import PredicateKind
from ..graph.scoring import PathProfile, rank_paths, path_profile
from .resolvers import ResolverPipeline

@dataclass(frozen=True)
class GroundingCandidate:
    concept_id: int
    label: str
    best_path_profile: PathProfile
    reasons: tuple[str, ...]
    paths: tuple[GraphPath, ...]


@dataclass(frozen=True)
class GroundingConstraints:
    parent_ids: tuple[int, ...]
    allowed_domains: tuple[str, ...]
    allowed_vocabularies: Optional[tuple[str, ...]] = None
    require_standard: bool = False
    max_depth: int = 6


def _passes_constraints(
    kg: KnowledgeGraph,
    concept_id: int,
    constraints: GroundingConstraints,
) -> tuple[bool, list[str]]:
    reasons = []

    c = kg.concept_view(concept_id)

    # domain constraint
    if constraints.allowed_domains:
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
    if constraints.require_standard and c.standard_concept is None:
        return False, ["concept is non-standard"]

    return True, reasons


def _find_hierarchy_paths(
    kg: KnowledgeGraph,
    concept_id: int,
    parent_ids: tuple[int, ...],
    *,
    max_depth: int,
) -> list[GraphPath]:
    paths = []

    for parent in parent_ids:
        found, trace = find_shortest_paths(
            kg,
            source=concept_id,
            target=parent,
            predicate_kinds={PredicateKind.ONTOLOGICAL},
            max_depth=max_depth,
            max_paths=3,
        )
        paths.extend(found)

    return paths


def _best_profile(
    kg: KnowledgeGraph,
    paths: list[GraphPath],
) -> PathProfile:
    profiles = [path_profile(kg, p) for p in paths]
    return min(profiles)  # uses PathProfile.__lt__

def ground_term(
    kg: KnowledgeGraph,
    text: str,
    *,
    constraints: GroundingConstraints,
    resolver_pipeline: ResolverPipeline,
) -> list[GroundingCandidate]:

    results: list[GroundingCandidate] = []

    for hit in resolver_pipeline.resolve(kg, text):
        ok, reasons = _passes_constraints(kg, hit.concept_id, constraints)
        if not ok:
            continue

        paths = _find_hierarchy_paths(
            kg,
            hit.concept_id,
            constraints.parent_ids,
            max_depth=constraints.max_depth,
        )

        if not paths:
            continue  # fails hierarchy constraint

        best_path_profile = _best_profile(kg, paths)

        c = kg.concept_view(hit.concept_id)

        results.append(
            GroundingCandidate(
                concept_id=hit.concept_id,
                label=c.concept_name,
                best_path_profile=best_path_profile,
                reasons=tuple(reasons),
                paths=tuple(paths),
            )
        )

    results.sort(key=lambda r: r.best_path_profile)
    return results
