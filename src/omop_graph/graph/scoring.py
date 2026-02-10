from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, ClassVar
from collections import Counter

from .edges import PredicateKind
from .paths import GraphPath, PathStep
from .traverse import GraphTrace, TraceStep
from .kg import KnowledgeGraph, ConceptView
from .paths import find_shortest_paths

from omop_graph.utils.types import ResolverConfidence

import logging
import numpy as np
logger = logging.getLogger(__name__)

"""
Path scoring and explanation.

Scope: Scoring and explaining paths through the graph.
i.e. Which path is better and why?
"""

@dataclass(frozen=True)
class PathExplanationStep:
    step: PathStep
    traversal_depth: int | None
    predicate_kind: PredicateKind
    reason: str

@dataclass(frozen=True)
class PathExplanation:
    path: GraphPath
    profile: PathProfile
    steps: tuple[PathExplanationStep, ...]

    @classmethod
    def from_path(
        cls,
        kg: KnowledgeGraph,
        path: GraphPath,
        trace: GraphTrace,
        confidence: ResolverConfidence,
    ) -> "PathExplanation":
        steps: list[PathExplanationStep] = []
        profile = PathProfile.from_path(kg, path, confidence=confidence)

        for step in path.steps:
            ts = trace_contains_step(trace, step)
            kind = kg.predicate_kind(step.predicate)
            reason = kind.label()
            steps.append(
                PathExplanationStep(
                    step=step,
                    traversal_depth=ts.depth if ts else None,
                    predicate_kind=kind,
                    reason=reason,
                )
            )
        return cls(
            path=path,
            profile=profile,
            steps=tuple(steps),
        )

        
@dataclass(frozen=True)
class PathScore:
    """
    Immutable scoring record for a graph path traversal.

    Unlike previous iterations which used a multiplicative decay model ($S \\in [0,1]$),
    this class implements an **additive point system**. It starts with a base score 
    derived from the initial text match confidence and adjusts it based on topological 
    moves in the graph.

    The final score represents a trade-off between **Trust** (is this a Standard Concept?) 
    and **Specificity** (how specific is this term compared to the root?).

    Attributes
    ----------
    confidence_score : float
        The starting points awarded based on the initial LLM text match quality. 
        (e.g., Exact match = 10.0, Fuzzy = 6.0).
    depth_bonus : float
        Accumulated points for traversing `Is A` (ONTO_UP) edges. 
        Rewards specificity (deeper concepts are more valuable).
    drift_penalty : float
        Accumulated penalty points for "noisy" transitions. 
        Includes non-ontological mappings, composition edges, or traversing down 
        the hierarchy (generalization).

    Notes
    -----
    **Scoring Formula**
    
    $$ S_{total} = S_{confidence} + S_{depth} - P_{drift} $$

    **Scoring Intuition**
    
    1.  **Base Trust**: We start with a high score if the text match is exact.
    2.  **The "Standard" Gate**: If the path fails to anchor to a Standard OMOP Concept, 
        a massive penalty (`_PENALTY_NON_STANDARD`) is applied, effectively disqualifying 
        the result.
    3.  **Drift vs. Depth**: 
        - We punish lateral moves (mappings) and generalizations (downward steps).
        - We reward specialization (upward steps). 
        - This encourages the algorithm to find the most specific Standard Concept 
          closest to the original search term.
    """
    confidence_score: float
    depth_bonus: float
    drift_penalty: float
    predicate_kind_indices: Optional[dict[PredicateKind, list[int]]] = field(default=None, compare=False)

    _BASE_EXACT: ClassVar[float] = 10.0
    _BASE_SYNONYM: ClassVar[float] = 9.0
    _BASE_FUZZY: ClassVar[float] = 6.0
    _BONUS_PER_UP_STEP: ClassVar[float] = 1.0
    _PENALTY_DOWN_STEP: ClassVar[float] = 2.0 # Not sure if that should be even penalised as we become more specific
    _PENALTY_NON_STANDARD: ClassVar[float] = 10.0  # Heavy penalty
    _PENALTY_EXTRA_MAPPING: ClassVar[float] = 3.0
    _PENALTY_COMPOSITION: ClassVar[float] = 1.0

    @property
    def total_score(self) -> float:
        return self.confidence_score + self.depth_bonus - self.drift_penalty
    
    def __lt__(self, other: "PathScore") -> bool:
        return self.total_score < other.total_score
    
    def __eq__(self, other: "PathScore") -> bool:
        return self.total_score == other.total_score
    
    def __repr__(self) -> str:
        return f"PathScore(total={self.total_score:.2f}, confidence={self.confidence_score:.2f}, depth_bonus={self.depth_bonus:.2f}, drift_penalty={self.drift_penalty:.2f})"

    @classmethod
    def from_predicate_indices(
        cls, 
        confidence: ResolverConfidence,
        predicate_kind_indices: dict[PredicateKind, list[int]],
        standard_anchor_found: bool
    ) -> "PathScore":
        
        # 1. Base Score
        if confidence == ResolverConfidence.EXACT:
            confidence_score = cls._BASE_EXACT
        elif confidence == ResolverConfidence.EXACT_SYNONYM:
            confidence_score = cls._BASE_SYNONYM
        else:
            confidence_score = cls._BASE_FUZZY
                
        depth_bonus = 0.0
        drift_penalty = 0.0

        for predicate_kind, indices in predicate_kind_indices.items():
            if predicate_kind == PredicateKind.MAPPING or predicate_kind == PredicateKind.VERSIONING:
                drift_penalty += cls._PENALTY_EXTRA_MAPPING * len(indices)
            elif predicate_kind == PredicateKind.ONTO_UP:
                depth_bonus += cls._BONUS_PER_UP_STEP * len(indices)
            elif predicate_kind == PredicateKind.ONTO_DOWN:
                drift_penalty += cls._PENALTY_DOWN_STEP * len(indices)
            elif predicate_kind == PredicateKind.COMPOSITION:
                drift_penalty += cls._PENALTY_COMPOSITION * len(indices)

        if not standard_anchor_found:
            drift_penalty += cls._PENALTY_NON_STANDARD
        
        return cls(
            confidence_score=confidence_score,
            depth_bonus=depth_bonus,
            drift_penalty=drift_penalty,
            predicate_kind_indices=predicate_kind_indices,
        )


@dataclass(frozen=True)
class PathProfile:
    """
    Represents the resolved 'Anchor Concept' discovered along a graph path.

    This class fundamentally changes the resolution logic from "scoring a whole path" 
    to "finding a trusted anchor." It traverses the path starting from the LLM's 
    candidate term and stops at the **first Standard OMOP Concept** it encounters.

    - **If a Standard Concept is found**: It becomes the `concept_id` for this profile. 
      The path to reach it is scored for 'drift' (distance/noise).
    - **If NO Standard Concept is found**: The profile reverts to the original 
      candidate ID but applies a heavy 'Non-Standard' penalty to the score.

    Attributes
    ----------
    score : PathScore
        The calculated score object containing the breakdown of points (trust, depth, drift).
    concept_id : int
        The ID of the *resolved* concept. This is either the Standard Concept found 
        mid-path, or the original concept if no standard anchor was found.
    concept_name : str
        The name of the resolved concept.
    is_standard : bool
        True if `concept_id` is a Standard OMOP Concept. If False, this profile 
        will likely have a very low score.
    original_concept_id : int
        The ID of the starting node (the raw candidate from the LLM/Search).
    path : GraphPath
        The full topological path from the original candidate to the root (or end of traversal).

    Methods
    -------
    from_path(...)
        Factory method that traverses the path to identify the 'Standard Anchor'. 
        It promotes the specific `MAPPING` or `VERSIONING` edge that leads to a 
        Standard Concept as the 'Anchor Step' (exempt from penalty), while treating 
        all other edges as scoring modifiers.
    """
    
    path_score: PathScore
    concept_id: int
    concept_name: str
    is_standard: bool
    original_concept_id: int
    original_concept_name: str
    path: GraphPath


    def __lt__(self, other: "PathProfile") -> bool:
        return self.score < other.score
    
    def __eq__(self, other: "PathProfile") -> bool:
        return self.score == other.score
    
    def __repr__(self) -> str:
        return f"PathProfile(score={self.score:.2f}, concept_id={self.concept_id} [{self.concept_name}])"
    
    @property
    def score(self) -> float:
        return self.path_score.total_score

    @classmethod
    def from_path(
        cls, 
        kg: KnowledgeGraph, 
        path: GraphPath, 
        confidence: ResolverConfidence,
        embedding_sims: np.ndarray | None = None
    ) -> "PathProfile":
        
        # Path Traversal
        standard_anchor: Optional[tuple[int, str]] = None
        
        # Pre-fetch views to check standard status
        concept_views = kg.concept_views(path.nodes())
        predicate_kinds = kg.predicate_kinds(tuple(p.predicate for p in path.steps))

        predicate_kind_indices = {}
        for step_idx in range(len(path.steps)):
            predicate_kind = predicate_kinds[step_idx]

            # We promote the first swap to a standard concept as the anchor point
            if (
                (
                    predicate_kind == PredicateKind.MAPPING or 
                    predicate_kind == PredicateKind.VERSIONING
                ) and (  # Leads to standard concept and standard_anchor not been found yet
                    not standard_anchor and concept_views[step_idx + 1].standard_concept
                )
            ):
                standard_anchor = (concept_views[step_idx + 1].concept_id, concept_views[step_idx + 1].concept_name)

            else:
                if predicate_kind not in predicate_kind_indices:
                    predicate_kind_indices[predicate_kind] = []
                predicate_kind_indices[predicate_kind].append(step_idx)

        path_score = PathScore.from_predicate_indices(
            confidence=confidence,
            predicate_kind_indices=predicate_kind_indices,
            standard_anchor_found=standard_anchor is not None
        )
    
        if standard_anchor is None:
            concept_id = concept_views[0].concept_id
            concept_name = concept_views[0].concept_name
            is_standard = concept_views[0].standard_concept
        else:
            concept_id, concept_name = standard_anchor
            is_standard = True

        return cls(
            path_score=path_score,
            concept_id=concept_id,
            concept_name=concept_name,
            is_standard=is_standard,
            original_concept_id=concept_views[0].concept_id,
            original_concept_name=concept_views[0].concept_name,
            path=path,
        )


def rank_path_profiles(path_profiles: list[PathProfile]) -> list[PathProfile]:
    """Ranks path profiles by their path score in descending order."""
    return sorted(path_profiles, key=lambda p: p.score, reverse=True)

def get_best_path_profile(path_profiles: list[PathProfile]) -> PathProfile:
    """Returns the best path profile based on path score."""
    return max(path_profiles, key=lambda p: p.score)

def trace_contains_step(trace: GraphTrace, step: PathStep) -> TraceStep | None:
    for ts in trace.steps:
        if ts.node != step.subject:
            continue
        for e in ts.expanded_edges:
            if (
                e.object_id == step.object
                and e.predicate_id == step.predicate
            ):
                return ts
    return None

def rank_paths(
    kg: KnowledgeGraph,
    paths: list[GraphPath],
    confidence: ResolverConfidence,
) -> list[GraphPath]:
    profiles = {
        path: PathProfile.from_path(kg, path, confidence=confidence)
        for path in paths
    }
    return sorted(paths, key=lambda p: profiles[p].score, reverse=True)

def find_ranked_paths_with_explanations(
    kg: KnowledgeGraph,
    source: int,
    target: int,
    search_term: str,
    confidence: ResolverConfidence,
    *,
    predicate_kinds: set[PredicateKind] | None = None,
    max_depth: int = 6,
    on=None,
    max_paths: int = 20,
):
    paths, trace = find_shortest_paths(
        kg,
        source,
        target,
        predicate_kinds=predicate_kinds,
        max_depth=max_depth,
        on=on,
        max_paths=max_paths,
        traced=True,
    )

    if not paths:
        return []

    ranked = rank_paths(kg, paths, confidence=confidence)

    return [
        PathExplanation.from_path(kg, path, trace) # type: ignore
        for path in ranked
    ]
