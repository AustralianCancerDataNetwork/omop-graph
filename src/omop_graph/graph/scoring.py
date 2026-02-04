from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, ClassVar

from .edges import PredicateKind
from .paths import GraphPath, PathStep
from .traverse import GraphTrace, TraceStep
from .kg import KnowledgeGraph
from .paths import find_shortest_paths

from omop_graph.utils.types import ResolverConfidence

import rapidfuzz

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
        search_term: str,
        confidence: ResolverConfidence,
    ) -> "PathExplanation":
        steps: list[PathExplanationStep] = []
        profile = PathProfile.from_path(kg, path, search_term=search_term, confidence=confidence)

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
class PathProfile:
    r"""
    Profile of a graph path connecting a candidate concept to a restricting ancestor.

    This class encapsulates the topological and semantic features of a path in the 
    Knowledge Graph (KG) and computes a scalar quality score. This score is used 
    to rank competing grounding candidates, prioritizing paths that are semantically 
    "pure" (standard concepts, consistent vocabulary, ontological edges) over 
    paths that rely on loose mappings or non-standard terms.

    Parameters
    ----------
    hops : int
        The total number of edges in the path.
    invalid_concepts : int
        Count of nodes in the path that are flagged as invalid/deprecated.
    non_standard_concepts : int
        Count of nodes that are not Standard OMOP concepts (e.g., source codes).
    vocab_switches : int
        Count of transitions between different vocabularies (e.g., SNOMED -> LOINC).
    ontological_edges : int
        Count of strict 'Is A' or 'Subsumes' relationships.
    mapping_edges : int
        Count of 'Maps to' relationships.
    metadata_edges : int
        Count of loose or metadata-based relationships (e.g., 'Has domain').
    confidence : ResolverConfidence
        The initial confidence level of the text match (e.g., EXACT, EXACT_SYNONYM).
    search_term : str
        The raw text extracted by the LLM.
    matched_concept_name : str
        The official name of the concept found in the KG.

    Notes
    -----
    **Scoring Algorithm**

    The `score` property calculates a quality metric ($S \in [0.0, 1.0]$) using a 
    multiplicative penalty model. The scoring logic follows this intuition:

    1.  **Base Score**:
        - If `confidence` is EXACT or EXACT_SYNONYM, $S_{base} = 1.0$.
        - Otherwise, the lexical similarity (Jaro-Winkler) between 
          the `search_term` and `matched_concept_name`.

    2.  **Hard Constraints**:
        - If `invalid_concepts > 0`, the score is strictly $0.0$.

    3.  **Structural Penalties (Multipliers)**:
        The base score is degraded by graph imperfections. We use multipliers rather 
        than subtraction to ensure that a perfect text match on a "bad" concept 
        can still be outranked by a decent match on a "good" concept, but never 
        drops below zero.

        - **Non-Standard Concepts**: We strongly prefer Standard OMOP concepts. A Standard concept with a 60% text match ($0.6$) 
          will beat a Non-Standard concept with a 100% text match (using penalty_non_standard=0.5). Strong penalty.
        
        - **Vocabulary Switches**: Jumping between vocabularies (e.g., SNOMED to ICD10) introduces semantic drift risk. Mild penalty.
        
        - **Mapping Edges**: 'Maps to' is generally trusted but implies a translation rather than a direct ontological parent. Very slight penalty.
         
        - **Metadata Edges**: These edges are often associative rather than hierarchical. Moderate penalty.

    4.  **Path Length Decay**: We prefer the shortest *valid* path. The decay is very slow to ensure that specific (deep) concepts are not 
        unfairly penalised against generic (shallow) ones.
    """
    hops: int
    invalid_concepts: int
    non_standard_concepts: int
    vocab_switches: int
    ontological_edges: int
    mapping_edges: int
    metadata_edges: int
    confidence: "ResolverConfidence"
    search_term: str
    matched_concept_name: str
    _score: Optional[float] = field(default=None)

    # Penalty class vars
    penalty_non_standard: ClassVar[float] = 0.5
    penalty_vocab_switch: ClassVar[float] = 0.9
    penalty_mapping_switch: ClassVar[float] = 0.95
    penalty_metadata_edges: ClassVar[float] = 0.8
    penalty_hops: ClassVar[float] = 0.98

    def __post_init__(self):
        # Calculate and cache the score immediately.
        # Uses object.__setattr__ to bypass frozen=True constraint for caching.
        object.__setattr__(self, "_score", self.score) 

    @property     
    def score(self) -> float:
        if self._score is not None:
            return self._score
            
        # 1. Hard Rejects
        if self.invalid_concepts > 0:
            return 0.0
            
        # 2. Base Score Calculation
        if self.confidence == ResolverConfidence.EXACT:
            score = 1.0
        elif self.confidence == ResolverConfidence.EXACT_SYNONYM:
            score = 1.0  # NOTE: could also be something less
        else:
            score = lexical_similarity(self.matched_concept_name, self.search_term)

        # 3. Structural Penalties
        if self.non_standard_concepts > 0:
            score *= self.penalty_non_standard ** self.non_standard_concepts

        if self.vocab_switches > 0:
            score *= self.penalty_vocab_switch ** self.vocab_switches

        if self.mapping_edges > 0:
            score *= self.penalty_mapping_switch ** self.mapping_edges
            
        if self.metadata_edges > 0:
            score *= self.penalty_metadata_edges ** self.metadata_edges

        # 4. Path Length Decay
        if self.hops > 0:
            score *= self.penalty_hops ** self.hops

        return score

    def __lt__(self, other: "PathProfile") -> bool:
        return self.score < other.score
    
    def __gt__(self, other: "PathProfile") -> bool:
        return self.score > other.score
    
    def __eq__(self, other: "PathProfile") -> bool:
        return self.score == other.score
    
    @classmethod
    def from_path(cls, kg: KnowledgeGraph, path: GraphPath, search_term: str, confidence: ResolverConfidence) -> "PathProfile":
        invalid = 0
        non_standard = 0
        vocab_switches = 0

        prev_vocab = None
        for cid in path.nodes():
            c = kg.concept_view(cid)
            if c.invalid_reason:
                invalid += 1
            if c.standard_concept is None:
                non_standard += 1
            if prev_vocab and c.vocabulary_id != prev_vocab:
                vocab_switches += 1
            prev_vocab = c.vocabulary_id

        ont = map_ = meta = 0
        for step in path.steps:
            kind = kg.predicate_kind(step.predicate)
            if kind == PredicateKind.ONTOLOGICAL:
                ont += 1
            elif kind == PredicateKind.MAPPING:
                map_ += 1
            else:
                meta += 1

        # The starting point of the graph corresponds to the concept ID predicted by the LLM
        c = kg.concept_view(path[0].subject)  
        
        return cls(
            hops=len(path.steps),
            invalid_concepts=invalid,
            non_standard_concepts=non_standard,
            vocab_switches=vocab_switches,
            ontological_edges=ont,
            mapping_edges=map_,
            metadata_edges=meta,
            confidence=confidence,
            search_term=search_term,
            matched_concept_name=c.concept_name
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
    search_term: str,
    confidence: ResolverConfidence,
) -> list[GraphPath]:
    profiles = {
        path: PathProfile.from_path(kg, path, search_term=search_term, confidence=confidence)
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

    ranked = rank_paths(kg, paths, search_term=search_term, confidence=confidence)

    return [
        PathExplanation.from_path(kg, path, trace) # type: ignore
        for path in ranked
    ]


def lexical_similarity(a: str, b: str) -> float:
    """
    Simple lexical similarity metric between two strings.

    Returns a value between 0 and 1, where 1 means identical strings.
    """
    return rapidfuzz.distance.JaroWinkler.similarity(a.lower(), b.lower())