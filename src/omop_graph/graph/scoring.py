from __future__ import annotations
from dataclasses import dataclass

from .edges import PredicateKind
from .paths import GraphPath, PathStep
from .traverse import GraphTrace, TraceStep
from .kg import KnowledgeGraph
from .paths import find_shortest_paths

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

@dataclass(frozen=True)
class PathProfile:
    hops: int
    invalid_concepts: int
    non_standard_concepts: int
    vocab_switches: int

    ontological_edges: int
    mapping_edges: int
    metadata_edges: int

    def path_rank(self) -> tuple:
        """
        Lower is better.
        """
        return (
            self.invalid_concepts,        # never want invalid concepts
            self.non_standard_concepts,   # prefer standard
            self.metadata_edges,          # noisy
            self.mapping_edges,           # allowed but less ideal
            self.vocab_switches,           # continuity matters
            self.hops,                    # finally: shortest
            -self.ontological_edges,      # prefer structure if tied
        )

    def __lt__(self, other: PathProfile) -> bool:
        return self.path_rank() < other.path_rank()

def path_profile(kg: KnowledgeGraph, path: GraphPath) -> PathProfile:
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

    return PathProfile(
        hops=len(path.steps),
        invalid_concepts=invalid,
        non_standard_concepts=non_standard,
        vocab_switches=vocab_switches,
        ontological_edges=ont,
        mapping_edges=map_,
        metadata_edges=meta,
    )

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

def explain_path(
    kg: KnowledgeGraph,
    path: GraphPath,
    trace: GraphTrace,
) -> PathExplanation:
    steps: list[PathExplanationStep] = []
    profile = path_profile(kg, path)

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
    return PathExplanation(
        path=path,
        profile=profile,
        steps=tuple(steps),
    )

def rank_paths(
    kg: KnowledgeGraph,
    paths: list[GraphPath],
) -> list[GraphPath]:
    profiles = {
        path: path_profile(kg, path)
        for path in paths
    }
    return sorted(paths, key=lambda p: profiles[p].path_rank())

def find_ranked_paths_with_explanations(
    kg,
    source: int,
    target: int,
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

    ranked = rank_paths(kg, paths)

    return [
        explain_path(kg, path, trace) # type: ignore
        for path in ranked
    ]
