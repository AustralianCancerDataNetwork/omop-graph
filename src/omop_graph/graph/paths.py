from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque, defaultdict
from typing import Optional, Any, Tuple, List, Dict, Literal
import heapq

import numpy as np

from .edges import PredicateKind, EdgeView
from .traverse import GraphTrace, TraceStep
from .kg import KnowledgeGraph
from ..reasoning.resolvers import ResolverConfidence, CandidateHit

import logging
logger = logging.getLogger(__name__)


"""
Pathfinding algorithms.

Pure path-finding functions that accept a KnowledgeGraph instance

Scope: Algorithms that find paths between nodes.
i.e. What paths exist between nodes (does not yet score or explain them)
"""

@dataclass(frozen=True)
class Node:
    concept_id: int
    is_standard: bool

@dataclass(frozen=True)
class PathStep:
    subject: Node
    predicate: str
    object: Node

@dataclass(frozen=True)
class GraphPath:
    steps: tuple[PathStep, ...]

    @property
    def start_concept_id(self) -> int:
        if not self.steps:
            raise ValueError("Empty path has no start concept")
        return self.steps[0].subject.concept_id
    
    def get_first_standard_concept_id(self) -> Optional[int]:
        for step in self.steps:
            if step.subject.is_standard:
                return step.subject.concept_id
            if step.object.is_standard:
                return step.object.concept_id
        return None

    def nodes(self):
        if not self.steps:
            return ()
        return (self.steps[0].subject,) + tuple(s.object for s in self.steps)
    
    def __getitem__(self, index):
        return self.steps[index]
    
    def __len__(self):
        return len(self.steps)


    def __repr__(self) -> str:
        if not self.steps:
            return "GraphPath(<empty>)"
        return f"GraphPath(len={len(self.steps)})"

    def explain(self, kg: "KnowledgeGraph") -> str:
        if not self.steps:
            return "source == target"

        parts = []
        for s in self.steps:
            subj = kg.concept_view(s.subject)
            obj = kg.concept_view(s.object)
            pred = kg.predicate(s.predicate)

            parts.append(
                f"{subj.concept_name} "
                f"-[{pred.name}]-> "
                f"{obj.concept_name}"
            )

        return "\n  ↳ ".join(parts)

def reconstruct_paths(source, target, meet, parents_fwd, parents_bwd):
    def left(n):
        if n == source:
            return [()]
        out = []
        for p, pred in parents_fwd[n]:
            for L in left(p):
                out.append(L + (PathStep(p, pred, n),))
        return out

    def right(n):
        if n == target:
            return [()]
        out = []
        for nxt, pred in parents_bwd[n]:
            for R in right(nxt):
                out.append((PathStep(n, pred, nxt),) + R)
        return out

    return [GraphPath(L + R) for L in left(meet) for R in right(meet)]

def find_shortest_paths(
    kg,
    source: int,
    target: int,
    *,
    predicate_kinds: set[PredicateKind] | None = None,
    max_depth: int = 6,
    on=None,
    max_paths: int = 20,
    traced: bool = False,
) -> tuple[list[GraphPath], GraphTrace | None]:
    """
    Find shortest paths using bidirectional BFS.

    If trace=True, returns a GraphTrace containing only the
    nodes and edges actually expanded during the search.
    """
    if source == target:
        path = GraphPath(steps=())
        trace = GraphTrace(seeds=(source,), steps=[], terminated_reason="source_equals_target") if traced else None
        return [path], trace

    q_fwd = deque([source])
    q_bwd = deque([target])

    depth_fwd = {source: 0}
    depth_bwd = {target: 0}

    parents_fwd: dict[int, list[tuple[int, str]]] = defaultdict(list)
    parents_bwd: dict[int, list[tuple[int, str]]] = defaultdict(list)

    best_total_depth: Optional[int] = None
    meeting_nodes: set[int] = set()
    trace_steps: list[TraceStep] = []

    while q_fwd and q_bwd:
        expand_forward = len(q_fwd) <= len(q_bwd)
        expanded: list[EdgeView] = []
        if expand_forward:
            cur = q_fwd.popleft()
            d = depth_fwd[cur]

            if d >= max_depth:
                continue
            
            for e in kg.iter_edges(
                cur,
                direction="out",
                predicate_kinds=predicate_kinds,
                on=on,
            ):
                nxt = e.object_id
                nd = d + 1
                if nd > max_depth:
                    continue

                expanded.append(e)

                if nxt not in depth_fwd:
                    depth_fwd[nxt] = nd
                    q_fwd.append(nxt)

                if depth_fwd[nxt] == nd:
                    parents_fwd[nxt].append((cur, e.predicate_id))

                if nxt in depth_bwd:
                    total = nd + depth_bwd[nxt]
                    if best_total_depth is None or total < best_total_depth:
                        best_total_depth = total
                        meeting_nodes = {nxt}
                    elif total == best_total_depth:
                        meeting_nodes.add(nxt)

        else:
            cur = q_bwd.popleft()
            d = depth_bwd[cur]

            if d >= max_depth:
                continue

            for e in kg.iter_edges(
                cur,
                direction="in",
                predicate_kinds=predicate_kinds,
                on=on,
            ):
                expanded.append(e)
                prev = e.subject_id
                nd = d + 1
                if nd > max_depth:
                    continue

                if prev not in depth_bwd:
                    depth_bwd[prev] = nd
                    q_bwd.append(prev)

                if depth_bwd[prev] == nd:
                    parents_bwd[prev].append((cur, e.predicate_id))

                if prev in depth_fwd:
                    total = depth_fwd[prev] + nd
                    if best_total_depth is None or total < best_total_depth:
                        best_total_depth = total
                        meeting_nodes = {prev}
                    elif total == best_total_depth:
                        meeting_nodes.add(prev)

        if traced:
            trace_steps.append(TraceStep(depth=d, node=cur, expanded_edges=tuple(expanded)))

        # no shorter path possible
        if best_total_depth is not None:
            min_fwd = min(
                (depth_fwd[n] for n in q_fwd),
                default=depth_fwd[source],
            )
            min_bwd = min(
                (depth_bwd[n] for n in q_bwd),
                default=depth_bwd[target],
            )
            if min_fwd + min_bwd >= best_total_depth:
                break

    if not meeting_nodes:
        return [], (
            GraphTrace(
                seeds=(source,),
                steps=trace_steps,
                terminated_reason="no_path",
            ) if traced else None
        )

    paths: list[GraphPath] = []
    for meet in meeting_nodes:
        paths.extend(
            reconstruct_paths(
                source, target, meet, parents_fwd, parents_bwd
            )
        )
        if len(paths) >= max_paths:
            break

    graph_trace = (
        GraphTrace(
            seeds=(source,),
            steps=trace_steps,
            terminated_reason="shortest_paths_found",
        )
        if traced else None
    )

    return paths[:max_paths], graph_trace

def find_shortest_paths_batch(
    kg,
    source: int,
    target: int,
    *,
    predicate_kinds: set[PredicateKind] | frozenset[PredicateKind] | None = None,
    max_depth: int = 6,
    on=None,
    max_paths: int = 20,
) -> list[GraphPath]:
    
    if source == target:
        return [GraphPath(steps=())]

    # Frontiers: The set of nodes we are currently expanding
    fwd_frontier = {source}
    bwd_frontier = {target}

    # Visited/Depth tracking
    depth_fwd = {source: 0}
    depth_bwd = {target: 0}

    # Parents for path reconstruction
    parents_fwd: dict[int, list[tuple[int, str]]] = defaultdict(list)
    parents_bwd: dict[int, list[tuple[int, str]]] = defaultdict(list)

    best_total_depth: Optional[int] = None
    meeting_nodes: set[int] = set()

    # Loop until frontiers are empty
    while fwd_frontier and bwd_frontier:
        
        # 1. Expand the smaller frontier (Optimization: Balanced Bi-BFS)
        expand_forward = len(fwd_frontier) <= len(bwd_frontier)
        
        # Setup variables based on direction
        if expand_forward:
            current_layer_nodes = tuple(fwd_frontier)
            other_frontier = bwd_frontier
            direction = "out"
            current_depth_map = depth_fwd
            other_depth_map = depth_bwd
            current_parents = parents_fwd
        else:
            current_layer_nodes = tuple(bwd_frontier)
            other_frontier = fwd_frontier
            direction = "in"
            current_depth_map = depth_bwd
            other_depth_map = depth_fwd
            current_parents = parents_bwd
        
        # 2. Batch Query: Get all edges for the current layer in ONE shot
        batch_edges = kg.iter_edges_batch(
            current_layer_nodes,
            direction=direction,
            predicate_kinds=frozenset(predicate_kinds) if predicate_kinds else None,
            on=on
        )

        next_frontier = set()
        
        # 3. Process edges in memory
        for e in batch_edges:
            # Identify Start (u) and End (v) relative to traversal direction
            u = e.subject_id if expand_forward else e.object_id
            v = e.object_id if expand_forward else e.subject_id
            
            d = current_depth_map[u]
            nd = d + 1

            if nd > max_depth:
                continue

            # Update visited/parents
            if v not in current_depth_map:
                current_depth_map[v] = nd
                next_frontier.add(v)
                current_parents[v].append((u, e.predicate_id))
            elif current_depth_map[v] == nd:
                # Found another path to the same node at the same optimal depth
                current_parents[v].append((u, e.predicate_id))

            # Check for collision (Did we meet the other side?)
            if v in other_depth_map:
                total = nd + other_depth_map[v]
                if best_total_depth is None or total < best_total_depth:
                    best_total_depth = total
                    meeting_nodes = {v}
                elif total == best_total_depth:
                    meeting_nodes.add(v)

        # 4. Stop Condition check
        if best_total_depth is not None:
            # Shallowest possible node in the NEXT layer we just built
            min_current = min((current_depth_map[n] for n in next_frontier), default=999)
            
            # Shallowest possible node waiting in the OTHER frontier
            min_other = min((other_depth_map[n] for n in other_frontier), default=999)
            
            # If the best potential new path is already worse than what we found, stop.
            if min_current + min_other >= best_total_depth:
                break

        # Move to next layer
        if expand_forward:
            fwd_frontier = next_frontier
        else:
            bwd_frontier = next_frontier

    # Reconstruct paths
    if not meeting_nodes:
        return []

    paths: list[GraphPath] = []
    for meet in meeting_nodes:
        paths.extend(
            reconstruct_paths(source, target, meet, parents_fwd, parents_bwd)
        )
        if len(paths) >= max_paths:
            break
            
    return paths[:max_paths]


@dataclass(order=True)
class QueueItem:
    cost: float
    node: Node = field(compare=False)
    rc: ResolverConfidence = field(compare=False)
    iterations: int = field(default=0, compare=False) 

@dataclass(frozen=True)
class StandardConcept:
    concept_id: int
    concept_name: str
    separation: int
    original_id: int
    original_name: str
    matched_label: str
    resolver_confidence: ResolverConfidence
    hierarchy_cost: float = 0.0


def get_unique_standard_concepts(concepts: list[StandardConcept]) -> list[StandardConcept]:
    sorted_concepts = sorted(
        concepts,
        key=lambda x: (
            x.separation,           # Lower is better (Ascending)
            x.resolver_confidence.value, # Lower is better (this is the ranking in the enum)
            x.hierarchy_cost        # Lower is better (Ascending)
        )
    )

    # 2. Use a dictionary to pick the first one seen for each ID
    unique_best_concepts = {}
    for concept in sorted_concepts:
        if concept.concept_id not in unique_best_concepts:
            unique_best_concepts[concept.concept_id] = concept

    result = list(unique_best_concepts.values())
    return result


def find_standard_paths(
    kg: KnowledgeGraph,
    target: int,
    candidate: CandidateHit,
    predicate_kinds: Optional[frozenset[Any]] = None,
    max_depth: int = 6,
    max_concepts: Optional[int] = None,
    num_hops: int = 1,
    *args,
    **kwargs
) -> List[StandardConcept]:
    """Alternative search for standard concepts w.r.t target concepts. The idea
    is to translate as past as possible to a standard-concept and then use the
    `concept_ancestor` table to create paths and rank them accordingly.
    This way, we don't need to traverse EVERY path possible but instead only go
    to paths that are worthwhile exploring while accelerating the retrieval.
    
    """
    # --- COST CONFIGURATION ---
    #COST_NS_TO_STD = 1
    #COST_STD_TO_STD = 0 # No cost traversing between standard concepts
    #COST_NS_TO_NS = 5
    #COST_PREDICATES = {
    #    PredicateKind.ONTO_UP: 2,
    #    PredicateKind.ONTO_DOWN: 2,
    #    PredicateKind.MAPS_TO: 1,
    #}
    # -------------------------------

    source_view = kg.concept_view(candidate.concept_id)
    source_is_std = source_view.standard_concept if source_view else False

    # Initialise the queue
    queue = [QueueItem(
        cost=0, 
        node=Node(candidate.concept_id, source_is_std), 
        rc=candidate.resolver_confidence, 
        iterations=0
    )]
    visited: Dict[Tuple[int, bool], int] = {}

    found_standard_concepts = []

    max_concepts_provided = max_concepts is not None
    
    while queue:
        item = heapq.heappop(queue)  # type: ignore
        subject_node = item.node
        cost = item.cost
        rc = item.rc
        iterations = item.iterations

        if max_concepts_provided and len(found_standard_concepts) > max_concepts:
            break
        
        # Prevent infinite loops
        if iterations > num_hops:
            continue

        if subject_node.is_standard:
            # We found a standard concept
            potential_ancestor = kg.get_potential_ancestor(child_id=subject_node.concept_id, parent_id=target)
            if potential_ancestor is not None:
                if potential_ancestor.min_levels_of_separation > max_depth:
                    continue

                found_standard_concepts.append(StandardConcept(
                    hierarchy_cost=cost,
                    concept_id=subject_node.concept_id,
                    concept_name=kg.concept_view(subject_node.concept_id).concept_name,
                    separation=potential_ancestor.min_levels_of_separation,
                    original_id=candidate.concept_id,
                    original_name=source_view.concept_name,
                    matched_label=candidate.matched_label,
                    resolver_confidence=rc,
                ))
                continue

        # We have not found anything so we need to get to the next best concept_id as fast as it goes.
        edges = list(kg.iter_edges_batch(
            (subject_node.concept_id,), 
            direction="out", 
            predicate_kinds=predicate_kinds
        ))
        if not edges:
            continue
        
        # Singular trip to the DB
        object_views = kg.concept_views(tuple(e.object_id for e in edges))

        for edge, object_view in zip(edges, object_views):
            object_id = edge.object_id 
            predicate_id = edge.predicate_id
            converted_predicate_kind = kg.predicate_kind(predicate_id)

            object_is_std = object_view.standard_concept
            if not object_is_std:
                continue # This is for now. Only allow one hope as most concepts have a direct Non-Std to Std hop
            
            # Not punishing doing the mapping hop from NS to STD as that is the whole point of this function
            new_cost = cost
            #new_cost = cost + COST_PREDICATES[converted_predicate_kind]  # Could fail but I am willing to take that risk for now

            heapq.heappush(queue,  # type: ignore
                QueueItem( # type: ignore
                    cost=new_cost,
                    node=Node(concept_id=object_id, is_standard=object_is_std),
                    rc=ResolverConfidence.PARTIAL,  # We mapped so we gotta reduce the confidence
                    iterations=iterations + 1
                )
            )

    return found_standard_concepts

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
    concept_id: int
    concept_name: str
    is_standard: bool
    original_concept_id: int
    original_concept_name: str
    path: GraphPath
    
    def __repr__(self) -> str:
        return f"PathProfile(concept_id={self.concept_id} [{self.concept_name}])"
    

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
                    predicate_kind == PredicateKind.MAPS_TO or 
                    predicate_kind == PredicateKind.VERSIONING or 
                    predicate_kind == PredicateKind.MAPS_FROM
                ) and (  # Leads to standard concept and standard_anchor not been found yet
                    not standard_anchor and concept_views[step_idx + 1].standard_concept
                )
            ):
                standard_anchor = (concept_views[step_idx + 1].concept_id, concept_views[step_idx + 1].concept_name)

            else:
                if predicate_kind not in predicate_kind_indices:
                    predicate_kind_indices[predicate_kind] = []
                predicate_kind_indices[predicate_kind].append(step_idx)
    
        if standard_anchor is None:
            concept_id = concept_views[0].concept_id
            concept_name = concept_views[0].concept_name
            is_standard = concept_views[0].standard_concept
        else:
            concept_id, concept_name = standard_anchor
            is_standard = True

        return cls(
            concept_id=concept_id,
            concept_name=concept_name,
            is_standard=is_standard,
            original_concept_id=concept_views[0].concept_id,
            original_concept_name=concept_views[0].concept_name,
            path=path,
        )


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