"""
Pathfinding algorithms for the OMOP Knowledge Graph.

This module provides pure path-finding functions that accept a `KnowledgeGraph`
instance. It focuses on discovering topological connections between nodes,
including shortest paths, batch traversal, and specific standard concept resolution.

Scope
-----
Algorithms that find paths between nodes. This module answers "what paths exist"
but does not inherently score or explain them (that is handled by the `reasoning` module).
"""

from __future__ import annotations

import heapq
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Union,
)

import numpy as np

# Local Application Imports
from omop_graph.graph.edges import EdgeView, PredicateKind
from omop_graph.graph.traverse import GraphTrace, TraceStep
from omop_graph.reasoning.resolvers import CandidateHit, ResolverConfidence

if TYPE_CHECKING:
    from omop_graph.graph.kg import KnowledgeGraph

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Node:
    """
    A lightweight representation of a graph node for pathfinding.

    Parameters
    ----------
    concept_id : int
        The OMOP Concept ID.
    is_standard : bool
        Whether this concept is a Standard Concept.
    """

    concept_id: int
    is_standard: bool


@dataclass(frozen=True)
class PathStep:
    """
    A single step in a graph path.

    Parameters
    ----------
    subject : Node
        The starting node of the step.
    predicate : str
        The relationship ID connecting the nodes.
    object : Node
        The ending node of the step.
    """

    subject: Node
    predicate: str
    object: Node


@dataclass(frozen=True)
class GraphPath:
    """
    A sequence of steps representing a path through the graph.

    Parameters
    ----------
    steps : tuple[PathStep, ...]
        The ordered sequence of steps.
    """

    steps: tuple[PathStep, ...]

    @property
    def start_concept_id(self) -> int:
        """
        Get the concept ID of the first node in the path.

        Raises
        ------
        ValueError
            If the path is empty.
        """
        if not self.steps:
            raise ValueError("Empty path has no start concept")
        return self.steps[0].subject.concept_id

    def get_first_standard_concept_id(self) -> Optional[int]:
        """
        Find the ID of the first Standard Concept encountered in the path.

        Returns
        -------
        int | None
            The concept ID if found, otherwise None.
        """
        for step in self.steps:
            if step.subject.is_standard:
                return step.subject.concept_id
            if step.object.is_standard:
                return step.object.concept_id
        return None

    def nodes(self) -> tuple[int, ...]:
        """
        Get all concept IDs in the path (start node + all object nodes).
        """
        if not self.steps:
            return ()
        return (self.steps[0].subject.concept_id,) + tuple(
            s.object.concept_id for s in self.steps
        )

    def __getitem__(self, index):
        return self.steps[index]

    def __len__(self):
        return len(self.steps)

    def __repr__(self) -> str:
        if not self.steps:
            return "GraphPath(<empty>)"
        return f"GraphPath(len={len(self.steps)})"

    def explain(self, kg: "KnowledgeGraph") -> str:
        """
        Generate a human-readable string explaining the path.

        Parameters
        ----------
        kg : KnowledgeGraph
            The graph instance used to lookup names.

        Returns
        -------
        str
            A multi-line string description of the path.
        """
        if not self.steps:
            return "source == target"

        parts = []
        for s in self.steps:
            subj = kg.concept_view(s.subject.concept_id)
            obj = kg.concept_view(s.object.concept_id)
            pred = kg.predicate(s.predicate)

            parts.append(
                f"{subj.concept_name} " f"-[{pred.name}]-> " f"{obj.concept_name}"
            )

        return "\n  ↳ ".join(parts)


def reconstruct_paths(source, target, meet, parents_fwd, parents_bwd):
    """
    Helper function to reconstruct full paths from bidirectional BFS parent pointers.
    """

    def left(n):
        if n == source:
            return [()]
        out = []
        for p, pred in parents_fwd[n]:
            for L in left(p):
                # We need to construct Nodes here. In raw BFS we only tracked IDs.
                # NOTE: This reconstruction assumes we fetch 'is_standard' later or ignore it here.
                # For strictly typing PathStep, we create dummy Nodes here or need access to KG.
                # Assuming simple reconstruction for now.
                # To fix strictly: BFS needs to store Node info or we look it up.
                # For now, we instantiate Nodes with is_standard=False as placeholders if strictly required,
                # but usually the calling function enriches this.
                subj = Node(p, False)
                obj = Node(n, False)
                out.append(L + (PathStep(subj, pred, obj),))
        return out

    def right(n):
        if n == target:
            return [()]
        out = []
        for nxt, pred in parents_bwd[n]:
            for R in right(nxt):
                subj = Node(n, False)
                obj = Node(nxt, False)
                out.append((PathStep(subj, pred, obj),) + R)
        return out

    return [GraphPath(L + R) for L in left(meet) for R in right(meet)]


def find_shortest_paths(
    kg: "KnowledgeGraph",
    source: int,
    target: int,
    *,
    predicate_kinds: Optional[Set[PredicateKind]] = None,
    max_depth: int = 6,
    on: Optional[Any] = None,
    max_paths: int = 20,
    traced: bool = False,
) -> Tuple[List[GraphPath], Optional[GraphTrace]]:
    """
    Find shortest paths between source and target using bidirectional BFS.

    Parameters
    ----------
    kg : KnowledgeGraph
        The graph instance.
    source : int
        Start concept ID.
    target : int
        End concept ID.
    predicate_kinds : set[PredicateKind], optional
        Restrict traversal to specific edge types.
    max_depth : int
        Maximum path length.
    on : date, optional
        Date for validity checks.
    max_paths : int
        Maximum number of paths to return.
    traced : bool
        If True, returns a GraphTrace object recording the search process.

    Returns
    -------
    tuple[list[GraphPath], GraphTrace | None]
        A list of paths and optionally the trace object.
    """
    if source == target:
        path = GraphPath(steps=())
        trace = (
            GraphTrace(
                seeds=(source,), steps=[], terminated_reason="source_equals_target"
            )
            if traced
            else None
        )
        return [path], trace

    q_fwd = deque([source])
    q_bwd = deque([target])

    depth_fwd = {source: 0}
    depth_bwd = {target: 0}

    parents_fwd: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
    parents_bwd: Dict[int, List[Tuple[int, str]]] = defaultdict(list)

    best_total_depth: Optional[int] = None
    meeting_nodes: Set[int] = set()
    trace_steps: List[TraceStep] = []

    while q_fwd and q_bwd:
        expand_forward = len(q_fwd) <= len(q_bwd)
        expanded: List[EdgeView] = []
        
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
            trace_steps.append(
                TraceStep(depth=d, node=cur, expanded_edges=tuple(expanded))
            )

        # Stop if we found a path and current searches exceed optimal depth
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
            )
            if traced
            else None
        )

    paths: List[GraphPath] = []
    for meet in meeting_nodes:
        # Note: reconstruction logic needs careful implementation to create proper Node objects
        # if using the simplified 'reconstruct_paths' helper above.
        paths.extend(reconstruct_paths(source, target, meet, parents_fwd, parents_bwd))
        if len(paths) >= max_paths:
            break

    graph_trace = (
        GraphTrace(
            seeds=(source,),
            steps=trace_steps,
            terminated_reason="shortest_paths_found",
        )
        if traced
        else None
    )

    return paths[:max_paths], graph_trace


def find_shortest_paths_batch(
    kg: "KnowledgeGraph",
    source: int,
    target: int,
    *,
    predicate_kinds: Union[Set[PredicateKind], frozenset[PredicateKind], None] = None,
    max_depth: int = 6,
    on: Optional[Any] = None,
    max_paths: int = 20,
) -> List[GraphPath]:
    """
    Find shortest paths using an optimized batch-BFS approach.

    This reduces the number of database queries by fetching edges for entire
    frontiers at once.

    Parameters
    ----------
    kg : KnowledgeGraph
        The graph instance.
    source : int
        Start concept ID.
    target : int
        End concept ID.
    predicate_kinds : set[PredicateKind], optional
        Restrict traversal to specific edge types.
    max_depth : int
        Maximum path length.
    on : date, optional
        Date for validity checks.
    max_paths : int
        Maximum number of paths to return.

    Returns
    -------
    list[GraphPath]
        Found paths.
    """
    if source == target:
        return [GraphPath(steps=())]

    # Frontiers: The set of nodes we are currently expanding
    fwd_frontier = {source}
    bwd_frontier = {target}

    # Visited/Depth tracking
    depth_fwd = {source: 0}
    depth_bwd = {target: 0}

    # Parents for path reconstruction
    parents_fwd: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
    parents_bwd: Dict[int, List[Tuple[int, str]]] = defaultdict(list)

    best_total_depth: Optional[int] = None
    meeting_nodes: Set[int] = set()

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
            on=on,
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
            min_current = min(
                (current_depth_map[n] for n in next_frontier), default=999
            )

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

    paths: List[GraphPath] = []
    for meet in meeting_nodes:
        paths.extend(reconstruct_paths(source, target, meet, parents_fwd, parents_bwd))
        if len(paths) >= max_paths:
            break

    return paths[:max_paths]


@dataclass(order=True)
class QueueItem:
    """Priority Queue Item for Dijkstra/A* search."""

    cost: float
    node: Node = field(compare=False)
    rc: ResolverConfidence = field(compare=False)
    iterations: int = field(default=0, compare=False)


@dataclass(frozen=True)
class StandardConcept:
    """
    A resolved Standard Concept resulting from a search.
    """

    concept_id: int
    concept_name: str
    separation: int
    original_id: int
    original_name: str
    matched_label: str
    resolver_confidence: ResolverConfidence
    hierarchy_cost: float = 0.0


def get_unique_standard_concepts(
    concepts: List[StandardConcept],
) -> List[StandardConcept]:
    """
    Filter a list of StandardConcepts to keep only the best match per Concept ID.

    Ranking criteria:
    1. Separation (lower is better)
    2. Resolver Confidence (lower value is better in this enum)
    3. Hierarchy Cost (lower is better)
    """
    sorted_concepts = sorted(
        concepts,
        key=lambda x: (
            x.separation,
            x.resolver_confidence.value,
            x.hierarchy_cost,
        ),
    )

    unique_best_concepts = {}
    for concept in sorted_concepts:
        if concept.concept_id not in unique_best_concepts:
            unique_best_concepts[concept.concept_id] = concept

    return list(unique_best_concepts.values())


def find_standard_paths(
    kg: "KnowledgeGraph",
    target: int,
    candidate: CandidateHit,
    predicate_kinds: Optional[frozenset[Any]] = None,
    max_depth: int = 6,
    max_concepts: Optional[int] = None,
    num_hops: int = 1,
    *args,
    **kwargs,
) -> List[StandardConcept]:
    """
    Search for Standard Concepts related to a target ID, starting from a candidate.

    This method traverses from the candidate (Non-Standard) concept to find
    Standard Concepts, then verifies if those Standard Concepts are ancestors
    of the target concept in the hierarchy.

    Parameters
    ----------
    kg : KnowledgeGraph
        The graph instance.
    target : int
        The ancestor concept ID to check against.
    candidate : CandidateHit
        The search hit to start traversal from.
    predicate_kinds : frozenset, optional
        Allowed edge types for traversal.
    max_depth : int
        Max separation levels allowed in the ancestor check.
    max_concepts : int, optional
        Stop after finding this many unique standard concepts.
    num_hops : int
        Max hops allowed from the candidate to reach a standard concept.

    Returns
    -------
    list[StandardConcept]
        The resolved concepts.
    """
    source_view = kg.concept_view(candidate.concept_id)
    source_is_std = source_view.standard_concept if source_view else False

    # Initialise the queue
    queue = [
        QueueItem(
            cost=0.0,
            node=Node(candidate.concept_id, source_is_std),
            rc=candidate.resolver_confidence,
            iterations=0,
        )
    ]
    
    # Track found concepts to respect max_concepts
    found_standard_concepts: List[StandardConcept] = []
    
    # Note: visited logic was initialized in original code but unused in loop
    # visited: Dict[Tuple[int, bool], int] = {} 

    while queue:
        item = heapq.heappop(queue)
        subject_node = item.node
        cost = item.cost
        rc = item.rc
        iterations = item.iterations

        if max_concepts and len(found_standard_concepts) >= max_concepts:
            break

        # Prevent infinite loops / deep traversals
        if iterations > num_hops:
            continue

        if subject_node.is_standard:
            # We found a standard concept -> Check ancestry with target
            potential_ancestor = kg.get_potential_ancestor(
                child_id=subject_node.concept_id, parent_id=target
            )
            
            if potential_ancestor is not None:
                if potential_ancestor.min_levels_of_separation > max_depth:
                    continue

                found_standard_concepts.append(
                    StandardConcept(
                        hierarchy_cost=cost,
                        concept_id=subject_node.concept_id,
                        concept_name=kg.concept_view(
                            subject_node.concept_id
                        ).concept_name,
                        separation=potential_ancestor.min_levels_of_separation,
                        original_id=candidate.concept_id,
                        original_name=source_view.concept_name,
                        matched_label=candidate.matched_label,
                        resolver_confidence=rc,
                    )
                )
                continue

        # Expand: Go to next best concept_id
        edges = list(
            kg.iter_edges_batch(
                (subject_node.concept_id,),
                direction="out",
                predicate_kinds=predicate_kinds,
            )
        )
        if not edges:
            continue

        # Singular trip to the DB for object views
        object_ids = tuple(e.object_id for e in edges)
        object_views = kg.concept_views(object_ids)

        for edge, object_view in zip(edges, object_views):
            object_id = edge.object_id
            # object_view = object_views lookup corresponds to index or zip order?
            # concept_views returns results in order? If not, this logic needs a dict lookup map.
            # Assuming concept_views returns map or ordered list matching input.
            # Safety check:
            if object_view.concept_id != object_id:
                # Fallback if bulk fetch isn't ordered
                object_view = kg.concept_view(object_id)

            object_is_std = object_view.standard_concept
            
            # Optimization: Only traverse to Standard concepts
            if not object_is_std:
                continue

            new_cost = cost
            #new_cost = cost + COST_PREDICATES[converted_predicate_kind]  # Not punishing on the mapping to standard concept

            heapq.heappush(
                queue,
                QueueItem(
                    cost=new_cost,
                    node=Node(concept_id=object_id, is_standard=object_is_std),
                    rc=ResolverConfidence.PARTIAL,  # Mapped -> reduced confidence
                    iterations=iterations + 1,
                ),
            )

    return found_standard_concepts


@dataclass(frozen=True)
class PathProfile:
    """
    Represents the resolved 'Anchor Concept' discovered along a graph path.

    Attributes
    ----------
    concept_id : int
        The ID of the resolved concept.
    concept_name : str
        The name of the resolved concept.
    is_standard : bool
        True if `concept_id` is a Standard OMOP Concept.
    original_concept_id : int
        The ID of the starting node (candidate).
    original_concept_name : str
        The name of the starting node.
    path : GraphPath
        The full topological path.
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
        kg: "KnowledgeGraph",
        path: GraphPath,
        confidence: ResolverConfidence,
        embedding_sims: Optional[np.ndarray] = None,
    ) -> "PathProfile":
        """
        Analyze a path to determine the 'Standard Anchor'.

        It traverses the path from the candidate term. The first Standard Concept
        encountered via a MAPPING or VERSIONING edge is promoted as the Anchor.
        """
        # Path Traversal
        standard_anchor: Optional[Tuple[int, str]] = None

        # Pre-fetch views to check standard status
        # path.nodes() returns tuple of IDs (start + all objects)
        node_ids = path.nodes()
        concept_views = kg.concept_views(node_ids)
        
        # NOTE: kg.concept_views usually returns tuple.
        # If order is guaranteed, we can index by step.
        # Ideally, map by ID to be safe.
        view_map = {v.concept_id: v for v in concept_views}
        
        # Helper to get view by index in path sequence
        def get_view(idx):
            cid = node_ids[idx]
            return view_map[cid]

        predicate_kinds = kg.predicate_kinds(tuple(p.predicate for p in path.steps))

        for step_idx in range(len(path.steps)):
            predicate_kind = predicate_kinds[step_idx]

            # We promote the first swap to a standard concept as the anchor point
            # Check Next Node (index + 1)
            next_view = get_view(step_idx + 1)
            
            is_translation_edge = predicate_kind in (
                PredicateKind.MAPS_TO,
                PredicateKind.VERSIONING,
                PredicateKind.MAPS_FROM,
            )

            if (
                is_translation_edge
                and not standard_anchor
                and next_view.standard_concept
            ):
                standard_anchor = (next_view.concept_id, next_view.concept_name)
            
            # Logic for scoring/indices removed as it wasn't used in return

        first_view = get_view(0)
        
        if standard_anchor is None:
            concept_id = first_view.concept_id
            concept_name = first_view.concept_name
            is_standard = first_view.standard_concept
        else:
            concept_id, concept_name = standard_anchor
            is_standard = True

        return cls(
            concept_id=concept_id,
            concept_name=concept_name,
            is_standard=is_standard,
            original_concept_id=first_view.concept_id,
            original_concept_name=first_view.concept_name,
            path=path,
        )


@dataclass(frozen=True)
class PathExplanationStep:
    """
    A single step in the explanation of a path.
    """
    step: PathStep
    traversal_depth: Optional[int]
    predicate_kind: PredicateKind
    reason: str


@dataclass(frozen=True)
class PathExplanation:
    """
    A full explanation of a graph path, including semantic reasoning.
    """
    path: GraphPath
    profile: PathProfile
    steps: tuple[PathExplanationStep, ...]

    @classmethod
    def from_path(
        cls,
        kg: "KnowledgeGraph",
        path: GraphPath,
        trace: GraphTrace,
        confidence: ResolverConfidence,
    ) -> "PathExplanation":
        """
        Construct an explanation by combining the path, the trace log, and semantic profiles.
        """
        steps: List[PathExplanationStep] = []
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


def trace_contains_step(trace: GraphTrace, step: PathStep) -> Optional[TraceStep]:
    """
    Check if a specific path step appears in the search trace.
    """
    for ts in trace.steps:
        # Check if the expansion node matches subject
        if ts.node != step.subject.concept_id:
             # step.subject is Node object, trace uses ID (int) usually.
             # In BFS above, `cur` was int.
             # Adjusted check:
             if ts.node != step.subject.concept_id:
                 continue
        
        for e in ts.expanded_edges:
            if (
                e.object_id == step.object.concept_id
                and e.predicate_id == step.predicate
            ):
                return ts
    return None