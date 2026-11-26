"""
Pathfinding algorithms for the OMOP Knowledge Graph.

This module provides pure path-finding functions that accept a `KnowledgeGraph`
instance. It focuses on discovering topological connections between nodes,
including shortest paths and batch traversal.

Scope
-----
Algorithms that find paths between nodes. This module answers "what paths exist"
but does not score or explain them, which the `reasoning` package does.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from omop_graph.graph.model.edges import DEFAULT_EDGE_FILTER, EdgeFilter, EdgeView
from omop_graph.graph.traverse import GraphTrace, TraceStep

if TYPE_CHECKING:
    from omop_graph.graph.kg import KnowledgeGraph


DEFAULT_PATH_FILTER = replace(DEFAULT_EDGE_FILTER, within_domain=True)
"""Edges path search follows by default: active, not metadata, within one domain to limit noise."""


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

    def get_first_standard_concept_id(self) -> int | None:
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
        return (self.steps[0].subject.concept_id,) + tuple(s.object.concept_id for s in self.steps)

    def __getitem__(self, index):
        return self.steps[index]

    def __len__(self):
        return len(self.steps)

    def __repr__(self) -> str:
        if not self.steps:
            return "GraphPath(<empty>)"
        return f"GraphPath(len={len(self.steps)})"

    def explain(self, kg: KnowledgeGraph) -> str:
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

            parts.append(f"{subj.concept_name} -[{pred.name}]-> {obj.concept_name}")

        return "\n  ↳ ".join(parts)


def reconstruct_paths(
    source,
    target,
    meet,
    parents_fwd,
    parents_bwd,
    concept_standard_map: dict[int, bool],
):
    """
    Reconstruct full paths from bidirectional BFS parent pointers.

    Parameters
    ----------
    concept_standard_map : dict[int, bool]
        Mapping of concept_id → is_standard for all nodes discovered during BFS.
        Built with a single batched ``kg.concept_views`` call after the BFS completes
        so that every ``Node`` carries the correct flag with zero extra DB round-trips.
    """

    def std(cid: int) -> bool:
        return concept_standard_map.get(cid, False)

    def left(n):
        if n == source:
            return [()]
        out = []
        for p, pred in parents_fwd[n]:
            for L in left(p):
                out.append(L + (PathStep(Node(p, std(p)), pred, Node(n, std(n))),))
        return out

    def right(n):
        if n == target:
            return [()]
        out = []
        for nxt, pred in parents_bwd[n]:
            for R in right(nxt):
                out.append((PathStep(Node(n, std(n)), pred, Node(nxt, std(nxt))),) + R)
        return out

    return [GraphPath(L + R) for L in left(meet) for R in right(meet)]


def find_shortest_paths(
    kg: KnowledgeGraph,
    source: int,
    target: int,
    edge_filter: EdgeFilter = DEFAULT_PATH_FILTER,
    max_depth: int = 6,
    max_paths: int = 20,
    traced: bool = False,
) -> tuple[list[GraphPath], GraphTrace | None]:
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
    edge_filter : EdgeFilter
        Which edges to traverse. Defaults to active edges within a domain; set
        ``within_domain=False`` to allow cross-domain edges such as SNOMED
        attribute relationships (Has asso morph, Has finding site, etc.).
    max_depth : int
        Maximum depth searched from each end, so a path can have up to twice as many
        steps. Defaults to ``6``.
    max_paths : int
        Maximum number of paths to return. Defaults to ``20``.
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
            GraphTrace(seeds=(source,), steps=[], terminated_reason="source_equals_target")
            if traced
            else None
        )
        return [path], trace

    q_fwd = deque([source])
    q_bwd = deque([target])

    depth_fwd = {source: 0}
    depth_bwd = {target: 0}

    parents_fwd: dict[int, list[tuple[int, str]]] = defaultdict(list)
    parents_bwd: dict[int, list[tuple[int, str]]] = defaultdict(list)

    best_total_depth: int | None = None
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

            for e in kg.edges(cur, "out", edge_filter):
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

            for e in kg.edges(cur, "in", edge_filter):
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

    # One batched lookup to get is_standard for every discovered concept so that
    # reconstructed Node objects carry the correct flag (zero extra per-node DB calls).
    all_discovered = tuple(set(depth_fwd.keys()) | set(depth_bwd.keys()))
    concept_standard_map: dict[int, bool] = {
        v.concept_id: v.standard_concept for v in kg.concept_views(all_discovered)
    }

    paths: list[GraphPath] = []
    for meet in meeting_nodes:
        paths.extend(
            reconstruct_paths(source, target, meet, parents_fwd, parents_bwd, concept_standard_map)
        )
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
    kg: KnowledgeGraph,
    source: int,
    target: int,
    edge_filter: EdgeFilter = DEFAULT_PATH_FILTER,
    max_depth: int = 6,
    max_paths: int = 20,
) -> list[GraphPath]:
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
    edge_filter : EdgeFilter
        Which edges to traverse. Defaults to active edges within a domain; set
        ``within_domain=False`` to allow cross-domain edges such as SNOMED
        attribute relationships (Has asso morph, Has finding site, etc.).
    max_depth : int
        Maximum depth searched from each end, so a path can have up to twice as many
        steps. Defaults to ``6``.
    max_paths : int
        Maximum number of paths to return. Defaults to ``20``.

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
    parents_fwd: dict[int, list[tuple[int, str]]] = defaultdict(list)
    parents_bwd: dict[int, list[tuple[int, str]]] = defaultdict(list)

    best_total_depth: int | None = None
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
        batch_edges = kg.edges(current_layer_nodes, direction, edge_filter)

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

    if not meeting_nodes:
        return []

    all_discovered = tuple(set(depth_fwd.keys()) | set(depth_bwd.keys()))
    concept_standard_map: dict[int, bool] = {
        v.concept_id: v.standard_concept for v in kg.concept_views(all_discovered)
    }

    paths: list[GraphPath] = []
    for meet in meeting_nodes:
        paths.extend(
            reconstruct_paths(source, target, meet, parents_fwd, parents_bwd, concept_standard_map)
        )
        if len(paths) >= max_paths:
            break

    return paths[:max_paths]
