from __future__ import annotations
from dataclasses import dataclass
from collections import deque, defaultdict
from typing import Optional
import itertools
import heapq

from omop_graph.graph import kg

from .edges import PredicateKind, EdgeView
from .traverse import traverse, GraphTrace, TraceStep


"""
Pathfinding algorithms.

Pure path-finding functions that accept a KnowledgeGraph instance

Scope: Algorithms that find paths between nodes.
i.e. What paths exist between nodes (does not yet score or explain them)
"""

@dataclass(frozen=True)
class PathStep:
    subject: int
    predicate: str
    object: int

@dataclass(frozen=True)
class GraphPath:
    steps: tuple[PathStep, ...]

    def nodes(self):
        if not self.steps:
            return ()
        return (self.steps[0].subject,) + tuple(s.object for s in self.steps)
    
    def __getitem__(self, index):
        return self.steps[index]
    
    def __len__(self):
        return len(self.steps)

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


from collections import deque, defaultdict
from typing import Optional

def find_shortest_paths_batch(
    kg,
    source: int,
    target: int,
    *,
    predicate_kinds: set[PredicateKind] | None = None,
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


def find_shortest_paths_dijkstra(
    kg,
    source: int,
    target: int,
    max_weight: float = 10.0,
    predicate_kinds: set[PredicateKind] | None = None,
    max_paths: int = 20,
    on=None,
):
    # Tie-breaker for when weights are identical (prevents PathStep comparison error)
    counter = itertools.count()
    
    pq = [(0.0, next(counter), source, [])]
    visited: dict[int, float] = {}

    while pq:
        current_w, _, u, path = heapq.heappop(pq)

        if u == target:
            assert all(isinstance(step, PathStep) for step in path), "PathStep expected"
            return GraphPath(path), current_w  # type: ignore

        if u in visited and visited[u] <= current_w:
            continue
        visited[u] = current_w

        edges = list(kg.iter_edges(u, direction="out", predicate_kinds=predicate_kinds, on=on))
        
        # Calculate the penalty ONCE for this source node
        # Logarithmic scaling is usually better so weights don't explode
        # weight_penalty = 0.1 * len(edges) 
        import math
        weight_penalty = math.log1p(len(edges)) # log(1 + degree) is a standard NLP/Graph approach
        edge_weight = 1.0 + weight_penalty

        for e in edges:
            v = e.object_id
            new_w = current_w + edge_weight
            
            if new_w <= max_weight:
                new_step = PathStep(subject=u, predicate=e.predicate_id, object=v)
                heapq.heappush(pq, (new_w, next(counter), v, path + [new_step]))

    return None, None
