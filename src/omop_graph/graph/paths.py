from __future__ import annotations
from dataclasses import dataclass
from collections import deque, defaultdict
from typing import Optional

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

