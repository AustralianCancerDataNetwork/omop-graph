from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from typing import Iterable
from datetime import date
from .edges import EdgeView, PredicateKind


"""
Graph traversal algorithms.

Scope: Algorithms that explore the graph structure.
i.e. Given some seed nodes, how do I explore the graph?
"""

@dataclass(frozen=True)
class Subgraph:
    nodes: frozenset[int]
    edges: tuple[EdgeView, ...]


@dataclass
class TraceStep:
    depth: int
    node: int
    expanded_edges: tuple[EdgeView, ...]


@dataclass
class GraphTrace:
    seeds: tuple[int, ...]
    steps: list[TraceStep]
    terminated_reason: str | None = None


def traverse(
    kg,
    seeds: Iterable[int],
    *,
    predicate_kinds: set[PredicateKind] | None,
    max_depth: int,
    on: date | None,
    max_nodes: int | None,
    trace: bool,
) -> tuple[Subgraph, GraphTrace | None]:
    
    seeds = tuple(dict.fromkeys(seeds))  # deduplicate while preserving order
    visited = set()
    edges_out = []
    steps = []

    q = deque((s, 0) for s in seeds)
    terminated = None

    while q:
        node, depth = q.popleft()

        if node in visited:
            continue

        visited.add(node)

        if max_nodes and len(visited) >= max_nodes:
            terminated = "max_nodes"
            break

        if depth >= max_depth:
            continue

        expanded: list[EdgeView] = []

        for e in kg.iter_edges(
            node,
            direction="out",
            predicate_kinds=predicate_kinds,
            active_only=True,
            on=on,
        ):
            expanded.append(e)
            edges_out.append(e)

            nxt = e.object_id
            if nxt not in visited:
                q.append((nxt, depth + 1))

        if trace:
            steps.append(TraceStep(depth=depth, node=node, expanded_edges=tuple(expanded)))

    dedup = {(e.subject_id, e.predicate_id, e.object_id): e for e in edges_out}
    sg = Subgraph(frozenset(visited), tuple(dedup.values()))

    return sg, (GraphTrace(tuple(seeds), steps, terminated) if trace else None)
