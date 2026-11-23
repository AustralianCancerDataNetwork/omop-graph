"""
Graph traversal algorithms.

This module provides generic algorithms for exploring the graph structure starting
from a set of seed nodes. It handles BFS expansion, depth limits, and execution tracing.

Scope
-----
Algorithms that explore the graph structure (e.g., "Find all neighbors within 3 hops").
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from contextlib import closing
from dataclasses import dataclass
from typing import TYPE_CHECKING

# Local Application Imports
from omop_graph.graph.edges import EdgeFilter, EdgeView

if TYPE_CHECKING:
    from omop_graph.graph.kg import KnowledgeGraph


@dataclass(frozen=True)
class Subgraph:
    """
    A subset of the graph consisting of a specific set of nodes and edges.

    Parameters
    ----------
    nodes : frozenset[int]
        The set of Concept IDs included in this subgraph.
    edges : tuple[EdgeView, ...]
        The edges connecting these nodes.
    """

    nodes: frozenset[int]
    edges: tuple[EdgeView, ...]


@dataclass
class TraceStep:
    """
    A single step in a traversal trace, recording the state at one node expansion.

    Parameters
    ----------
    depth : int
        The depth at which this node was visited.
    node : int
        The Concept ID of the node being expanded.
    expanded_edges : tuple[EdgeView, ...]
        The edges found outgoing from this node.
    """

    depth: int
    node: int
    expanded_edges: tuple[EdgeView, ...]


@dataclass
class GraphTrace:
    """
    A record of a full graph traversal execution.

    Useful for debugging why a path was or wasn't found.

    Parameters
    ----------
    seeds : tuple[int, ...]
        The starting concept IDs.
    steps : list[TraceStep]
        The sequence of expansions performed.
    terminated_reason : str, optional
        Why the traversal stopped (e.g., 'max_nodes', 'max_depth').
    """

    seeds: tuple[int, ...]
    steps: list[TraceStep]
    terminated_reason: str | None = None

    def summary(self, kg: KnowledgeGraph, max_steps: int = 10) -> str:
        """
        Generate a human-readable summary of the traversal.

        Parameters
        ----------
        kg : KnowledgeGraph
            Graph instance for name lookups.
        max_steps : int
            Number of steps to show before truncating.

        Returns
        -------
        str
            Summary string.
        """
        lines = [f"Seeds: {self.seeds}"]
        for _, step in enumerate(self.steps[:max_steps]):
            concept = kg.concept_view(step.node)
            lines.append(
                f"[depth={step.depth}] expanded {concept.concept_name} "
                f"({len(step.expanded_edges)} edges)"
            )
        if len(self.steps) > max_steps:
            lines.append(f"... ({len(self.steps) - max_steps} more steps)")
        lines.append(f"Terminated: {self.terminated_reason}")
        return "\n".join(lines)


def traverse(
    kg: KnowledgeGraph,
    seeds: Iterable[int],
    edge_filter: EdgeFilter | None,
    max_depth: int,
    max_nodes: int | None,
    trace: bool,
) -> tuple[Subgraph, GraphTrace | None]:
    """
    Perform a Breadth-First Search (BFS) traversal starting from seed nodes.

    When ``max_nodes`` is reached, which nodes fill the budget is deterministic
    because edges are read in key order. Queued nodes inside the budget are not
    expanded, so edges those later expansions could have found can be absent.
    When the seeds alone reach the budget, no seed is expanded and the result
    has no edges.

    Parameters
    ----------
    kg : KnowledgeGraph
        The graph instance to query.
    seeds : Iterable[int]
        The starting Concept IDs.
    edge_filter : EdgeFilter, optional
        Which edges to follow. Defaults to ``EdgeFilter()``.
    max_depth : int
        Maximum distance from seeds to explore.
    max_nodes : int, optional
        Stop after discovering this many unique nodes.
    trace : bool
        If True, return a GraphTrace object.

    Returns
    -------
    tuple[Subgraph, GraphTrace | None]
        The resulting subgraph and optionally the execution trace.
    """
    # Deduplicate seeds while preserving order
    # (Python 3.7+ dict guarantees insertion order)
    unique_seeds = tuple(dict.fromkeys(seeds))

    # Seeds precede discovered neighbours in BFS order. Keep only the seeds
    # that fit the budget, matching the order in which the old visited set
    # reached its cap.
    budgeted_seeds = unique_seeds[: max(max_nodes, 1)] if max_nodes else unique_seeds
    discovered = set(budgeted_seeds)
    edges_out: list[EdgeView] = []
    trace_steps: list[TraceStep] = []

    # Queue stores (concept_id, depth)
    q = deque((s, 0) for s in budgeted_seeds)
    terminated = None

    while q:
        node, depth = q.popleft()

        if max_nodes and len(discovered) >= max_nodes:
            terminated = "max_nodes"
            break

        if depth >= max_depth:
            continue

        expanded: list[EdgeView] = []
        with kg.session_factory() as session:
            edge_iter = kg.iter_edges(
                session=session,
                concept_ids=node,
                direction="out",
                edge_filter=edge_filter,
                ordered=True,
            )
            with closing(edge_iter):
                for edge in edge_iter:
                    if edge.object_id not in discovered:
                        discovered.add(edge.object_id)
                        q.append((edge.object_id, depth + 1))

                    expanded.append(edge)
                    edges_out.append(edge)
                    if max_nodes and len(discovered) >= max_nodes:
                        terminated = "max_nodes"
                        break

        if trace:
            trace_steps.append(TraceStep(depth=depth, node=node, expanded_edges=tuple(expanded)))
        if terminated == "max_nodes":
            break

    dedup = {(e.subject_id, e.predicate_id, e.object_id): e for e in edges_out}
    sg = Subgraph(nodes=frozenset(discovered), edges=tuple(dedup.values()))

    graph_trace = (
        GraphTrace(seeds=unique_seeds, steps=trace_steps, terminated_reason=terminated)
        if trace
        else None
    )

    return sg, graph_trace
