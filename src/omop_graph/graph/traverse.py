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
from contextlib import closing
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Iterable, List, Optional, Set, Tuple

# Local Application Imports
from omop_graph.extensions.omop_alchemy import PredicateKind
from omop_graph.graph.edges import EdgeView

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
    steps: List[TraceStep]
    terminated_reason: Optional[str] = None

    def summary(self, kg: "KnowledgeGraph", max_steps: int = 10) -> str:
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


@dataclass
class _TraversalState:
    queue: deque[Tuple[int, int]]
    discovered: set[int]
    max_nodes: Optional[int]
    predicate_kinds: Optional[Set[PredicateKind]]
    on: Optional[date]


def _expand_node(
    kg: "KnowledgeGraph",
    node: int,
    depth: int,
    state: _TraversalState,
) -> Tuple[List[EdgeView], bool]:
    """Collect one node's edges, stopping when a new target fills the budget."""
    expanded: List[EdgeView] = []
    with kg.session_factory() as session:
        edge_iter = kg.iter_edges(
            session=session,
            concept_ids=node,
            direction="out",
            predicate_kinds=frozenset(state.predicate_kinds)
            if state.predicate_kinds is not None
            else None,
            active_only=True,
            on=state.on,
            stream=True,
        )
        with closing(edge_iter):
            for edge in edge_iter:
                if edge.object_id not in state.discovered:
                    state.discovered.add(edge.object_id)
                    state.queue.append((edge.object_id, depth + 1))

                expanded.append(edge)
                if state.max_nodes and len(state.discovered) >= state.max_nodes:
                    return expanded, True

    return expanded, False


def traverse(
    kg: "KnowledgeGraph",
    seeds: Iterable[int],
    predicate_kinds: Optional[Set[PredicateKind]],
    max_depth: int,
    on: Optional[date],
    max_nodes: Optional[int],
    trace: bool,
) -> Tuple[Subgraph, Optional[GraphTrace]]:
    """
    Perform a Breadth-First Search (BFS) traversal starting from seed nodes.

    Parameters
    ----------
    kg : KnowledgeGraph
        The graph instance to query.
    seeds : Iterable[int]
        The starting Concept IDs.
    predicate_kinds : set[PredicateKind], optional
        Restrict traversal to specific edge types.
    max_depth : int
        Maximum distance from seeds to explore.
    on : date, optional
        Filter for edges active on this date.
    max_nodes : int, optional
        Stop after discovering this many unique nodes.
    trace : bool
        If True, return a GraphTrace object.

    Returns
    -------
    tuple[Subgraph, GraphTrace | None]
        The resulting subgraph and optionally the execution trace.

    When ``max_nodes`` is reached, which nodes fill the budget is unspecified
    (it depends on database row order). Traversal stops as soon as the node
    budget is discovered, so queued nodes inside the budget are not expanded;
    edges those later expansions could have found are absent from the result.
    """
    # Deduplicate seeds while preserving order
    # (Python 3.7+ dict guarantees insertion order)
    unique_seeds = tuple(dict.fromkeys(seeds))

    # Seeds precede discovered neighbours in BFS order. Keep only the seeds
    # that fit the budget, matching the order in which the old visited set
    # reached its cap.
    budgeted_seeds = unique_seeds[: max(max_nodes, 1)] if max_nodes else unique_seeds
    discovered = set(budgeted_seeds)
    edges_out: List[EdgeView] = []
    trace_steps: List[TraceStep] = []

    # Queue stores (concept_id, depth)
    q = deque((s, 0) for s in budgeted_seeds)
    state = _TraversalState(q, discovered, max_nodes, predicate_kinds, on)
    terminated = None

    while q:
        node, depth = q.popleft()

        if state.max_nodes and len(state.discovered) >= state.max_nodes:
            terminated = "max_nodes"
            break

        if depth >= max_depth:
            continue

        # Iterate over outgoing edges until one more node would exceed the budget.
        expanded, budget_reached = _expand_node(kg, node, depth, state)
        edges_out.extend(expanded)

        if trace:
            trace_steps.append(
                TraceStep(depth=depth, node=node, expanded_edges=tuple(expanded))
            )
        if budget_reached:
            terminated = "max_nodes"
            break

    # Deduplicate edges and drop any that target a node outside the budget.
    dedup = {
        (e.subject_id, e.predicate_id, e.object_id): e
        for e in edges_out
        if e.subject_id in state.discovered and e.object_id in state.discovered
    }
    sg = Subgraph(nodes=frozenset(state.discovered), edges=tuple(dedup.values()))

    graph_trace = (
        GraphTrace(seeds=unique_seeds, steps=trace_steps, terminated_reason=terminated)
        if trace
        else None
    )

    return sg, graph_trace
