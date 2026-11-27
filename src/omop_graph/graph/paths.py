"""
Pathfinding algorithms for the OMOP Knowledge Graph.

This module provides pure path-finding functions that accept a `KnowledgeGraph`
instance. It focuses on discovering topological connections between nodes,
that is the shortest paths between two concepts.

Scope
-----
Algorithms that find paths between nodes. This module answers "what paths exist"
but does not score or explain them, which the ``reasoning`` package does.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from operator import itemgetter
from typing import TYPE_CHECKING, Literal

from omop_graph.graph.model.edges import DEFAULT_EDGE_FILTER, EdgeFilter
from omop_graph.graph.traverse import GraphTrace, TerminationReason, TraceStep

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


def _distinct_paths(
    kg: KnowledgeGraph,
    source: int,
    target: int,
    meeting_nodes: Iterable[int],
    parents_fwd: dict[int, list[tuple[int, str]]],
    parents_bwd: dict[int, list[tuple[int, str]]],
    max_paths: int,
) -> list[GraphPath]:
    """
    Build the paths through the meeting nodes, each once, up to max_paths.

    A path that passes through several meeting nodes is built from each of them, so the
    paths are collected as a set before they are counted. The meeting nodes are taken in
    ascending order, which makes the cut at max_paths the same on every run. The concepts
    on the paths are read in one batch, so that every ``Node`` carries its standard flag.

    Parameters
    ----------
    kg : KnowledgeGraph
        The graph to read the concepts from.
    source : int
        Start concept ID.
    target : int
        End concept ID.
    meeting_nodes : Iterable[int]
        The concepts where the two ends of the search met.
    parents_fwd : dict[int, list[tuple[int, str]]]
        The parents of each concept reached from the source.
    parents_bwd : dict[int, list[tuple[int, str]]]
        The parents of each concept reached from the target.
    max_paths : int
        Maximum number of paths to return.

    Returns
    -------
    list[GraphPath]
        The distinct paths, at most max_paths.
    """
    discovered = tuple(set(parents_fwd) | set(parents_bwd) | {source, target})
    standard = {view.concept_id: view.standard_concept for view in kg.concept_views(discovered)}
    paths: dict[GraphPath, None] = {}
    for meet in sorted(meeting_nodes):
        paths.update(
            dict.fromkeys(
                reconstruct_paths(source, target, meet, parents_fwd, parents_bwd, standard)
            )
        )
        if len(paths) >= max_paths:
            break
    return list(paths)[:max_paths]


@dataclass
class _SearchEnd:
    """
    One end of a bidirectional search and what has been reached from it.

    Parameters
    ----------
    direction : Literal["out", "in"]
        The edges read from a concept: ``out`` from the source, ``in`` from the target.
    depth : dict[int, int]
        The number of steps from this end to each concept reached.
    parents : dict[int, list[tuple[int, str]]]
        For each concept, the concepts one step nearer this end with the relationship
        between them, for every shortest way to reach it.
    frontier : tuple[int, ...]
        The concepts reached last, which are expanded next, in ascending order.
    """

    direction: Literal["out", "in"]
    depth: dict[int, int]
    parents: dict[int, list[tuple[int, str]]] = field(default_factory=lambda: defaultdict(list))
    frontier: tuple[int, ...] = ()

    @classmethod
    def at(cls, concept_id: int, direction: Literal["out", "in"]) -> _SearchEnd:
        """The end that has reached only its own concept."""
        return cls(direction, {concept_id: 0}, frontier=(concept_id,))

    @property
    def level(self) -> int:
        """The number of steps from this end to its frontier."""
        return self.depth[self.frontier[0]]

    def expand(
        self,
        kg: KnowledgeGraph,
        edge_filter: EdgeFilter,
        other: _SearchEnd,
        traced: bool,
    ) -> tuple[set[int], list[TraceStep]]:
        """
        Read the edges of the whole frontier in one query and move one step further.

        The edges are processed in ascending order of their concepts, so the result does not
        depend on the order the database returns them in.

        Parameters
        ----------
        kg : KnowledgeGraph
            The graph to read.
        edge_filter : EdgeFilter
            Which edges to follow.
        other : _SearchEnd
            The other end of the search.
        traced : bool
            If True, record the edges read from each concept of the frontier.

        Returns
        -------
        tuple[set[int], list[TraceStep]]
            The concepts reached that the other end has reached too, and the trace steps.
        """
        frontier, level = self.frontier, self.level
        reads = sorted(
            (
                (e.subject_id, e.object_id, e.predicate_id, e)
                if self.direction == "out"
                else (e.object_id, e.subject_id, e.predicate_id, e)
                for e in kg.edges(frontier, self.direction, edge_filter)
            ),
            key=itemgetter(0, 1, 2),
        )
        met: set[int] = set()
        reached: set[int] = set()
        for near, far, predicate, _ in reads:
            if far not in self.depth:
                self.depth[far] = level + 1
                reached.add(far)
            if self.depth[far] == level + 1:
                self.parents[far].append((near, predicate))
            if far in other.depth:
                met.add(far)
        self.frontier = tuple(sorted(reached))

        steps: list[TraceStep] = []
        if traced:
            read_from = defaultdict(list)
            for near, _, _, edge in reads:
                read_from[near].append(edge)
            steps = [
                TraceStep(depth=level, node=concept_id, expanded_edges=tuple(read_from[concept_id]))
                for concept_id in frontier
            ]
        return met, steps


def find_shortest_paths(
    kg: KnowledgeGraph,
    source: int,
    target: int,
    *,
    edge_filter: EdgeFilter = DEFAULT_PATH_FILTER,
    max_depth: int = 6,
    max_paths: int = 20,
    traced: bool = False,
) -> tuple[list[GraphPath], GraphTrace | None]:
    """
    Find the shortest paths between source and target, searching from both ends.

    Each step reads the edges of a whole frontier in one query, from the end with fewer
    concepts to expand. The result does not depend on the order the database returns edges in.

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
        Maximum number of steps of a path. The two ends share this budget: an end stops
        expanding once the depths of both together would exceed it. Defaults to ``6``.
    max_paths : int
        Maximum number of paths to return, each once. When there are more, the paths through
        the lowest concept id where the two ends meet come first. Defaults to ``20``.
    traced : bool
        If True, returns a GraphTrace object recording the search process.

    Returns
    -------
    tuple[list[GraphPath], GraphTrace | None]
        A list of paths and optionally the trace object.
    """
    if source == target:
        trace = (
            GraphTrace(
                seeds=(source,), steps=[], terminated_reason=TerminationReason.SOURCE_EQUALS_TARGET
            )
            if traced
            else None
        )
        return [GraphPath(steps=())], trace

    forward = _SearchEnd.at(source, "out")
    backward = _SearchEnd.at(target, "in")
    meeting_nodes: set[int] = set()
    trace_steps: list[TraceStep] = []

    # An end with nothing left to expand has reached everything it can, so no path exists.
    while forward.frontier and backward.frontier and forward.level + backward.level < max_depth:
        end, other = (
            (forward, backward)
            if len(forward.frontier) <= len(backward.frontier)
            else (backward, forward)
        )
        meeting_nodes, steps = end.expand(kg, edge_filter, other, traced)
        trace_steps += steps
        if meeting_nodes:
            break

    if not meeting_nodes:
        reason = TerminationReason.NO_PATH
        paths: list[GraphPath] = []
    else:
        reason = TerminationReason.SHORTEST_PATHS_FOUND
        paths = _distinct_paths(
            kg, source, target, meeting_nodes, forward.parents, backward.parents, max_paths
        )
    trace = GraphTrace(seeds=(source,), steps=trace_steps, terminated_reason=reason)
    return paths, (trace if traced else None)
