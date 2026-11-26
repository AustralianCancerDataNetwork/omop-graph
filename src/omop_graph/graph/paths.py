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
from collections import defaultdict, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import (
    TYPE_CHECKING,
)

# Local Application Imports
from omop_graph.extensions.omop_alchemy import PredicateKind
from omop_graph.graph.model.edges import DEFAULT_EDGE_FILTER, EdgeFilter, EdgeView
from omop_graph.graph.model.nodes import AncestorMatch, ConceptView, LabelMatchKind
from omop_graph.graph.traverse import GraphTrace, TraceStep
from omop_graph.reasoning.resolvers import CandidateHit

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
        Maximum path length. Defaults to ``6``.
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

            with kg.session_factory() as session:
                for e in kg.iter_edges(
                    session=session,
                    concept_ids=cur,
                    direction="out",
                    edge_filter=edge_filter,
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

            with kg.session_factory() as session:
                for e in kg.iter_edges(
                    session=session,
                    concept_ids=cur,
                    direction="in",
                    edge_filter=edge_filter,
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
        Maximum path length. Defaults to ``6``.
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
        with kg.session_factory() as session:
            batch_edges = kg.iter_edges(
                session=session,
                concept_ids=current_layer_nodes,
                direction=direction,
                edge_filter=edge_filter,
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


@dataclass(order=True)
class QueueItem:
    """Frontier item for the cost-prioritised BFS in find_standard_paths.

    Attributes
    ----------
    cost : float
        Accumulated traversal cost. Currently always 0.0 (uniform BFS). Reserved as
        live infrastructure for future weighted traversal; see Notes in find_standard_paths.
    node : Node
        The graph node at this position in the frontier.
    candidate_index : int
        Position in the search's candidates of the candidate this item was reached from.
    iterations : int
        BFS depth (number of hops from the candidate).
    """

    cost: float
    node: Node = field(compare=False)
    candidate_index: int = field(compare=False)
    iterations: int = field(default=0, compare=False)


@dataclass(frozen=True)
class StandardConcept:
    """
    A resolved Standard Concept resulting from a search.

    Attributes
    ----------
    concept_id : int
        The OMOP Concept ID of the resolved standard concept.
    concept_name : str
        The name of the resolved standard concept.
    separation : int
        How far this standard concept is from where it needs to be, with a meaning that
        depends on whether ancestor targets were given to ``find_standard_paths``:
            - Targets given (ancestor-constrained grounding): this is the ancestor-
            hierarchy distance (via ``concept_ancestor.min_levels_of_separation``)
            from this concept to the required parent.
            - No targets given (unconstrained grounding): this is the hop count
            from the original found concept to the standard concept.
        This is the only distance field consumed by scoring (``scoring.py``'s parsimony penalty).
    original_id : int
        The OMOP Concept ID of the original candidate that search started from.
    original_name : str
        The name of the original candidate concept.
    matched_concept_label : str
        The text (name or synonym) that the original candidate matched on.
    match_kind : LabelMatchKind
        How the original candidate was matched (exact, partial, full-text, embedding).
    synonym : bool
        Whether the original candidate matched via a synonym rather than the
        primary concept name.
    hierarchy_cost : float, default 0.0
        Reserved for future weighted traversal; see ``QueueItem.cost`` and the
        Notes on ``find_standard_paths``. Currently always 0.0.
    identity_hops : int, default 0
        The number of edges walked from the original candidate to reach this concept.
        Not used in scoring as of now.
    """

    concept_id: int
    concept_name: str
    separation: int
    original_id: int
    original_name: str
    matched_concept_label: str
    match_kind: LabelMatchKind
    synonym: bool
    hierarchy_cost: float = 0.0
    identity_hops: int = 0


def _load_views(
    kg: KnowledgeGraph, views: dict[int, ConceptView], concept_ids: Iterable[int]
) -> None:
    """Add the views of the concepts in concept_ids that views lacks."""
    missing = {concept_id for concept_id in concept_ids if concept_id not in views}
    for batch in kg.batches(missing):
        views.update((view.concept_id, view) for view in kg.concept_views(batch, sort=False))


def find_standard_paths(
    kg: KnowledgeGraph,
    targets: tuple[int, ...] | None,
    candidates: Sequence[CandidateHit],
    edge_filter: EdgeFilter = DEFAULT_PATH_FILTER,
    max_depth: int = 6,
    max_concepts: int | None = None,
) -> list[list[StandardConcept]]:
    """
    Search for the standard concepts reachable from each candidate, optionally verified
    against ancestor targets.

    Performs a breadth-first search (BFS) from all candidates at once. Each BFS wave drains
    the entire queue, and reads the edges, concepts and ancestors of the whole wave in
    single queries. Non-standard concepts are expanded by following their outgoing edges
    to standard concepts.

    Notes
    -----
    This function has two modes:

    1. When ``targets`` is provided, the standard concepts of each wave are checked against
    the targets with one batched concept_ancestor query. Standard concepts that satisfy at
    least one target ancestor constraint are recorded and not expanded further, preventing
    dilution by more distant concepts. Standard concepts with no ancestry match are expanded
    further (e.g. deprecated-standard -> replacement-standard chains).

    2. When ``targets`` is None or empty, there is no ancestor to verify against: the
    first standard concept reached on each branch is accepted directly and not expanded
    further. This is a looser, unconstrained form of grounding. It "grounds" a
    candidate to its standard form without being able to disambiguate against a known
    hierarchy branch. See ``StandardConcept.separation`` for how distance is measured in
    this mode.

    Parameters
    ----------
    kg : KnowledgeGraph
        The graph instance.
    targets : tuple of int, optional
        Ancestor concept IDs to verify candidates against. A result is produced for
        each target that a reached standard concept is a genuine descendant of. When
        None or empty, no ancestor verification is performed (see above).
    candidates : Sequence of CandidateHit
        The initial search hits to start traversal from.
    edge_filter : EdgeFilter
        Which edges to traverse. Defaults to active edges within a domain. Callers
        in the grounding pipeline restrict it to ``PredicateKind.IDENTITY``, limiting
        traversal to Maps-to relationships between non-standard and standard concepts.
    max_depth : int
        Maximum min_levels_of_separation permitted in the concept_ancestor check when
        ``targets`` is given, or maximum identity-hop count permitted when ``targets``
        is None. Defaults to ``6``.
    max_concepts : int, optional
        Per-target cap on unique standard concepts collected for each candidate (or an
        overall cap when ``targets`` is None). Once every target of a candidate has reached
        this count, the search for that candidate stops.

    Returns
    -------
    list of list of StandardConcept
        For each candidate in order, its deduplicated standard concepts that satisfy at
        least one target ancestor constraint, or, when ``targets`` is None, every standard
        concept reached.

    Notes
    -----
    The search is currently plain BFS because all edge costs are uniform (0.0). The
    QueueItem.cost field and the heapq structure are preserved as infrastructure for
    future weighted traversal. To upgrade to Dijkstra, define a COST_PREDICATES mapping
    from PredicateKind to a numeric cost (e.g. IDENTITY=0, HIERARCHY=1, ASSOCIATION=2),
    set new_cost accordingly in the expansion loop, and change the wave drain from a
    full-queue drain to a single-cost-level drain so that lower-cost nodes are always
    processed before higher-cost ones. A* would additionally require a domain-specific
    admissible heuristic added to the priority.
    """
    # Placeholder for the non-parent-ID grounding so the target accumulation works
    unconstrained_placeholder: int = 0

    views: dict[int, ConceptView] = {}
    _load_views(kg, views, (candidate.concept_id for candidate in candidates))

    queue = [
        QueueItem(
            cost=0.0,
            node=Node(candidate.concept_id, bool(views[candidate.concept_id].standard_concept)),
            candidate_index=index,
            iterations=0,
        )
        for index, candidate in enumerate(candidates)
    ]
    # Per candidate, the shallowest iteration we have enqueued per concept to avoid
    # unbounded duplicate growth in high-degree neighborhoods.
    visited_min_iteration: list[dict[int, int]] = [
        {candidate.concept_id: 0} for candidate in candidates
    ]
    found_standard_concepts: list[dict[int, list[StandardConcept]]] = [
        {target_id: [] for target_id in targets} if targets else {unconstrained_placeholder: []}
        for _ in candidates
    ]

    def is_done(candidate_index: int) -> bool:
        if not max_concepts:
            return False
        found = found_standard_concepts[candidate_index]
        return all(len(concepts) >= max_concepts for concepts in found.values())

    def standard_concept_of(item: QueueItem, separation: int) -> StandardConcept:
        candidate = candidates[item.candidate_index]
        return StandardConcept(
            hierarchy_cost=item.cost,
            concept_id=item.node.concept_id,
            concept_name=views[item.node.concept_id].concept_name,
            separation=separation,
            original_id=candidate.concept_id,
            original_name=views[candidate.concept_id].concept_name,
            matched_concept_label=candidate.matched_concept_label,
            match_kind=candidate.match_kind,
            synonym=candidate.synonym,
            identity_hops=item.iterations,
        )

    while queue:
        # Drain the entire heap to process it at once and reduce round-trips to the DB,
        # leaving out the candidates whose search is done.
        wave = [
            item
            for _ in range(len(queue))
            if not is_done((item := heapq.heappop(queue)).candidate_index)
        ]

        dedup_standard_items: dict[tuple[int, int], QueueItem] = {}
        for item in wave:
            if item.node.is_standard:
                dedup_standard_items.setdefault((item.candidate_index, item.node.concept_id), item)
        standard_items = list(dedup_standard_items.values())
        expand_items = [item for item in wave if not item.node.is_standard]

        if targets:
            multi_ancestors: dict[int, dict[int, AncestorMatch]] = {}
            standard_ids = {item.node.concept_id for item in standard_items}
            for batch in kg.batches(standard_ids):
                multi_ancestors.update(kg.get_potential_ancestors_batch(batch, targets))

            for item in standard_items:
                ancestor_matches = multi_ancestors.get(item.node.concept_id)
                if not ancestor_matches:
                    # No target is an ancestor, so keep walking (e.g. to a replacement).
                    expand_items.append(item)
                    continue
                found = found_standard_concepts[item.candidate_index]
                for target_id, potential_ancestor in ancestor_matches.items():
                    if potential_ancestor.min_levels_of_separation > max_depth:
                        continue
                    if max_concepts and len(found[target_id]) >= max_concepts:
                        continue
                    found[target_id].append(
                        standard_concept_of(item, potential_ancestor.min_levels_of_separation)
                    )
        else:
            # Unconstrained: there is no ancestor to check, so accept the standard concept.
            for item in standard_items:
                found = found_standard_concepts[item.candidate_index][unconstrained_placeholder]
                if max_concepts and len(found) >= max_concepts:
                    continue
                found.append(standard_concept_of(item, item.iterations))

        # Expand: Go to next best concept_id. One query reads the edges of the whole wave.
        object_ids_by_subject: defaultdict[int, list[int]] = defaultdict(list)
        subject_ids = {item.node.concept_id for item in expand_items}
        for batch in kg.batches(subject_ids):
            for subject_id, _, object_id in kg.relationships(batch, edge_filter=edge_filter):
                object_ids_by_subject[subject_id].append(object_id)
        _load_views(kg, views, (o for ids in object_ids_by_subject.values() for o in ids))

        for item in expand_items:
            next_iterations = item.iterations + 1
            visited = visited_min_iteration[item.candidate_index]

            for object_id in dict.fromkeys(object_ids_by_subject[item.node.concept_id]):
                object_view = views.get(object_id)

                # Optimization: Only traverse to Standard concepts
                if object_view is None or not object_view.standard_concept:
                    continue

                # Early-stop for unconstrained mode:
                # Number of hops from candidate to standard concept
                # exceeds max_depth to prevent infinite expansion
                if not targets and next_iterations > max_depth:
                    continue
                prev_best_iteration = visited.get(object_id)
                if prev_best_iteration is not None and prev_best_iteration <= next_iterations:
                    continue
                visited[object_id] = next_iterations

                new_cost = item.cost
                # Cost differentiation is reserved for future use. When COST_PREDICATES is
                # defined (e.g. IDENTITY=0, HIERARCHY=1, ASSOCIATION=2), replace with:
                #   new_cost = cost + COST_PREDICATES[predicate_kind_for_this_edge]
                # If costs become non-uniform the wave-drain must change to a per-cost-tier
                # drain (one heappop at a time) to preserve lowest-cost-first ordering.

                heapq.heappush(
                    queue,
                    QueueItem(
                        cost=new_cost,
                        node=Node(concept_id=object_id, is_standard=True),
                        candidate_index=item.candidate_index,
                        iterations=next_iterations,
                    ),
                )

    # list comprehension for flattening + deduplication, keeping the order found
    return [
        list(dict.fromkeys(concept for concepts in found.values() for concept in concepts))
        for found in found_standard_concepts
    ]


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
        kg: KnowledgeGraph,
        path: GraphPath,
        match_kind: LabelMatchKind,
        source_concept_id: int | None = None,
    ) -> PathProfile:
        """
        Analyze a path to determine the 'Standard Anchor'.

        The first Standard Concept encountered via an IDENTITY edge is promoted as
        the anchor.

        Notes
        -----
        For zero-hop paths (source == target), ``source_concept_id``
        must be provided; a ``ValueError`` is raised otherwise.

        Parameters
        ----------
        source_concept_id : int, optional
            Required when ``path`` has no steps (i.e. source == target).
        """
        if not path.steps:
            if source_concept_id is None:
                raise ValueError(
                    "source_concept_id is required for zero-hop paths "
                    "(find_shortest_paths was called with source == target)."
                )
            view = kg.concept_view(source_concept_id)
            return cls(
                concept_id=source_concept_id,
                concept_name=view.concept_name,
                is_standard=view.standard_concept,
                original_concept_id=source_concept_id,
                original_concept_name=view.concept_name,
                path=path,
            )

        node_ids = path.nodes()
        view_map = {v.concept_id: v for v in kg.concept_views(node_ids)}

        def get_view(idx):
            return view_map[node_ids[idx]]

        predicate_kinds = kg.predicate_kinds(tuple(p.predicate for p in path.steps))
        standard_anchor: tuple[int, str] | None = None

        for step_idx in range(len(path.steps)):
            next_view = get_view(step_idx + 1)
            if (
                predicate_kinds[step_idx] is PredicateKind.IDENTITY
                and not standard_anchor
                and next_view.standard_concept
            ):
                standard_anchor = (next_view.concept_id, next_view.concept_name)

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
    traversal_depth: int | None
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
        kg: KnowledgeGraph,
        path: GraphPath,
        trace: GraphTrace,
        match_kind: LabelMatchKind,
    ) -> PathExplanation:
        """
        Construct an explanation by combining the path, the trace log, and semantic profiles.
        """
        steps: list[PathExplanationStep] = []
        source = (
            path.steps[0].subject.concept_id
            if path.steps
            else (trace.seeds[0] if trace.seeds else None)
        )
        profile = PathProfile.from_path(kg, path, match_kind=match_kind, source_concept_id=source)

        for step in path.steps:
            ts = trace_contains_step(trace, step)
            kind = kg.predicate_kind(step.predicate)
            reason = kind.value
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
    """
    Check if a specific path step appears in the search trace.
    """
    for ts in trace.steps:
        if ts.node != step.subject.concept_id:
            continue
        for e in ts.expanded_edges:
            if e.object_id == step.object.concept_id and e.predicate_id == step.predicate:
                return ts
    return None
