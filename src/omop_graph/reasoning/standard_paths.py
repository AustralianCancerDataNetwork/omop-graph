"""Search from candidate concepts to the standard concepts they stand for."""

from __future__ import annotations

import heapq
import logging
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from omop_graph.graph.model.edges import EdgeFilter
from omop_graph.graph.model.matches import LabelMatchKind
from omop_graph.graph.model.nodes import AncestorMatch, ConceptView
from omop_graph.graph.paths import DEFAULT_PATH_FILTER, Node
from omop_graph.reasoning.resolvers import CandidateHit

if TYPE_CHECKING:
    from omop_graph.graph.kg import KnowledgeGraph

logger = logging.getLogger(__name__)


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
