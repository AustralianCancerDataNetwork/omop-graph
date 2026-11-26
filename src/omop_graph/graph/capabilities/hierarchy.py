"""The ``Is a`` hierarchy: parents, children, closure and ancestry."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import replace

from omop_alchemy.cdm.query import ConceptFilter
from sqlalchemy import Row

from ...extensions.relationship_classification import (
    HierarchyDirection,
)
from ..model.edges import (
    IDENTITY_FILTER,
    IS_A_FILTER,
    EdgeFilter,
)
from ..model.nodes import (
    AncestorMatch,
)

# Local Application Imports
from ..model.scope import restricted_to_ids
from ..queries import (
    q_children,
    q_closure,
    q_concept_num_ancestors,
    q_concept_potential_ancestor,
    q_concept_potential_ancestors_batch,
    q_entities,
    q_parents,
)
from .concepts import ConceptsMixin
from .edges import EdgesMixin


class HierarchyMixin(EdgesMixin, ConceptsMixin):
    """The ``Is a`` hierarchy: parents, children, closure and ancestry."""

    def specificity(self, concept_id: int) -> float:
        """
        Compute specificity as the inverse of out-degree.
        Higher is more specific.

        Parameters
        ----------
        concept_id : int
            The concept.

        Returns
        -------
        float
            One over the number of outgoing edges, and 1.0 without any.
        """
        out_edges = self.edges(direction="out", concept_ids=concept_id)
        if not out_edges:
            return 1.0
        return 1.0 / len(out_edges)

    def parents(self, concept_id: int, isa_only: bool = False) -> tuple[int, ...]:
        """
        Retrieve the parent concept ids of a concept.

        Parameters
        ----------
        concept_id : int
            The concept.
        isa_only : bool
            If True, the parents are the targets of its active ``Is a`` edges. Otherwise
            they are its level 1 ancestors in the ``concept_ancestor`` table, which hold
            classification and cohort ancestry besides ``Is a``.

        Returns
        -------
        tuple[int, ...]
            The parent concept ids.
        """
        if isa_only:
            return tuple(
                edge.object_id
                for edge in self.edges(concept_id, direction="out", edge_filter=IS_A_FILTER)
            )
        with self.session_factory() as session:
            parents = tuple(session.execute(q_parents(concept_id)).scalars())
        return parents

    def children(self, concept_id) -> tuple[int, ...]:
        """
        Retrieve children Concept IDs of concept using Concept_Ancestor table.

        Parameters
        ----------
        concept_id : int
            The concept.

        Returns
        -------
        tuple[int, ...]
            The child concept ids.
        """
        with self.session_factory() as session:
            children = tuple(session.execute(q_children(concept_id)).scalars())
        return children

    def closure(
        self,
        concept_ids: Iterable[int],
        direction: HierarchyDirection = HierarchyDirection.UP,
        edge_filter: EdgeFilter = IS_A_FILTER,
        concept_filter: ConceptFilter | None = None,
        standardise: bool = False,
    ) -> dict[int, frozenset[int]]:
        """
        Find the concepts reached by following edges repeatedly from each concept.

        By default this is the strict ``Is a`` ancestors, which unlike :meth:`parents`
        (``concept_ancestor``) holds no cohort, therapeutic or composition ancestry. Other
        edge filters give the closure over other relationships, which can be very large.

        Parameters
        ----------
        concept_ids : Iterable[int]
            Concepts to start from.
        direction : HierarchyDirection
            ``UP`` follows an edge from its subject to its object, as to ancestors, and
            ``DOWN`` from its object to its subject, as to descendants.
        edge_filter : EdgeFilter
            The edges to follow. Relationships without a classification are not followed.
        concept_filter : ConceptFilter, optional
            Report only the concepts it selects. The walk still passes through the others,
            so a concept above an excluded one is still reached. It is a scope, so it
            must not set ``limit``.
        standardise : bool
            If True, a non-standard concept stands in for the one standard concept it maps
            to, see :meth:`standard_concept`. It reaches that concept and everything reached
            from it, whether or not concept_filter selects the standard concept itself. A
            concept without a standard concept reaches nothing.

        Returns
        -------
        dict[int, frozenset[int]]
            Every requested concept mapped to the concepts reached from it, never itself.

        Raises
        ------
        UnknownConceptError
            If a requested concept does not exist.
        ValueError
            If ``standardise`` is True and a concept maps to several standard concepts.
        """
        starts = tuple(dict.fromkeys(concept_ids))
        self.require_concepts(starts)
        return self._closure(starts, direction, edge_filter, concept_filter, standardise)

    def _closure(
        self,
        concept_ids: Iterable[int],
        direction: HierarchyDirection = HierarchyDirection.UP,
        edge_filter: EdgeFilter = IS_A_FILTER,
        concept_filter: ConceptFilter | None = None,
        standardise: bool = False,
    ) -> dict[int, frozenset[int]]:
        """Walk as :meth:`closure` does, without checking that the concepts exist."""
        starts = tuple(dict.fromkeys(concept_ids))
        if standardise:
            stands_in = self._stands_in(starts)
            walked = self._closure(
                {stand_in for group in stands_in.values() for stand_in in group},
                direction,
                edge_filter,
                concept_filter,
            )
            return {
                concept_id: frozenset(
                    set().union(
                        *(walked[stand_in] for stand_in in group), set(group) - {concept_id}
                    )
                )
                for concept_id, group in stands_in.items()
            }
        reached: dict[int, set[int]] = {concept_id: set() for concept_id in starts}
        resolved = self._resolve_edge_filter(edge_filter)
        if resolved is not None:
            followed = frozenset(self._relationship_mapping)
            if resolved.predicate_ids is not None:
                followed &= resolved.predicate_ids
            resolved = replace(resolved, predicate_ids=followed)
        if resolved is not None and resolved.predicate_ids:
            with self.session_factory() as session:
                for batch in self.batches(starts):
                    for origin, other in session.execute(q_closure(batch, direction, resolved)):
                        reached[origin].add(other)
        if concept_filter is not None:
            in_scope = self._within(set().union(*reached.values()), concept_filter)
            reached = {concept_id: others & in_scope for concept_id, others in reached.items()}
        return {concept_id: frozenset(others) for concept_id, others in reached.items()}

    def standardise(
        self, concept_ids: Iterable[int], edge_filter: EdgeFilter = IDENTITY_FILTER
    ) -> dict[int, tuple[int, ...]]:
        """
        Find the standard concepts each concept stands for.

        A standard concept stands for itself. Any other concept stands for the standard
        concepts at the end of its identity edges, such as ``Maps to`` and ``Concept
        replaced by``, which may be several, and which this lists all of. Only a standard concept is a valid mapping
        target, so an edge to any other concept is not followed. A classification concept
        stands for the standard concept it maps to, as an ATC class stands for its
        ingredient, and for none when it has no mapping.

        Parameters
        ----------
        concept_ids : Iterable[int]
            Concepts to standardise.
        edge_filter : EdgeFilter
            The edges that map a concept to its standard concepts.

        Returns
        -------
        dict[int, tuple[int, ...]]
            Every requested concept mapped to its standard concepts in ascending order,
            empty when it has none.

        Raises
        ------
        UnknownConceptError
            If a requested concept does not exist.
        """
        ids = tuple(dict.fromkeys(concept_ids))
        self.require_concepts(ids)
        return self._standardise(ids, edge_filter)

    def _standardise(
        self, ids: tuple[int, ...], edge_filter: EdgeFilter = IDENTITY_FILTER
    ) -> dict[int, tuple[int, ...]]:
        """Standardise as :meth:`standardise` does, without checking that the concepts exist."""
        standard = self._standard_among(ids)
        mapped: dict[int, set[int]] = {concept_id: set() for concept_id in ids}
        sources = tuple(concept_id for concept_id in ids if concept_id not in standard)
        for subject_id, _, object_id in self.relationships(sources, edge_filter=edge_filter):
            mapped[subject_id].add(object_id)
        standard_targets = self._standard_among(set().union(*mapped.values()))
        return {
            concept_id: (
                (concept_id,)
                if concept_id in standard
                else tuple(sorted(mapped[concept_id] & standard_targets))
            )
            for concept_id in ids
        }

    def standard_concept(
        self, concept_ids: Iterable[int], edge_filter: EdgeFilter = IDENTITY_FILTER
    ) -> dict[int, int | None]:
        """
        Find the one standard concept each concept stands for.

        Where :meth:`standardise` lists every standard concept, this is for callers that
        need a single one, which none of several can be.

        Parameters
        ----------
        concept_ids : Iterable[int]
            Concepts to standardise.
        edge_filter : EdgeFilter
            The edges that map a concept to its standard concepts.

        Returns
        -------
        dict[int, int | None]
            Every requested concept mapped to its standard concept, which is itself when it
            is standard, and None when it has none.

        Raises
        ------
        UnknownConceptError
            If a requested concept does not exist.
        ValueError
            If a concept maps to several standard concepts, naming them. Pass the standard
            concept you mean.
        """
        ids = tuple(dict.fromkeys(concept_ids))
        self.require_concepts(ids)
        return self._standard_concept(ids, edge_filter)

    def _standard_concept(
        self, ids: tuple[int, ...], edge_filter: EdgeFilter = IDENTITY_FILTER
    ) -> dict[int, int | None]:
        """Find the standard concepts as :meth:`standard_concept` does, for concepts that exist."""
        targets = self._standardise(ids, edge_filter)
        if several := {
            concept_id: found for concept_id, found in targets.items() if len(found) > 1
        }:
            raise ValueError(
                f"Each of these concepts maps to several standard concepts, so none can stand "
                f"for it: {several}. Pass the standard concept you mean."
            )
        return {concept_id: found[0] if found else None for concept_id, found in targets.items()}

    def _stands_in(self, concept_ids: Iterable[int]) -> dict[int, tuple[int, ...]]:
        """Map each concept to the concepts it stands in for: itself, then its one standard concept."""
        return {
            concept_id: (concept_id,) if standard in (None, concept_id) else (concept_id, standard)
            for concept_id, standard in self._standard_concept(
                tuple(dict.fromkeys(concept_ids))
            ).items()
        }

    def _standard_among(self, concept_ids: Iterable[int]) -> set[int]:
        """Select the standard concepts among concept_ids, the valid mapping targets."""
        return self._within(concept_ids, ConceptFilter(require_standard=True))

    def _within(self, concept_ids: Iterable[int], concept_filter: ConceptFilter) -> set[int]:
        """Select the concepts among concept_ids that concept_filter selects."""
        found: set[int] = set()
        with self.session_factory() as session:
            for batch in self.batches(set(concept_ids)):
                stmt = q_entities(restricted_to_ids(concept_filter, batch))
                found.update(session.execute(stmt).scalars())
        return found

    def get_potential_ancestor(self, child_id: int, parent_id: int) -> AncestorMatch | None:
        """
        Check if an ancestry relationship exists between a child and parent.

        Parameters
        ----------
        child_id : int
            The descendant.
        parent_id : int
            The ancestor.

        Returns
        -------
        AncestorMatch | None
            The ancestry row, or None when the parent is not an ancestor.
        """

        with self.session_factory() as session:
            row = session.execute(q_concept_potential_ancestor(child_id, parent_id)).first()

        if row is None:
            return None
        return AncestorMatch(
            ancestor_concept_id=row.ancestor_concept_id,
            descendant_concept_id=row.descendant_concept_id,
            min_levels_of_separation=row.min_levels_of_separation,
        )

    def get_potential_ancestors_batch(
        self,
        child_ids: tuple[int, ...],
        parent_ids: tuple[int, ...],
    ) -> dict[int, dict[int, AncestorMatch]]:
        """Check which candidate parents are ancestors of one or more children.

        Parameters
        ----------
        child_ids : tuple of int
            A tuple of descendant concept IDs for batch mode.
        parent_ids : tuple of int
            Candidate ancestor concept IDs to check.

        Returns
        -------
        Dict[int, Dict[int, AncestorMatch]]
            When child_id is tuple: maps child_id -> {ancestor_concept_id -> AncestorMatch}.
        """
        if not parent_ids:
            return {}

        rows: list[Row] = []
        with self.session_factory() as session:
            for children, parents in self._pair_batches(child_ids, parent_ids):
                rows.extend(session.execute(q_concept_potential_ancestors_batch(children, parents)))

        result: dict[int, dict[int, AncestorMatch]] = defaultdict(dict)
        for row in rows:
            result[row.descendant_concept_id][row.ancestor_concept_id] = AncestorMatch(
                ancestor_concept_id=row.ancestor_concept_id,
                descendant_concept_id=row.descendant_concept_id,
                min_levels_of_separation=row.min_levels_of_separation,
            )
        return result

    def get_num_ancestors(self, concept_ids: tuple[int, ...]) -> dict[int, int]:
        """
        Get the count of ancestors for a batch of concepts.

        Parameters
        ----------
        concept_ids : tuple[int, ...]
            The concepts to count for.

        Returns
        -------
        dict[int, int]
            The number of ancestors of each concept that has any.
        """
        with self.session_factory() as session:
            rows = list(self._rows_in_batches(session, q_concept_num_ancestors, concept_ids))
        return {row.concept_id: row.num_ancestors for row in rows}
