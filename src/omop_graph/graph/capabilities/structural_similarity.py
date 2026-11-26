"""Structural similarity of concepts through the ``Is a`` hierarchy."""

from __future__ import annotations

import functools
import math
from collections.abc import Iterable

from omop_alchemy.cdm.query import ConceptFilter

from ...extensions.relationship_classification import HierarchyDirection
from ..model.edges import IS_A_FILTER, EdgeFilter
from ..model.scope import DEFAULT_CONCEPT_FILTER
from ..model.similarity import ConceptSimilarity, ICSource
from ..queries import q_concept_count, q_concept_num_descendants
from .hierarchy import HierarchyMixin


class StructuralSimilarityMixin(HierarchyMixin):
    """
    Common ancestors, information content and pairwise similarity of concepts.

    The ancestors are those of :meth:`closure`, the strict ``Is a`` hierarchy by default,
    and each concept counts as its own ancestor and descendant. A ``concept_filter`` narrows the
    concepts that are reported and counted, while the walk still passes through the others.
    """

    @functools.cached_property
    def _descendant_counts(self) -> dict[tuple[ICSource, ConceptFilter, int], int]:
        """The descendants counted for each concept, by source and scope, kept for this graph."""
        return {}

    @functools.cached_property
    def _scope_sizes(self) -> dict[ConceptFilter, int]:
        """The number of concepts each scope selects, kept for this graph."""
        return {}

    def _lineage(
        self,
        concept_ids: Iterable[int],
        direction: HierarchyDirection,
        edge_filter: EdgeFilter,
        concept_filter: ConceptFilter,
        standardise: bool = False,
    ) -> dict[int, frozenset[int]]:
        """Map each concept to itself and the concepts reached from it."""
        reached = self._closure(concept_ids, direction, edge_filter, concept_filter, standardise)
        return {concept_id: others | {concept_id} for concept_id, others in reached.items()}

    def _pair_lineage(
        self,
        concept_a: int,
        concept_b: int,
        direction: HierarchyDirection,
        edge_filter: EdgeFilter,
        concept_filter: ConceptFilter,
        standardise: bool = False,
    ) -> dict[int, frozenset[int]]:
        """Map two concepts, which must exist, each to itself and the concepts reached from it."""
        self.require_concepts((concept_a, concept_b))
        return self._lineage(
            (concept_a, concept_b), direction, edge_filter, concept_filter, standardise
        )

    def common_ancestors(
        self,
        concept_a: int,
        concept_b: int,
        edge_filter: EdgeFilter = IS_A_FILTER,
        concept_filter: ConceptFilter = DEFAULT_CONCEPT_FILTER,
        standardise: bool = False,
    ) -> frozenset[int]:
        """
        Return the concepts that are, or are above, both concepts.

        Parameters
        ----------
        concept_a : int
            The first concept.
        concept_b : int
            The second concept.
        edge_filter : EdgeFilter
            The edges that make up the hierarchy, ``Is a`` by default.
        concept_filter : ConceptFilter
            The concepts that are reported.
        standardise : bool
            If True, a non-standard concept stands in for the one standard concept it maps to.

        Returns
        -------
        frozenset[int]
            The concepts above or equal to both.

        Raises
        ------
        UnknownConceptError
            If either concept does not exist.
        """
        lineage = self._pair_lineage(
            concept_a, concept_b, HierarchyDirection.UP, edge_filter, concept_filter, standardise
        )
        return lineage[concept_a] & lineage[concept_b]

    def common_descendants(
        self,
        concept_a: int,
        concept_b: int,
        edge_filter: EdgeFilter = IS_A_FILTER,
        concept_filter: ConceptFilter = DEFAULT_CONCEPT_FILTER,
        standardise: bool = False,
    ) -> frozenset[int]:
        """
        Return the concepts that are, or are below, both concepts.

        Parameters
        ----------
        concept_a : int
            The first concept.
        concept_b : int
            The second concept.
        edge_filter : EdgeFilter
            The edges that make up the hierarchy, ``Is a`` by default.
        concept_filter : ConceptFilter
            The concepts that are reported.
        standardise : bool
            If True, a non-standard concept stands in for the one standard concept it maps to.

        Returns
        -------
        frozenset[int]
            The concepts below or equal to both.

        Raises
        ------
        UnknownConceptError
            If either concept does not exist.
        """
        lineage = self._pair_lineage(
            concept_a, concept_b, HierarchyDirection.DOWN, edge_filter, concept_filter, standardise
        )
        return lineage[concept_a] & lineage[concept_b]

    def most_recent_common_ancestors(
        self,
        concept_a: int,
        concept_b: int,
        edge_filter: EdgeFilter = IS_A_FILTER,
        concept_filter: ConceptFilter = DEFAULT_CONCEPT_FILTER,
        standardise: bool = False,
    ) -> frozenset[int]:
        """
        Return the common ancestors that have no descendant among the common ancestors.

        They are found from the ancestor sets alone, so no descendants are counted.

        Parameters
        ----------
        concept_a : int
            The first concept.
        concept_b : int
            The second concept.
        edge_filter : EdgeFilter
            The edges that make up the hierarchy, ``Is a`` by default.
        concept_filter : ConceptFilter
            The concepts that are reported.
        standardise : bool
            If True, a non-standard concept stands in for the one standard concept it maps to.

        Returns
        -------
        frozenset[int]
            The common ancestors with no descendant among them.

        Raises
        ------
        UnknownConceptError
            If either concept does not exist.
        """
        lineage = self._pair_lineage(
            concept_a, concept_b, HierarchyDirection.UP, edge_filter, concept_filter, standardise
        )
        return self._most_recent(
            lineage[concept_a] & lineage[concept_b], edge_filter, concept_filter
        )

    def _most_recent(
        self, common: frozenset[int], edge_filter: EdgeFilter, concept_filter: ConceptFilter
    ) -> frozenset[int]:
        """Drop the concepts of common that lie above another concept of it."""
        above = self._closure(common, HierarchyDirection.UP, edge_filter, concept_filter)
        return common - frozenset().union(*above.values())

    def jaccard_similarity(
        self,
        concept_a: int,
        concept_b: int,
        edge_filter: EdgeFilter = IS_A_FILTER,
        concept_filter: ConceptFilter = DEFAULT_CONCEPT_FILTER,
        standardise: bool = False,
    ) -> float:
        """
        Return the common ancestors over all ancestors of either concept.

        Parameters
        ----------
        concept_a : int
            The first concept.
        concept_b : int
            The second concept.
        edge_filter : EdgeFilter
            The edges that make up the hierarchy, ``Is a`` by default.
        concept_filter : ConceptFilter
            The concepts that are reported.
        standardise : bool
            If True, a non-standard concept stands in for the one standard concept it maps to.

        Returns
        -------
        float
            The share of the ancestors of either concept that both have.

        Raises
        ------
        UnknownConceptError
            If either concept does not exist.
        ValueError
            If ``standardise`` is False and a concept is outside ``concept_filter``, or if
            ``standardise`` is True and a concept maps to several standard concepts.
        """
        lineage = self._pair_lineage(
            concept_a, concept_b, HierarchyDirection.UP, edge_filter, concept_filter, standardise
        )
        if not standardise:
            self._require_in_scope((concept_a, concept_b), concept_filter)
        return len(lineage[concept_a] & lineage[concept_b]) / len(
            lineage[concept_a] | lineage[concept_b]
        )

    def information_content(
        self,
        concept_ids: Iterable[int],
        source: ICSource = ICSource.IS_A,
        concept_filter: ConceptFilter = DEFAULT_CONCEPT_FILTER,
    ) -> dict[int, float]:
        """
        Return the information content of each concept, ``log2(concepts / descendants)``.

        The descendants include the concept itself. Both the descendants and the concepts
        are those in ``concept_filter``, so the top of a hierarchy is near zero and a leaf
        is high. Counts are cached for the lifetime of the graph.

        Parameters
        ----------
        concept_ids : Iterable[int]
            Concepts to score.
        source : ICSource
            Where the descendants are counted. With ``CONCEPT_ANCESTOR`` the scope only sets
            the number of concepts.
        concept_filter : ConceptFilter
            The concepts that count, as descendants and in total.

        Returns
        -------
        dict[int, float]
            Every requested concept mapped to its information content, never below zero.

        Raises
        ------
        UnknownConceptError
            If a requested concept does not exist.
        ValueError
            If a requested concept is outside ``concept_filter``, if it selects no concept,
            or if it selects fewer concepts than the descendants counted for one concept,
            as when ``CONCEPT_ANCESTOR`` is used with a scope narrower than the table.
        """
        ids = tuple(dict.fromkeys(concept_ids))
        self.require_concepts(ids)
        return self._information_content(ids, source, concept_filter)

    def _information_content(
        self,
        ids: tuple[int, ...],
        source: ICSource,
        concept_filter: ConceptFilter,
    ) -> dict[int, float]:
        """Score concepts that exist, as :meth:`information_content` does."""
        total = self._scope_size(concept_filter)
        if not total:
            raise ValueError("The concept_filter selects no concept, so there is nothing to count.")
        self._require_in_scope(ids, concept_filter)
        counts = self._count_descendants(ids, source, concept_filter)
        if too_many := [concept_id for concept_id in ids if counts[concept_id] > total]:
            raise ValueError(
                f"The descendants counted for {too_many[:3]} outnumber the {total} concepts the "
                f"concept_filter selects. Widen the concept_filter"
                + (
                    "."
                    if source is ICSource.IS_A
                    else ", or count the descendants within it with ICSource.IS_A."
                )
            )
        return {concept_id: math.log2(total / counts[concept_id]) for concept_id in ids}

    def _scope_size(self, concept_filter: ConceptFilter) -> int:
        """The number of concepts ``concept_filter`` selects."""
        if concept_filter not in self._scope_sizes:
            with self.session_factory() as session:
                self._scope_sizes[concept_filter] = session.execute(
                    q_concept_count(concept_filter)
                ).scalar_one()
        return self._scope_sizes[concept_filter]

    def _count_descendants(
        self, concept_ids: tuple[int, ...], source: ICSource, concept_filter: ConceptFilter
    ) -> dict[int, int]:
        """Count each concept and its descendants, reading the ones not cached."""
        missing = tuple(
            concept_id
            for concept_id in concept_ids
            if (source, concept_filter, concept_id) not in self._descendant_counts
        )
        if missing and source is ICSource.IS_A:
            below = self._closure(missing, HierarchyDirection.DOWN, IS_A_FILTER, concept_filter)
            counts = {concept_id: len(others) + 1 for concept_id, others in below.items()}
        elif missing:
            counts = {concept_id: 1 for concept_id in missing}
            with self.session_factory() as session:
                for batch in self.batches(missing):
                    for row in session.execute(q_concept_num_descendants(batch)):
                        counts[row.concept_id] += row.num_descendants
        else:
            counts = {}
        for concept_id, count in counts.items():
            self._descendant_counts[(source, concept_filter, concept_id)] = count
        return {
            concept_id: self._descendant_counts[(source, concept_filter, concept_id)]
            for concept_id in concept_ids
        }

    def _require_in_scope(self, concept_ids: Iterable[int], concept_filter: ConceptFilter) -> None:
        """Raise when a concept lies outside concept_filter, since it has no score there."""
        if outside := sorted(set(concept_ids) - self._within(concept_ids, concept_filter)):
            raise ValueError(
                f"The concepts {outside} are outside the concept_filter, so they cannot be "
                f"scored. Ask for a concept the scope selects, or widen the scope."
            )

    def _scored_concepts(
        self, concept_ids: tuple[int, ...], concept_filter: ConceptFilter, standardise: bool
    ) -> dict[int, int]:
        """
        The concept whose information content scores each concept, the concept itself.

        With standardise, a concept outside concept_filter is scored by the one standard
        concept it maps to. One that maps to several raises, see :meth:`standard_concept`.
        """
        scored = {concept_id: concept_id for concept_id in concept_ids}
        if standardise:
            outside = tuple(set(concept_ids) - self._within(concept_ids, concept_filter))
            for concept_id, standard in self._standard_concept(outside).items():
                scored[concept_id] = concept_id if standard is None else standard
        return scored

    def pairwise_similarity(
        self,
        concept_a: int,
        concept_b: int,
        source: ICSource = ICSource.IS_A,
        concept_filter: ConceptFilter = DEFAULT_CONCEPT_FILTER,
        standardise: bool = False,
    ) -> ConceptSimilarity:
        """
        Score the similarity of two concepts through the strict ``Is a`` hierarchy.

        Only the two concepts and their most recent common ancestors have their
        descendants counted, so the cost does not grow with the number of common ancestors.

        Parameters
        ----------
        concept_a : int
            The first concept.
        concept_b : int
            The second concept.
        source : ICSource
            Where descendants are counted for the information content.
        concept_filter : ConceptFilter
            The concepts that are reported and counted.
        standardise : bool
            If True, a non-standard concept stands in for the one standard concept it maps
            to when its ancestors are found, see :meth:`closure`. A concept outside
            ``concept_filter`` has its information content read from that standard concept.

        Returns
        -------
        ConceptSimilarity
            The scores of the pair.

        Raises
        ------
        UnknownConceptError
            If either concept does not exist.
        ValueError
            If a concept is outside ``concept_filter`` and ``standardise`` is False, or if
            ``standardise`` is True and a concept maps to several standard concepts.
            Otherwise see :meth:`information_content`.
        """
        lineage = self._pair_lineage(
            concept_a, concept_b, HierarchyDirection.UP, IS_A_FILTER, concept_filter, standardise
        )
        common = lineage[concept_a] & lineage[concept_b]
        recent = self._most_recent(common, IS_A_FILTER, concept_filter)
        scored = self._scored_concepts((concept_a, concept_b), concept_filter, standardise)
        scores = self._information_content(
            tuple({*scored.values(), *recent}), source, concept_filter
        )
        best = max(sorted(recent), key=lambda concept_id: scores[concept_id], default=None)
        ancestor_ic = scores[best] if best is not None else 0.0
        jaccard = len(common) / len(lineage[concept_a] | lineage[concept_b])
        return ConceptSimilarity(
            concept_a=concept_a,
            concept_b=concept_b,
            common_ancestors=common,
            most_recent_common_ancestors=recent,
            jaccard=jaccard,
            ancestor_id=best,
            ancestor_information_content=ancestor_ic,
            information_content_a=scores[scored[concept_a]],
            information_content_b=scores[scored[concept_b]],
            phenodigm=math.sqrt(jaccard * ancestor_ic) if jaccard and ancestor_ic else 0.0,
        )
