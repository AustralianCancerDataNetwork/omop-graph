"""Structural similarity of concepts through the ``Is a`` hierarchy."""

from __future__ import annotations

import functools
import math
from collections.abc import Iterable

from omop_alchemy.cdm.query import ConceptFilter

from ...extensions.omop_alchemy import HierarchyDirection
from ..edges import IS_A_FILTER, EdgeFilter
from ..queries import DEFAULT_CONCEPT_FILTER, q_concept_count, q_concept_num_descendants
from ..scope import scope_only
from ..structural_similarity import ConceptSimilarity, ICSource
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

    @scope_only
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

        Raises
        ------
        UnknownConceptError
            If either concept does not exist.
        """
        lineage = self._pair_lineage(
            concept_a, concept_b, HierarchyDirection.UP, edge_filter, concept_filter, standardise
        )
        return lineage[concept_a] & lineage[concept_b]

    @scope_only
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

        Raises
        ------
        UnknownConceptError
            If either concept does not exist.
        """
        lineage = self._pair_lineage(
            concept_a, concept_b, HierarchyDirection.DOWN, edge_filter, concept_filter, standardise
        )
        return lineage[concept_a] & lineage[concept_b]

    @scope_only
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
        """Drop the concepts of *common* that lie above another concept of it."""
        above = self._closure(common, HierarchyDirection.UP, edge_filter, concept_filter)
        return common - frozenset().union(*above.values())

    @scope_only
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

        Raises
        ------
        UnknownConceptError
            If either concept does not exist.
        """
        lineage = self._pair_lineage(
            concept_a, concept_b, HierarchyDirection.UP, edge_filter, concept_filter, standardise
        )
        return len(lineage[concept_a] & lineage[concept_b]) / len(
            lineage[concept_a] | lineage[concept_b]
        )

    @scope_only
    def information_content(
        self,
        concept_ids: Iterable[int],
        source: ICSource = ICSource.IS_A,
        concept_filter: ConceptFilter = DEFAULT_CONCEPT_FILTER,
    ) -> dict[int, float]:
        """
        Return the information content of each concept, ``-log2(descendants / concepts)``.

        The descendants include the concept itself, and the concepts are all those in
        ``concept_filter``, so the top of a hierarchy is near zero and a leaf is high. Counts are
        cached for the lifetime of the graph.

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
            If ``concept_filter`` selects no concept, or selects fewer concepts than
            the descendants counted for one concept, as when ``CONCEPT_ANCESTOR`` is used
            with a scope narrower than the table.
        """
        ids = tuple(dict.fromkeys(concept_ids))
        self.require_concepts(ids)
        return self._information_content(ids, source, concept_filter)

    def _information_content(
        self, ids: tuple[int, ...], source: ICSource, concept_filter: ConceptFilter
    ) -> dict[int, float]:
        """Score concepts that exist, as :meth:`information_content` does."""
        counts = self._count_descendants(ids, source, concept_filter)
        total = self._scope_size(concept_filter)
        if not total:
            raise ValueError("The concept_filter selects no concept, so there is nothing to count.")
        if too_many := [concept_id for concept_id in ids if counts[concept_id] > total]:
            raise ValueError(
                f"The descendants counted for {too_many[:3]} outnumber the {total} concepts the "
                f"concept_filter selects. Count the descendants within the concept_filter, with "
                f"ICSource.IS_A, or widen the concept_filter."
            )
        return {concept_id: -math.log2(counts[concept_id] / total) for concept_id in ids}

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

    @scope_only
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
            If True, a non-standard concept stands in for the standard concepts it maps to
            when its ancestors are found, see :meth:`closure`. The information content is
            that of the concept itself.

        Raises
        ------
        UnknownConceptError
            If either concept does not exist.
        """
        lineage = self._pair_lineage(
            concept_a, concept_b, HierarchyDirection.UP, IS_A_FILTER, concept_filter, standardise
        )
        common = lineage[concept_a] & lineage[concept_b]
        recent = self._most_recent(common, IS_A_FILTER, concept_filter)
        scores = self._information_content(
            tuple({concept_a, concept_b, *recent}), source, concept_filter
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
            information_content_a=scores[concept_a],
            information_content_b=scores[concept_b],
            phenodigm=math.sqrt(jaccard * ancestor_ic) if jaccard and ancestor_ic else 0.0,
        )
