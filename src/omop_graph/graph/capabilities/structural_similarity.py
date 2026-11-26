"""Structural similarity of concepts through the ``Is a`` hierarchy."""

from __future__ import annotations

import math
from collections.abc import Iterable

from omop_alchemy.cdm.query import ConceptFilter

from ...extensions.omop_alchemy import HierarchyDirection
from ..edges import IS_A_FILTER, EdgeFilter
from ..queries import DEFAULT_CONCEPT_FILTER, q_concept_count, q_concept_num_descendants
from ..structural_similarity import ConceptSimilarity, ICSource
from .hierarchy import HierarchyMixin


class StructuralSimilarityMixin(HierarchyMixin):
    """
    Common ancestors, information content and pairwise similarity of concepts.

    The ancestors are those of :meth:`closure`, the strict ``Is a`` hierarchy by default,
    and each concept counts as its own ancestor and descendant. A *scope* narrows the
    concepts that are reported and counted, while the walk still passes through the others.
    """

    _descendant_counts: dict[tuple[ICSource, ConceptFilter, int], int]
    _scope_sizes: dict[ConceptFilter, int]

    def _lineage(
        self,
        concept_ids: Iterable[int],
        direction: HierarchyDirection,
        edge_filter: EdgeFilter,
        scope: ConceptFilter,
    ) -> dict[int, frozenset[int]]:
        """Map each concept to itself and the concepts reached from it."""
        reached = self.closure(concept_ids, direction, edge_filter, scope)
        return {concept_id: others | {concept_id} for concept_id, others in reached.items()}

    def common_ancestors(
        self,
        concept_a: int,
        concept_b: int,
        edge_filter: EdgeFilter = IS_A_FILTER,
        scope: ConceptFilter = DEFAULT_CONCEPT_FILTER,
    ) -> frozenset[int]:
        """Return the concepts that are, or are above, both concepts."""
        lineage = self._lineage((concept_a, concept_b), HierarchyDirection.UP, edge_filter, scope)
        return lineage[concept_a] & lineage[concept_b]

    def common_descendants(
        self,
        concept_a: int,
        concept_b: int,
        edge_filter: EdgeFilter = IS_A_FILTER,
        scope: ConceptFilter = DEFAULT_CONCEPT_FILTER,
    ) -> frozenset[int]:
        """Return the concepts that are, or are below, both concepts."""
        lineage = self._lineage((concept_a, concept_b), HierarchyDirection.DOWN, edge_filter, scope)
        return lineage[concept_a] & lineage[concept_b]

    def most_recent_common_ancestors(
        self,
        concept_a: int,
        concept_b: int,
        edge_filter: EdgeFilter = IS_A_FILTER,
        scope: ConceptFilter = DEFAULT_CONCEPT_FILTER,
    ) -> frozenset[int]:
        """
        Return the common ancestors that have no descendant among the common ancestors.

        They are found from the ancestor sets alone, so no descendants are counted.
        """
        return self._most_recent(
            self.common_ancestors(concept_a, concept_b, edge_filter, scope), edge_filter, scope
        )

    def _most_recent(
        self, common: frozenset[int], edge_filter: EdgeFilter, scope: ConceptFilter
    ) -> frozenset[int]:
        """Drop the concepts of *common* that lie above another concept of it."""
        above = self.closure(common, HierarchyDirection.UP, edge_filter, scope)
        return common - frozenset().union(*above.values())

    def jaccard_similarity(
        self,
        concept_a: int,
        concept_b: int,
        edge_filter: EdgeFilter = IS_A_FILTER,
        scope: ConceptFilter = DEFAULT_CONCEPT_FILTER,
    ) -> float:
        """Return the common ancestors over all ancestors of either concept."""
        lineage = self._lineage((concept_a, concept_b), HierarchyDirection.UP, edge_filter, scope)
        return len(lineage[concept_a] & lineage[concept_b]) / len(
            lineage[concept_a] | lineage[concept_b]
        )

    def information_content(
        self,
        concept_ids: Iterable[int],
        source: ICSource = ICSource.IS_A,
        scope: ConceptFilter = DEFAULT_CONCEPT_FILTER,
    ) -> dict[int, float]:
        """
        Return the information content of each concept, ``-log2(descendants / concepts)``.

        The descendants include the concept itself, and the concepts are all those in
        *scope*, so the top of a hierarchy is near zero and a leaf is high. Counts are
        cached for the lifetime of the graph.

        Parameters
        ----------
        concept_ids : Iterable[int]
            Concepts to score.
        source : ICSource
            Where the descendants are counted. With ``CONCEPT_ANCESTOR`` the scope only sets
            the number of concepts.
        scope : ConceptFilter
            The concepts that count, as descendants and in total.

        Returns
        -------
        dict[int, float]
            Every requested concept mapped to its information content, never below zero.
        """
        ids = tuple(dict.fromkeys(concept_ids))
        counts = self._count_descendants(ids, source, scope)
        total = self._scope_size(scope)
        return {
            concept_id: max(0.0, -math.log2(counts[concept_id] / total)) if total else 0.0
            for concept_id in ids
        }

    def _scope_size(self, scope: ConceptFilter) -> int:
        """The number of concepts *scope* selects."""
        if scope not in self._scope_sizes:
            with self.session_factory() as session:
                self._scope_sizes[scope] = session.execute(q_concept_count(scope)).scalar_one()
        return self._scope_sizes[scope]

    def _count_descendants(
        self, concept_ids: tuple[int, ...], source: ICSource, scope: ConceptFilter
    ) -> dict[int, int]:
        """Count each concept and its descendants, reading the ones not cached."""
        missing = tuple(
            concept_id
            for concept_id in concept_ids
            if (source, scope, concept_id) not in self._descendant_counts
        )
        if missing and source is ICSource.IS_A:
            below = self.closure(missing, HierarchyDirection.DOWN, IS_A_FILTER, scope)
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
            self._descendant_counts[(source, scope, concept_id)] = count
        return {
            concept_id: self._descendant_counts[(source, scope, concept_id)]
            for concept_id in concept_ids
        }

    def pairwise_similarity(
        self,
        concept_a: int,
        concept_b: int,
        source: ICSource = ICSource.IS_A,
        scope: ConceptFilter = DEFAULT_CONCEPT_FILTER,
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
        scope : ConceptFilter
            The concepts that are reported and counted.
        """
        lineage = self._lineage((concept_a, concept_b), HierarchyDirection.UP, IS_A_FILTER, scope)
        common = lineage[concept_a] & lineage[concept_b]
        recent = self._most_recent(common, IS_A_FILTER, scope)
        scores = self.information_content({concept_a, concept_b, *recent}, source, scope)
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
