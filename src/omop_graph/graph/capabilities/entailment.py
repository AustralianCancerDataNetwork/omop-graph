"""Relationships entailed through the hierarchy."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Generator
from dataclasses import replace

from omop_alchemy.cdm.query import ConceptFilter

# Local Application Imports
from ...extensions.omop_alchemy import (
    HierarchyDirection,
)
from ..edges import (
    IS_A_RELATIONSHIP_ID,
    EdgeFilter,
)
from .hierarchy import HierarchyMixin


class EntailmentMixin(HierarchyMixin):
    """Relationships entailed through the hierarchy."""

    def entailed_relationships(
        self,
        subjects: tuple[int, ...] | None,
        objects: tuple[int, ...] | None,
        edge_filter: EdgeFilter | None = None,
        concept_filter: ConceptFilter | None = None,
    ) -> Generator[tuple[int, str, int], None, None]:
        """
        Query relationships, including those that follow through the ``Is a`` hierarchy.

        Each triple is yielded once. Every relationship *edge_filter* selects is treated
        this way, so the caller selects only those for which that is sound, see
        :attr:`omop_graph.graph.edges.Predicate.generalises`.

        Parameters
        ----------
        subjects : tuple[int, ...] | None
            Subject concept ids.
        objects : tuple[int, ...] | None
            Object concept ids. At least one of *subjects* and *objects* is required.
        edge_filter : EdgeFilter, optional
            Which asserted relationships to start from, as in :meth:`relationships`.
        concept_filter : ConceptFilter, optional
            Which concepts count as entailed objects, and for incoming relationships as
            entailed subjects, as in :meth:`closure`. Asserted edges are unaffected.

        Yields
        ------
        tuple[int, str, int]
            Triples of (subject_concept_id, relationship_id, object_concept_id).

        Raises
        ------
        ValueError
            If both *subjects* and *objects* are None.

        Notes
        -----
        An asserted (T, R, D) also holds for every subtype of T, so the subject moves
        down, and for every ancestor of D, so the object moves up. For example, "Lung
        cancer Has finding site Lung" also gives "Small cell lung cancer Has finding site
        Lung" and "Lung cancer Has finding site Thoracic organ".
        """
        if subjects is None and objects is None:
            raise ValueError("Entailed relationships need subjects or objects")
        if subjects is not None:
            triples = self._entailed_from_subjects(subjects, objects, edge_filter, concept_filter)
        else:
            assert objects is not None
            triples = self._entailed_into_objects(objects, edge_filter, concept_filter)
        seen: set[tuple[int, str, int]] = set()
        for triple in triples:
            if triple not in seen:
                seen.add(triple)
                yield triple

    @staticmethod
    def _without_is_a(edge_filter: EdgeFilter | None) -> EdgeFilter | None:
        """
        *edge_filter* without the ``Is a`` relationship, when it names its relationships.

        Inheriting ``Is a`` edges between the subtypes of a subject adds nothing to the
        ancestors the closure already gives.
        """
        if edge_filter is None or edge_filter.predicate_ids is None:
            return edge_filter
        return replace(
            edge_filter, predicate_ids=edge_filter.predicate_ids - {IS_A_RELATIONSHIP_ID}
        )

    def _entailed_from_subjects(
        self,
        subjects: tuple[int, ...],
        objects: tuple[int, ...] | None,
        edge_filter: EdgeFilter | None,
        concept_filter: ConceptFilter | None,
    ) -> Generator[tuple[int, str, int], None, None]:
        """
        Find the asserted edges of *subjects* and of their ``Is a`` ancestors, and
        generalise the filler of each up the ``Is a`` hierarchy.
        """
        wanted = None if objects is None else frozenset(objects)
        above = self.closure(subjects)
        own: dict[int, list[tuple[str, int]]] = defaultdict(list)
        inherited: dict[int, list[tuple[str, int]]] = defaultdict(list)
        for edges, ids, selected in (
            (own, tuple(dict.fromkeys(subjects)), edge_filter),
            (inherited, tuple(set().union(*above.values())), self._without_is_a(edge_filter)),
        ):
            for batch in self.batches(ids):
                for source, relationship_id, filler in self.relationships(
                    batch, edge_filter=selected
                ):
                    edges[source].append((relationship_id, filler))
        ancestors = self.closure(
            {filler for edges in (*own.values(), *inherited.values()) for _, filler in edges},
            concept_filter=concept_filter,
        )
        for subject_id in subjects:
            held = own[subject_id] + [
                edge for source in above[subject_id] for edge in inherited[source]
            ]
            for relationship_id, filler in held:
                for reached in (filler, *ancestors[filler]):
                    if wanted is None or reached in wanted:
                        yield subject_id, relationship_id, reached

    def _entailed_into_objects(
        self,
        objects: tuple[int, ...],
        edge_filter: EdgeFilter | None,
        concept_filter: ConceptFilter | None,
    ) -> Generator[tuple[int, str, int], None, None]:
        """
        Find the edges into *objects* and into their ``Is a`` descendants, and give each
        to the ``Is a`` descendants of its subject.
        """
        descendants = self.closure(objects, HierarchyDirection.DOWN, concept_filter=concept_filter)
        targets_by_filler: dict[int, list[int]] = defaultdict(list)
        for target, below in descendants.items():
            for filler in (target, *below):
                targets_by_filler[filler].append(target)
        asserted: list[tuple[int, str, int]] = []
        for batch in self.batches(targets_by_filler):
            asserted.extend(self.relationships(None, objects=batch, edge_filter=edge_filter))
        inheriting = {
            subject_id
            for subject_id, relationship_id, _ in asserted
            if relationship_id != IS_A_RELATIONSHIP_ID
        }
        subtypes = self.closure(inheriting, HierarchyDirection.DOWN, concept_filter=concept_filter)
        for subject_id, relationship_id, filler in asserted:
            holders = (subject_id,)
            if relationship_id != IS_A_RELATIONSHIP_ID:
                holders = (subject_id, *subtypes[subject_id])
            for target in targets_by_filler[filler]:
                for holder in holders:
                    yield holder, relationship_id, target
