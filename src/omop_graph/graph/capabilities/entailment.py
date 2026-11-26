"""Relationships entailed through the hierarchy."""

from __future__ import annotations

import functools
import logging
from collections import defaultdict
from collections.abc import Generator, Iterable
from dataclasses import replace

from omop_alchemy.cdm.query import ConceptFilter

from ...extensions.omop_alchemy import (
    HierarchyDirection,
)
from ..model.edges import (
    DEFAULT_EDGE_FILTER,
    IS_A_RELATIONSHIP_ID,
    EdgeFilter,
)
from ..model.scope import DEFAULT_CONCEPT_FILTER, _require_scope

# Local Application Imports
from .enumeration import EnumerationMixin
from .hierarchy import HierarchyMixin

logger = logging.getLogger(__name__)


class EntailmentMixin(HierarchyMixin, EnumerationMixin):
    """Relationships entailed through the hierarchy."""

    @functools.cached_property
    def entailing_relationship_ids(self) -> frozenset[str]:
        """The classified relationships whose fillers generalise up the ``Is a`` hierarchy."""
        return frozenset(
            relationship_id
            for relationship_id, mapping in self._relationship_mapping.items()
            if mapping.generalises
        )

    def entailed_relationships(
        self,
        subjects: tuple[int, ...] | None,
        objects: tuple[int, ...] | None,
        edge_filter: EdgeFilter | None = DEFAULT_EDGE_FILTER,
        concept_filter: ConceptFilter | None = None,
        standardise: bool = False,
    ) -> Generator[tuple[int, str, int], None, None]:
        """
        Query relationships, including those that follow through the ``Is a`` hierarchy.

        Of the relationships edge_filter selects, those that generalise, see
        :attr:`omop_graph.graph.model.edges.Predicate.generalises`, are entailed. The others are
        yielded as stored, and first. Each entailed triple is yielded once.

        Parameters
        ----------
        subjects : tuple[int, ...] | None
            Subject concept ids.
        objects : tuple[int, ...] | None
            Object concept ids. At least one of subjects and objects is required.
        edge_filter : EdgeFilter, optional
            Which asserted relationships to start from, as in :meth:`relationships`.
        concept_filter : ConceptFilter, optional
            Which concepts count as entailed objects, and for incoming relationships as
            entailed subjects, as in :meth:`closure`. Asserted edges are unaffected. A
            requested object it excludes only gets the relationships asserted to it, and a
            warning lists such objects.
        standardise : bool
            If True, a concept also has the entailed relationships of the standard concepts
            it maps to, see :meth:`standardise`, reported for the concept itself. Stored
            relationships are not lent in this way.

        Yields
        ------
        tuple[int, str, int]
            Triples of (subject_concept_id, relationship_id, object_concept_id).

        Raises
        ------
        ValueError
            If both subjects and objects are None.

        Notes
        -----
        An asserted (T, R, D) also holds for every subtype of T, so the subject moves
        down, and for every ancestor of D, so the object moves up. For example, "Lung
        cancer Has finding site Lung" also gives "Small cell lung cancer Has finding site
        Lung" and "Lung cancer Has finding site Thoracic organ".
        """
        if subjects is None and objects is None:
            raise ValueError("Entailed relationships need subjects or objects")
        if concept_filter is not None:
            _require_scope(concept_filter)
            if objects is not None and not standardise:
                self._warn_outside_scope(objects, concept_filter)
        resolved = self._resolve_edge_filter(edge_filter)
        if resolved is None:
            return
        selected = frozenset(self._relationship_mapping)
        if resolved.predicate_ids is not None:
            selected &= resolved.predicate_ids
        entailing = selected & self.entailing_relationship_ids
        asserted = selected - entailing
        if asserted:
            yield from self.relationships(
                subjects, objects=objects, edge_filter=replace(resolved, predicate_ids=asserted)
            )
        if entailing:
            yield from self._entailed_with_stand_ins(
                subjects,
                objects,
                replace(resolved, predicate_ids=entailing),
                concept_filter,
                standardise,
            )

    def _warn_outside_scope(self, objects: tuple[int, ...], concept_filter: ConceptFilter) -> None:
        """Log the requested objects that concept_filter excludes, which are not entailed."""
        outside = sorted(set(objects) - self._within(objects, concept_filter))
        if outside:
            logger.warning(
                "The concepts %s are outside the concept_filter, so no relationship is "
                "entailed to them. Only the relationships asserted to them are reported.",
                outside,
            )

    def gap_filled_relationships(
        self, concept_ids: Iterable[int], edge_filter: EdgeFilter | None = DEFAULT_EDGE_FILTER
    ) -> tuple[tuple[int, str, int], ...]:
        """
        Find the non-redundant relationships that connect a set of concepts.

        The entailed relationships between the concepts are taken, and for each relationship
        a triple is dropped when another concept of the set lies between its ends by the
        same relationship. This may leave edges that are formally redundant.

        Parameters
        ----------
        concept_ids : Iterable[int]
            The concepts to connect.
        edge_filter : EdgeFilter, optional
            Which relationships to consider, as in :meth:`entailed_relationships`.

        Returns
        -------
        tuple[tuple[int, str, int], ...]
            Triples of (subject_concept_id, relationship_id, object_concept_id), sorted.
        """
        ids = tuple(dict.fromkeys(concept_ids))
        triples = {
            (subject_id, relationship_id, object_id)
            for subject_id, relationship_id, object_id in self.entailed_relationships(
                ids, ids, edge_filter
            )
            if subject_id != object_id
        }
        above: dict[tuple[str, int], set[int]] = defaultdict(set)
        for subject_id, relationship_id, object_id in triples:
            above[relationship_id, subject_id].add(object_id)
        return tuple(
            sorted(
                (subject_id, relationship_id, object_id)
                for subject_id, relationship_id, object_id in triples
                if not any(
                    object_id in above[relationship_id, between]
                    for between in above[relationship_id, subject_id]
                )
            )
        )

    def all_entailed_relationships(
        self,
        edge_filter: EdgeFilter | None = DEFAULT_EDGE_FILTER,
        concept_filter: ConceptFilter = DEFAULT_CONCEPT_FILTER,
        standardise: bool = False,
    ) -> Generator[tuple[int, str, int], None, None]:
        """
        Yield the entailed relationships of every concept concept_filter selects.

        The concepts are read in batches of subjects, so the table is never held in memory.
        The other parameters are those of :meth:`entailed_relationships`.

        Parameters
        ----------
        edge_filter : EdgeFilter | None
            Which relationships to entail.
        concept_filter : ConceptFilter
            The concepts whose relationships are entailed, and the scope of the objects.
        standardise : bool
            If True, a non-standard concept stands in for its standard concept.

        Yields
        ------
        tuple[int, str, int]
            The subject, relationship id and object of each relationship.
        """
        for batch in self.batches(self.entities(concept_filter)):
            yield from self.entailed_relationships(
                batch, None, edge_filter, concept_filter, standardise
            )

    def _entailed_with_stand_ins(
        self,
        subjects: tuple[int, ...] | None,
        objects: tuple[int, ...] | None,
        edge_filter: EdgeFilter,
        concept_filter: ConceptFilter | None,
        standardise: bool,
    ) -> Generator[tuple[int, str, int], None, None]:
        """Entail along the hierarchy, for the stand-ins of each concept when standardise."""
        if not standardise:
            yield from self._entailed_along_is_a(subjects, objects, edge_filter, concept_filter)
            return
        by_subject = subjects is not None
        requested = subjects if subjects is not None else objects
        assert requested is not None
        stands_in = self._stands_in(requested)
        origins: dict[int, list[int]] = defaultdict(list)
        for origin, group in stands_in.items():
            for stand_in in group:
                origins[stand_in].append(origin)
        expanded = tuple(origins)
        seen: set[tuple[int, str, int]] = set()
        if by_subject:
            triples = self._entailed_along_is_a(expanded, objects, edge_filter, concept_filter)
            reported = ((origin, p, o) for s, p, o in triples for origin in origins[s])
        else:
            triples = self._entailed_along_is_a(None, expanded, edge_filter, concept_filter)
            reported = ((s, p, origin) for s, p, o in triples for origin in origins[o])
        for triple in reported:
            if triple not in seen:
                seen.add(triple)
                yield triple

    def _entailed_along_is_a(
        self,
        subjects: tuple[int, ...] | None,
        objects: tuple[int, ...] | None,
        edge_filter: EdgeFilter,
        concept_filter: ConceptFilter | None,
    ) -> Generator[tuple[int, str, int], None, None]:
        """Entail every relationship edge_filter selects, each triple once."""
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
        edge_filter without the ``Is a`` relationship, when it names its relationships.

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
        Find the asserted edges of subjects and of their ``Is a`` ancestors, and
        generalise the filler of each up the ``Is a`` hierarchy.
        """
        wanted = None if objects is None else frozenset(objects)
        above = self._closure(subjects)
        own: dict[int, list[tuple[str, int]]] = defaultdict(list)
        inherited: dict[int, list[tuple[str, int]]] = defaultdict(list)
        for edges, ids, selected in (
            (own, tuple(dict.fromkeys(subjects)), edge_filter),
            (inherited, tuple(set().union(*above.values())), self._without_is_a(edge_filter)),
        ):
            for source, relationship_id, filler in self.relationships(ids, edge_filter=selected):
                edges[source].append((relationship_id, filler))
        ancestors = self._closure(
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
        Find the edges into objects and into their ``Is a`` descendants, and give each
        to the ``Is a`` descendants of its subject.

        The descendants of the objects are all followed, whatever concept_filter selects,
        since an edge can end on a concept outside it. The filter limits the subjects that
        receive an edge and the objects an edge is entailed to. An edge asserted to an
        object is always reported.
        """
        in_scope = set(objects) if concept_filter is None else self._within(objects, concept_filter)
        descendants = self._closure(objects, HierarchyDirection.DOWN)
        targets_by_filler: dict[int, list[int]] = defaultdict(list)
        for target, below in descendants.items():
            for filler in (target, *below):
                targets_by_filler[filler].append(target)
        asserted = list(
            self.relationships(None, objects=targets_by_filler, edge_filter=edge_filter)
        )
        inheriting = {
            subject_id
            for subject_id, relationship_id, _ in asserted
            if relationship_id != IS_A_RELATIONSHIP_ID
        }
        subtypes = self._closure(inheriting, HierarchyDirection.DOWN, concept_filter=concept_filter)
        for subject_id, relationship_id, filler in asserted:
            holders = (subject_id,)
            if relationship_id != IS_A_RELATIONSHIP_ID:
                holders = (subject_id, *subtypes[subject_id])
            for target in targets_by_filler[filler]:
                if target != filler and target not in in_scope:
                    continue
                for holder in holders:
                    yield holder, relationship_id, target
