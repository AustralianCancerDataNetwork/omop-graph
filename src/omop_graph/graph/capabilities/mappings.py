"""Mappings between concepts of different vocabularies."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Generator, Iterable

from sqlalchemy import Row

from ...extensions.relationship_classification import PredicateSubkind

# Local Application Imports
from .._streaming import stream
from ..model.edges import (
    REPLACED_BY_RELATIONSHIP_ID,
    EdgeFilter,
    MappingStrength,
)
from ..model.nodes import (
    MappingCardinality,
    MappingView,
)
from ..queries import (
    q_mappings,
)
from .hierarchy import HierarchyMixin


class MappingsMixin(HierarchyMixin):
    """Mappings between concepts of different vocabularies."""

    def mappings(
        self,
        concept_ids: Iterable[int] | None = None,
        edge_filter: EdgeFilter | None = None,
        source: str | None = None,
    ) -> Generator[MappingView, None, None]:
        """
        Yield the edges starting from concepts, with the vocabularies of both ends.

        Edges from a concept to itself and edges without a classification are left out.
        Without concept_ids every edge is read through a server-side cursor.

        Parameters
        ----------
        concept_ids : Iterable[int] | None
            Concepts to start from, or None for all.
        edge_filter : EdgeFilter, optional
            Which edges to return, as in :meth:`relationships`.
        source : str, optional
            Only edges with this vocabulary on either side.

        Yields
        ------
        MappingView
            The edges, each with the vocabularies of both ends and its cardinality.
        """
        resolved = self._resolve_edge_filter(edge_filter)
        if resolved is None:
            return
        if concept_ids is None:
            statements = [stream(q_mappings(None, resolved, source))]
        else:
            ids = tuple(dict.fromkeys(concept_ids))
            statements = [q_mappings(batch, resolved, source) for batch in self.batches(ids)]
        with self.session_factory() as session:
            for stmt in statements:
                for row in session.execute(stmt):
                    if row.predicate_id in self._relationship_mapping:
                        yield self._mapping_view(row)

    @staticmethod
    def _mapping_view(row: Row) -> MappingView:
        """Build a mapping from a row of :func:`q_mappings`."""
        fields = row._asdict()
        cardinality = MappingCardinality.of(
            fields.pop("subjects_of_object"), fields.pop("objects_of_subject")
        )
        return MappingView(**fields, cardinality=cardinality)

    def mapping_predicates(self) -> dict[str, MappingStrength]:
        """
        Return the strength of each relationship that is a mapping, by relationship id.

        Returns
        -------
        dict[str, MappingStrength]
            The strength of each mapping relationship.
        """
        return {
            relationship_id: strength
            for relationship_id, mapping in self._relationship_mapping.items()
            if (
                strength := MappingStrength.of(
                    PredicateSubkind(mapping.predicate_kind, mapping.predicate_subkind)
                )
            )
        }

    def standard_mappings(
        self, concept_ids: Iterable[int], strength: MappingStrength = MappingStrength.EXACT
    ) -> dict[int, tuple[int, ...]]:
        """
        Find the standard concepts each concept maps to by relationships of one strength.

        The chain ends at a standard concept, so a standard concept maps to none. Only one
        step is taken from a non-standard concept, see :meth:`standardise`.

        Parameters
        ----------
        concept_ids : Iterable[int]
            Concepts to look up.
        strength : MappingStrength
            Which mapping relationships to follow.

        Returns
        -------
        dict[int, tuple[int, ...]]
            Every requested concept mapped to its standard concepts in ascending order,
            empty when it has none or is standard itself.

        Raises
        ------
        UnknownConceptError
            If a requested concept does not exist.
        """
        followed = frozenset(
            relationship_id
            for relationship_id, kind in self.mapping_predicates().items()
            if kind is strength
        )
        return {
            concept_id: tuple(other for other in reached if other != concept_id)
            for concept_id, reached in self.standardise(
                concept_ids, EdgeFilter(predicate_ids=followed)
            ).items()
        }

    def replacements(self, concept_ids: Iterable[int]) -> dict[int, tuple[int, ...]]:
        """
        Find the concepts that replace each concept, by its ``Concept replaced by`` edges.

        Parameters
        ----------
        concept_ids : Iterable[int]
            The concepts to find replacements for.

        Returns
        -------
        dict[int, tuple[int, ...]]
            Every requested concept mapped to its replacements, empty when it has none.
        """
        ids = tuple(dict.fromkeys(concept_ids))
        found: dict[int, list[int]] = defaultdict(list)
        for edge in self.edges(
            ids,
            direction="out",
            edge_filter=EdgeFilter(predicate_ids=frozenset({REPLACED_BY_RELATIONSHIP_ID})),
        ):
            found[edge.subject_id].append(edge.object_id)
        return {concept_id: tuple(found[concept_id]) for concept_id in ids}
