"""Predicates and their classification."""

from __future__ import annotations

import functools

from sqlalchemy import Row

# Local Application Imports
from ...extensions.omop_alchemy import (
    HierarchyDirection,
    PredicateKind,
    RelationshipMappingElement,
)
from ..model.edges import (
    Predicate,
)
from ..queries import q_all_predicates
from ._core import GraphCore


def _predicate_from_rows(row: Row, mapping: RelationshipMappingElement) -> Predicate:
    """Build a Predicate from a relationship row and its classification."""
    return Predicate(
        relationship_id=row.relationship_id,
        name=row.relationship_name,
        reverse_id=row.reverse_relationship_id,
        predicate_kind=mapping.predicate_kind,
        predicate_subkind=mapping.predicate_subkind,
        direction=mapping.direction,
        generalises=mapping.generalises,
    )


class PredicatesMixin(GraphCore):
    """Predicates and their classification."""

    def predicate(self, relationship_id: str) -> Predicate:
        """
        Retrieve a Predicate object by its relationship ID.

        Parameters
        ----------
        relationship_id : str
            The OMOP relationship ID (e.g., 'maps to').

        Returns
        -------
        Predicate
            The predicate definition.

        Raises
        ------
        AttributeError
            If the relationship has no classification.
        """
        found = self.find_predicate(relationship_id)
        if found is None:
            raise AttributeError(f"`{relationship_id}` not in relationship mapping.")
        return found

    def find_predicate(self, relationship_id: str) -> Predicate | None:
        """
        Retrieve a Predicate by its relationship ID, or None when it has no classification.

        Parameters
        ----------
        relationship_id : str
            The OMOP relationship id.

        Returns
        -------
        Predicate | None
            The predicate, or None when it has no classification.
        """
        return self._predicates_by_id.get(relationship_id)

    def predicate_name(self, relationship_id: str) -> str:
        """
        Retrieve the human-readable name of a classified relationship.

        Parameters
        ----------
        relationship_id : str
            The OMOP relationship id.

        Returns
        -------
        str
            The name of the relationship.

        Raises
        ------
        AttributeError
            If the relationship has no classification.
        """
        return self.predicate(relationship_id).name

    def predicate_kind(self, relationship_id: str) -> PredicateKind:
        """
        Classify the predicate into a semantic kind.

        Parameters
        ----------
        relationship_id : str
            The OMOP relationship id.

        Returns
        -------
        PredicateKind
            The kind of the relationship.
        """
        return self._mapping_for(relationship_id).predicate_kind

    def _mapping_for(self, relationship_id: str) -> RelationshipMappingElement:
        """Classification of relationship_id, raising AttributeError if it has none."""
        item = self._relationship_mapping.get(relationship_id)
        if item is None:
            raise AttributeError(f"`{relationship_id}` not in relationship mapping.")
        return item

    @functools.cached_property
    def default_relationship_ids(self) -> frozenset[str]:
        """
        The classified relationships to expose when none is named.

        A Hierarchy relationship is included in its up direction only, so each hierarchical
        edge appears once, from the child to the parent. The down direction is available by
        naming it. The metadata relationships are left out.
        """
        return frozenset(
            relationship_id
            for relationship_id, mapping in self._relationship_mapping.items()
            if mapping.predicate_kind is not PredicateKind.METADATA
            and not (
                mapping.predicate_kind is PredicateKind.HIERARCHY
                and mapping.direction is HierarchyDirection.DOWN
            )
        )

    def predicate_kinds(self, relationship_ids: tuple[str, ...]) -> tuple[PredicateKind, ...]:
        """
        Classify a batch of predicates.

        Parameters
        ----------
        relationship_ids : tuple[str, ...]
            The OMOP relationship ids.

        Returns
        -------
        tuple[PredicateKind, ...]
            The kind of each relationship, in order.
        """
        return tuple(self.predicate_kind(rel_id) for rel_id in relationship_ids)

    def reverse_predicate_id(self, relationship_id: str) -> str | None:
        """
        Get the reverse relationship ID, if it exists.

        Parameters
        ----------
        relationship_id : str
            The OMOP relationship id.

        Returns
        -------
        str | None
            The id of the reverse relationship, if it has one.
        """
        return self.predicate(relationship_id).reverse_id

    @functools.cached_property
    def _predicates_by_id(self) -> dict[str, Predicate]:
        """The classified predicates by relationship id, read once for the life of the graph."""
        with self.session_factory() as session:
            rows = session.execute(q_all_predicates()).all()
        return {
            row.relationship_id: _predicate_from_rows(
                row, self._relationship_mapping[row.relationship_id]
            )
            for row in rows
            if row.relationship_id in self._relationship_mapping
        }

    def predicates(self) -> tuple[Predicate, ...]:
        """
        Return all predicates known to the knowledge graph.

        They are read once, like the relationship classification they are built from.

        Returns
        -------
        tuple[Predicate, ...]
            Every classified predicate.
        """
        return tuple(self._predicates_by_id.values())
