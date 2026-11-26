"""Predicates and their classification."""

from __future__ import annotations

from sqlalchemy import Row

# Local Application Imports
from ...extensions.omop_alchemy import (
    PredicateKind,
    RelationshipMappingElement,
)
from ..edges import (
    Predicate,
)
from ..queries import (
    q_all_predicates,
    q_predicate_name,
    q_predicate_row,
)
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
        """
        with self.session_factory() as session:
            row = session.execute(q_predicate_row(relationship_id)).one()
        return _predicate_from_rows(row, self._mapping_for(relationship_id))

    def predicate_name(self, relationship_id: str) -> str:
        """
        Retrieve the human-readable name of a relationship.
        """
        # TODO: Not really necessary. The "ID" is mostly human-readable anyways.
        with self.session_factory() as session:
            predicate_name = session.execute(q_predicate_name(relationship_id)).scalar_one()
        return predicate_name

    def predicate_kind(self, relationship_id: str) -> PredicateKind:
        """
        Classify the predicate into a semantic kind.
        """
        return self._mapping_for(relationship_id).predicate_kind

    def _mapping_for(self, relationship_id: str) -> RelationshipMappingElement:
        """Classification of *relationship_id*, raising AttributeError if it has none."""
        item = self._relationship_mapping.get(relationship_id)
        if item is None:
            raise AttributeError(f"`{relationship_id}` not in relationship mapping.")
        return item

    def predicate_kinds(self, relationship_ids: tuple[str, ...]) -> tuple[PredicateKind, ...]:
        """
        Classify a batch of predicates.
        """
        return tuple(self.predicate_kind(rel_id) for rel_id in relationship_ids)

    def reverse_predicate_id(self, relationship_id: str) -> str | None:
        """
        Get the reverse relationship ID, if it exists.
        """
        return self.predicate(relationship_id).reverse_id

    def predicates(self) -> tuple[Predicate, ...]:
        """
        Return all predicates known to the knowledge graph.
        """
        with self.session_factory() as session:
            rows = session.execute(q_all_predicates()).all()
        return tuple(
            _predicate_from_rows(row, self._relationship_mapping[row.relationship_id])
            for row in rows
            if row.relationship_id in self._relationship_mapping
        )
