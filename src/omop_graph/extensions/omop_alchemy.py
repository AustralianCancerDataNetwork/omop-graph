# Extension to omop-alchemy package
import sqlalchemy as sa
import sqlalchemy.orm as so
from orm_loader.helpers import Base
from omop_alchemy.cdm.base import ReferenceTable, cdm_table, CDMTableBase

from enum import Enum
from dataclasses import dataclass


class PredicateKind(Enum):
    HIERARCHY = "Hierarchy"
    IDENTITY = "Identity"
    COMPOSITION = "Composition"
    ASSOCIATION = "Association"
    ATTRIBUTE = "Attribute"


@cdm_table
class RelationshipClass(ReferenceTable, CDMTableBase, Base):
    """
    Extensions table: Defines the semantic categories (Parent)
    and entity types (Child).
    """

    __tablename__ = "relationship_class"
    predicate_kind: so.Mapped[PredicateKind] = so.mapped_column(
        sa.Enum(
            PredicateKind,
            values_callable=lambda obj: [
                e.value for e in obj
            ],  # Use the value of the enum for storage
        ),
        primary_key=True,
    )
    predicate_subkind: so.Mapped[str] = so.mapped_column(
        sa.String(20), primary_key=True
    )
    description: so.Mapped[str] = so.mapped_column(sa.String(80), nullable=False)
    semantics: so.Mapped[str] = so.mapped_column(sa.String(40), nullable=False)
    inference: so.Mapped[str] = so.mapped_column(sa.String(40), nullable=False)


@cdm_table
class RelationshipMapping(ReferenceTable, CDMTableBase, Base):
    """
    Extensions table: Maps standard OMOP relationship_ids to
    their parent (predicate_kind - one of PredicateKind) and more fine-grained subclasses  .
    """

    __tablename__ = "relationship_mapping"

    relationship_id: so.Mapped[str] = so.mapped_column(
        sa.ForeignKey("relationship.relationship_id"), primary_key=True
    )
    predicate_kind: so.Mapped[PredicateKind] = so.mapped_column(
        sa.Enum(
            PredicateKind,
            values_callable=lambda obj: [
                e.value for e in obj
            ],  # Use the value of the enum for storage
        ),
        primary_key=True,
    )
    predicate_subkind: so.Mapped[str] = so.mapped_column(
        sa.String(20), primary_key=True
    )

    # Define the Composite Foreign Key in __table_args__
    __table_args__ = (
        sa.ForeignKeyConstraint(
            ["predicate_kind", "predicate_subkind"],
            [
                "relationship_class.predicate_kind",
                "relationship_class.predicate_subkind",
            ],
            name="fk_rel_mapping_to_rel_class",
        ),
    )


@dataclass(frozen=True, slots=True)
class RelationshipMappingElement:
    relationship_id: str
    predicate_kind: PredicateKind
    predicate_subkind: str

    @classmethod
    def from_relationship_mapping_entry(cls, entry) -> "RelationshipMappingElement":
        return cls(
            relationship_id=entry.relationship_id,
            predicate_kind=PredicateKind(entry.predicate_kind),
            predicate_subkind=entry.predicate_subkind,
        )


def load_relationship_mapping(
    session: so.Session,
) -> dict[str, RelationshipMappingElement]:
    """Load the entire RelationshipMapping table and return it as a dict keyed by relationship_id."""
    results = session.query(RelationshipMapping).all()
    return {
        row.relationship_id: RelationshipMappingElement.from_relationship_mapping_entry(
            row
        )
        for row in results
    }
