# Extension to omop-alchemy package
import sqlalchemy as sa
import sqlalchemy.orm as so
from orm_loader.helpers import create_tables
from omop_alchemy.cdm.base import (
    ReferenceTable,
    CDMExtensionTableBase,
    merge_table_args,
    role_fk,
)
from omop_alchemy.config import CDM_EXTENSION_SCHEMA
from oa_configurator import ResolvedCDMDatabase, Role

from enum import Enum
from dataclasses import dataclass
from typing import cast


class PredicateKind(Enum):
    HIERARCHY = "Hierarchy"
    IDENTITY = "Identity"
    COMPOSITION = "Composition"
    ASSOCIATION = "Association"
    ATTRIBUTE = "Attribute"


class RelationshipClass(ReferenceTable, CDMExtensionTableBase):
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
            ],
            # Must match CDMExtensionTableBase's own schema to ensure that
            # schema_translate_map routes the enum correctly.
            schema=CDM_EXTENSION_SCHEMA,
        ),
        primary_key=True,
    )
    predicate_subkind: so.Mapped[str] = so.mapped_column(
        sa.String(20), primary_key=True
    )
    description: so.Mapped[str] = so.mapped_column(sa.String(80), nullable=False)
    semantics: so.Mapped[str] = so.mapped_column(sa.String(40), nullable=False)
    inference: so.Mapped[str] = so.mapped_column(sa.String(40), nullable=False)


class RelationshipMapping(ReferenceTable, CDMExtensionTableBase):
    """
    Extensions table: Maps standard OMOP relationship_ids to
    their parent (predicate_kind - one of PredicateKind) and more fine-grained subclasses  .
    """

    __tablename__ = "relationship_mapping"

    relationship_id: so.Mapped[str] = so.mapped_column(
        sa.ForeignKey(role_fk(Role.VOCAB, "relationship.relationship_id")),
        primary_key=True,
    )
    predicate_kind: so.Mapped[PredicateKind] = so.mapped_column(
        sa.Enum(
            PredicateKind,
            values_callable=lambda obj: [
                e.value for e in obj
            ],
            # Must match CDMExtensionTableBase's own schema to ensure that
            # schema_translate_map routes the enum correctly.
            schema=CDM_EXTENSION_SCHEMA,
        ),
        primary_key=True,
    )
    predicate_subkind: so.Mapped[str] = so.mapped_column(
        sa.String(20), primary_key=True
    )

    # Qualified string to RelationshipClass as it lives in the same 
    # fixed extension schema (CDMExtensionTableBase's default).
    __table_args__ = merge_table_args(
        sa.ForeignKeyConstraint(
            ["predicate_kind", "predicate_subkind"],
            [
                f"{CDM_EXTENSION_SCHEMA}.relationship_class.predicate_kind",
                f"{CDM_EXTENSION_SCHEMA}.relationship_class.predicate_subkind",
            ],
            name="fk_rel_mapping_to_rel_class",
        ),
        {"schema": CDM_EXTENSION_SCHEMA},
    )


def create_extension_tables(
    connection: sa.Connection,
    *,
    resolved: ResolvedCDMDatabase,
) -> None:
    """Create the extension tables as they can physically exist on *resolved*.

    Goes through ``orm_loader``'s ``create_tables``: on a split deployment the
    foreign key to the vocabulary is left out, since no dialect can express a
    foreign key across two databases, and every same-tag key, including the
    composite key from relationship_mapping to relationship_class, is kept.

    The ORM classes keep their own ForeignKey declarations and remain what
    reads and writes these tables. SQLAlchemy infers join conditions from
    that in-Python metadata, not from constraints present in the database,
    so relationship loading is unaffected either way.
    """
    # cast: the declarative __table__ is typed FromClause, but a mapped class
    # backed by a table always carries a Table here.
    create_tables(
        connection,
        [
            cast(sa.Table, RelationshipClass.__table__),
            cast(sa.Table, RelationshipMapping.__table__),
        ],
        resolved=resolved,
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
