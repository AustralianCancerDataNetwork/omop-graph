# Extension to omop-alchemy package
import sqlalchemy as sa
import sqlalchemy.orm as so
from orm_loader.helpers import Base
from omop_alchemy.cdm.base import (
    ReferenceTable,
    cdm_table,
    CDMTableBase,
    merge_table_args,
    role_fk,
)
from oa_configurator import Role

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
    # Must match RelationshipMapping's schema, or SQLAlchemy silently builds a second, unlinked Table object.
    __table_args__ = merge_table_args({"schema": Role.PRIMARY.value})
    predicate_kind: so.Mapped[PredicateKind] = so.mapped_column(
        sa.Enum(
            PredicateKind,
            values_callable=lambda obj: [
                e.value for e in obj
            ],
            # Must match __table_args__'s schema to ensure that schema_translate_map
            # routes the enum correctly.
            schema=Role.PRIMARY.value,
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
        sa.ForeignKey(role_fk(Role.VOCAB, "relationship.relationship_id")),
        primary_key=True,
    )
    predicate_kind: so.Mapped[PredicateKind] = so.mapped_column(
        sa.Enum(
            PredicateKind,
            values_callable=lambda obj: [
                e.value for e in obj
            ],
            # Must match __table_args__'s schema to ensure that schema_translate_map
            # routes the enum correctly.
            schema=Role.PRIMARY.value,
        ),
        primary_key=True,
    )
    predicate_subkind: so.Mapped[str] = so.mapped_column(
        sa.String(20), primary_key=True
    )

    # Must match RelationshipClass's schema, or SQLAlchemy silently builds a second, unlinked Table object.
    __table_args__ = merge_table_args(
        sa.ForeignKeyConstraint(
            ["predicate_kind", "predicate_subkind"],
            [
                role_fk(Role.PRIMARY, "relationship_class.predicate_kind"),
                role_fk(Role.PRIMARY, "relationship_class.predicate_subkind"),
            ],
            name="fk_rel_mapping_to_rel_class",
        ),
        {"schema": Role.PRIMARY.value},
    )


def relationship_mapping_table_without_vocab_fk() -> sa.Table:
    """RelationshipMapping's DDL without its FK to vocab.relationship_id.

    Postgres has no cross-database inline FK, so when vocab_connection is a
    genuinely separate physical connection, CREATE TABLE on the real ORM
    table fails there. This reproduces the same columns under the same
    name/schema in a throwaway MetaData, so CREATE TABLE succeeds while
    RelationshipMapping's own ORM class still reads/writes the same
    physical table afterward.
    """
    metadata = sa.MetaData()
    return sa.Table(
        RelationshipMapping.__tablename__,
        metadata,
        sa.Column("relationship_id", sa.String(20), primary_key=True),
        sa.Column(
            "predicate_kind",
            sa.Enum(
                PredicateKind,
                values_callable=lambda obj: [e.value for e in obj],
                schema=Role.PRIMARY.value,
            ),
            primary_key=True,
        ),
        sa.Column("predicate_subkind", sa.String(20), primary_key=True),
        sa.ForeignKeyConstraint(
            ["predicate_kind", "predicate_subkind"],
            [
                RelationshipClass.__table__.c.predicate_kind,
                RelationshipClass.__table__.c.predicate_subkind,
            ],
            name="fk_rel_mapping_to_rel_class",
        ),
        schema=Role.PRIMARY.value,
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
