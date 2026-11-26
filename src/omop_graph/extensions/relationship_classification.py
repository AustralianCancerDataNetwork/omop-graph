"""The relationship classification: the tables that store it, its loader and its enums."""

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, cast

import sqlalchemy as sa
import sqlalchemy.orm as so
from oa_configurator import ResolvedCDMDatabase, Role
from omop_alchemy.cdm.base import (
    CDMTableBase,
    ReferenceTable,
    cdm_table,
    merge_table_args,
    role_fk,
)
from orm_loader.helpers import Base, create_tables


class PredicateKind(Enum):
    HIERARCHY = "Hierarchy"
    IDENTITY = "Identity"
    COMPOSITION = "Composition"
    ASSOCIATION = "Association"
    ATTRIBUTE = "Attribute"
    METADATA = "Metadata"


@dataclass(frozen=True, slots=True)
class PredicateSubkind:
    """A subkind name within its kind. The name alone is not unique across kinds."""

    kind: PredicateKind
    name: str


class Subkinds:
    """The subkinds that code depends on, each of which is a row of the packaged classification."""

    EXACT_EQUIVALENCE: ClassVar[PredicateSubkind] = PredicateSubkind(
        PredicateKind.IDENTITY, "Exact equivalence"
    )
    NEAR_EQUIVALENCE: ClassVar[PredicateSubkind] = PredicateSubkind(
        PredicateKind.IDENTITY, "Near equivalence"
    )
    MAPPING: ClassVar[PredicateSubkind] = PredicateSubkind(PredicateKind.ASSOCIATION, "Mapping")
    ADMINISTRATIVE: ClassVar[PredicateSubkind] = PredicateSubkind(
        PredicateKind.ASSOCIATION, "Administrative"
    )
    CONTRAINDICATION: ClassVar[PredicateSubkind] = PredicateSubkind(
        PredicateKind.ASSOCIATION, "Contraindication"
    )
    VALUE_MAPPING: ClassVar[PredicateSubkind] = PredicateSubkind(
        PredicateKind.ASSOCIATION, "Value mapping"
    )

    @classmethod
    def named(cls) -> tuple[PredicateSubkind, ...]:
        """Every subkind the code depends on."""
        return tuple(v for v in vars(cls).values() if isinstance(v, PredicateSubkind))


class HierarchyDirection(Enum):
    """Direction of a Hierarchy relationship, read from the stored edge ``concept_id_1 -> concept_id_2``.

    ``UP`` means ``concept_id_2`` is the parent or group of ``concept_id_1``.
    """

    UP = "up"
    DOWN = "down"


def _enum_values(enum_cls: type[Enum]) -> list[str]:
    return [member.value for member in enum_cls]


def _predicate_kind_enum() -> sa.Enum:
    """Use a varchar-backed enum on DuckDB and the native enum elsewhere."""
    return sa.Enum(
        PredicateKind,
        values_callable=_enum_values,
        schema=Role.PRIMARY.value,
    ).with_variant(
        sa.Enum(
            PredicateKind,
            values_callable=_enum_values,
            native_enum=False,
            create_constraint=True,
        ),
        "duckdb",
    )


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
        _predicate_kind_enum(), primary_key=True
    )
    predicate_subkind: so.Mapped[str] = so.mapped_column(sa.String(20), primary_key=True)
    description: so.Mapped[str] = so.mapped_column(sa.String(80), nullable=False)
    semantics: so.Mapped[str] = so.mapped_column(sa.String(40), nullable=False)
    inference: so.Mapped[str] = so.mapped_column(sa.String(40), nullable=False)


@cdm_table
class RelationshipMapping(ReferenceTable, CDMTableBase, Base):
    """
    Extensions table: Maps standard OMOP relationship_ids to
    their parent (predicate_kind - one of PredicateKind) and more fine-grained subclasses.
    Hierarchy relationships also have a direction, and a relationship may generalise.
    """

    __tablename__ = "relationship_mapping"

    relationship_id: so.Mapped[str] = so.mapped_column(
        sa.ForeignKey(role_fk(Role.VOCAB, "relationship.relationship_id")),
        primary_key=True,
    )
    predicate_kind: so.Mapped[PredicateKind] = so.mapped_column(
        _predicate_kind_enum(), primary_key=True
    )
    predicate_subkind: so.Mapped[str] = so.mapped_column(sa.String(20), primary_key=True)
    direction: so.Mapped[HierarchyDirection | None] = so.mapped_column(
        sa.Enum(
            HierarchyDirection,
            values_callable=_enum_values,
            native_enum=False,
            create_constraint=True,
        ),
        nullable=True,
    )
    generalises: so.Mapped[bool] = so.mapped_column(
        sa.Boolean, nullable=False, default=False, server_default=sa.false()
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


def create_extension_tables(
    connection: sa.Connection,
    *,
    resolved: ResolvedCDMDatabase,
) -> None:
    """Create the extension tables as they can physically exist on resolved.

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


class StaleClassificationError(RuntimeError):
    """The relationship classification tables are missing or lack columns of the current schema."""


@dataclass(frozen=True, slots=True)
class RelationshipMappingElement:
    relationship_id: str
    predicate_kind: PredicateKind
    predicate_subkind: str
    direction: HierarchyDirection | None
    generalises: bool

    @classmethod
    def from_relationship_mapping_entry(cls, entry) -> "RelationshipMappingElement":
        return cls(
            relationship_id=entry.relationship_id,
            predicate_kind=PredicateKind(entry.predicate_kind),
            predicate_subkind=entry.predicate_subkind,
            direction=HierarchyDirection(entry.direction) if entry.direction else None,
            generalises=bool(entry.generalises),
        )


def check_classification_schema(session: so.Session) -> None:
    """Check that the classification tables of the database have every column of the current schema.

    Parameters
    ----------
    session : so.Session
        Session on the database hosting the extension tables.

    Raises
    ------
    StaleClassificationError
        If a table does not exist or lacks columns, which a reload with
        ``omop-graph relationship-classification`` repairs.
    """
    bind = session.get_bind(RelationshipMapping)
    translate = bind.get_execution_options().get("schema_translate_map") or {}
    inspector = sa.inspect(bind)
    problems = []
    for model in (RelationshipClass, RelationshipMapping):
        table = cast(sa.Table, model.__table__)
        schema = translate.get(table.schema, table.schema)
        if not inspector.has_table(table.name, schema=schema):
            problems.append(f"table {table.name} does not exist")
            continue
        present = {column["name"] for column in inspector.get_columns(table.name, schema=schema)}
        missing = [column.name for column in table.columns if column.name not in present]
        if missing:
            problems.append(f"table {table.name} lacks the columns {missing}")
    if problems:
        raise StaleClassificationError(
            f"The relationship classification is outdated: {'; '.join(problems)}. "
            "Run `omop-graph relationship-classification` to load it again."
        )


def load_relationship_mapping(
    session: so.Session,
) -> dict[str, RelationshipMappingElement]:
    """Load the entire RelationshipMapping table and return it as a dict keyed by relationship_id.

    Raises
    ------
    StaleClassificationError
        If the classification tables are missing or lack columns.
    """
    check_classification_schema(session)
    results = session.query(RelationshipMapping).all()
    return {
        row.relationship_id: RelationshipMappingElement.from_relationship_mapping_entry(row)
        for row in results
    }
