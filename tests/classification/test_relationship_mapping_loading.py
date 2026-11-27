"""Loading the stored relationship classification into the knowledge graph."""

from __future__ import annotations

from collections.abc import Iterator
from typing import cast

import pytest
import sqlalchemy as sa
import sqlalchemy.exc
from sqlalchemy.orm import Session, sessionmaker

from omop_graph.extensions.relationship_classification import (
    HierarchyDirection,
    PredicateKind,
    RelationshipClass,
    RelationshipMapping,
    StaleClassificationError,
    load_relationship_mapping,
)
from omop_graph.graph.kg import KnowledgeGraph


@pytest.fixture
def engine() -> Iterator[sa.Engine]:
    """An in-memory database with the two extension tables and one Hierarchy, Identity and Attribute row."""
    engine = sa.create_engine("sqlite+pysqlite:///:memory:").execution_options(
        schema_translate_map={"primary": None, "vocab": None}
    )
    cast(sa.Table, RelationshipClass.__table__).create(engine)
    cast(sa.Table, RelationshipMapping.__table__).create(engine)
    with Session(engine) as session:
        session.add_all(
            [
                RelationshipClass(
                    predicate_kind=kind,
                    predicate_subkind=subkind,
                    description="d",
                    semantics="s",
                    inference="i",
                )
                for kind, subkind in [
                    (PredicateKind.HIERARCHY, "Taxonomic"),
                    (PredicateKind.IDENTITY, "Exact equivalence"),
                    (PredicateKind.ATTRIBUTE, "Anatomical"),
                ]
            ]
        )
        session.flush()
        session.add_all(
            [
                RelationshipMapping(
                    relationship_id="Is a",
                    predicate_kind=PredicateKind.HIERARCHY,
                    predicate_subkind="Taxonomic",
                    direction=HierarchyDirection.UP,
                ),
                RelationshipMapping(
                    relationship_id="Maps to",
                    predicate_kind=PredicateKind.IDENTITY,
                    predicate_subkind="Exact equivalence",
                ),
                RelationshipMapping(
                    relationship_id="Has finding site",
                    predicate_kind=PredicateKind.ATTRIBUTE,
                    predicate_subkind="Anatomical",
                    generalises=True,
                ),
            ]
        )
        session.commit()
    yield engine
    engine.dispose()


def test_mappings_carry_their_direction_and_whether_they_generalise(engine):
    with Session(engine) as session:
        mapping = load_relationship_mapping(session)

    assert mapping["Is a"].direction is HierarchyDirection.UP
    assert mapping["Maps to"].direction is None
    assert mapping["Has finding site"].generalises is True
    assert mapping["Maps to"].generalises is False


def test_a_table_without_a_column_names_the_column_and_the_command(engine):
    with engine.begin() as connection:
        connection.execute(sa.text("ALTER TABLE relationship_mapping DROP COLUMN generalises"))

    with pytest.raises(
        StaleClassificationError, match="relationship_mapping lacks the columns.*generalises"
    ) as raised:
        KnowledgeGraph(sessionmaker(bind=engine))

    assert "omop-graph relationship-classification" in str(raised.value)


def test_a_missing_table_names_the_table_and_the_command(engine):
    with engine.begin() as connection:
        connection.execute(sa.text("DROP TABLE relationship_mapping"))

    with pytest.raises(StaleClassificationError, match="table relationship_mapping does not exist"):
        KnowledgeGraph(sessionmaker(bind=engine))


def test_a_lost_connection_is_not_reported_as_a_stale_classification():
    def unreachable():
        raise sa.exc.OperationalError("connect", {}, Exception("connection refused"))

    with pytest.raises(sa.exc.OperationalError):
        KnowledgeGraph(unreachable)
