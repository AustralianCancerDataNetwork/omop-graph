"""DuckDB integration coverage for the relationship-classification command."""

from __future__ import annotations

import csv

import pytest
import sqlalchemy as sa
from fixtures.helpers import ATHENA_RELATIONSHIPS
from oa_configurator import Role

from omop_graph import cli
from omop_graph.db.session import cdm_session_factory
from omop_graph.extensions.omop_alchemy import (
    HierarchyDirection,
    PredicateKind,
    RelationshipClass,
    RelationshipMapping,
    StaleClassificationError,
    load_relationship_mapping,
)

IS_A_AND_SUBSUMES = [
    {
        "relationship_id": "Is a",
        "relationship_name": "Is a",
        "reverse_relationship_id": "Subsumes",
    },
    {
        "relationship_id": "Subsumes",
        "relationship_name": "Subsumes",
        "reverse_relationship_id": "Is a",
    },
]


def _seed_duckdb_vocabulary(
    engine, relationships: list[dict[str, str]] = IS_A_AND_SUBSUMES
) -> None:
    relationship_table = sa.Table(
        "relationship",
        sa.MetaData(),
        sa.Column("relationship_id", sa.String(20), primary_key=True),
        sa.Column("relationship_name", sa.String(255), nullable=False),
        sa.Column("reverse_relationship_id", sa.String(20), nullable=False),
        schema=Role.VOCAB.value,
    )
    with engine.begin() as connection:
        relationship_table.create(connection)
        connection.execute(relationship_table.insert(), relationships)


def test_relationship_classification_runs_and_is_idempotent_on_duckdb(duckdb_db, tmp_path):
    resolved = duckdb_db.resolved
    assert resolved is not None
    expected_classes = {(PredicateKind.HIERARCHY, "Taxonomic")}
    expected_mappings = {
        ("Is a", PredicateKind.HIERARCHY, "Taxonomic", HierarchyDirection.UP),
        ("Subsumes", PredicateKind.HIERARCHY, "Taxonomic", HierarchyDirection.DOWN),
    }

    (tmp_path / "predicate_classification.csv").write_text(
        "class,subclass,description,semantics,inference\n"
        "Hierarchy,Taxonomic,Lineage,Inheritance,Traversal\n"
    )
    (tmp_path / "predicate_mapping.csv").write_text(
        "class,subclass,r_id,r_name,direction\n"
        "Hierarchy,Taxonomic,Is a,Is a,up\n"
        "Hierarchy,Taxonomic,Subsumes,Subsumes,down\n"
    )
    _seed_duckdb_vocabulary(duckdb_db.committing_engine)
    cli.relationship_classification(str(tmp_path), resolved=resolved)

    with duckdb_db.committing_engine.connect() as connection:
        checks = (
            connection.execute(
                sa.text(
                    "SELECT constraint_text FROM duckdb_constraints() "
                    "WHERE table_name = 'relationship_class' AND constraint_type = 'CHECK'"
                )
            )
            .scalars()
            .all()
        )
        assert any(
            "predicate_kind" in check and all(f"'{kind.value}'" in check for kind in PredicateKind)
            for check in checks
        )
        actual_classes = {
            (kind, subkind)
            for kind, subkind in connection.execute(
                sa.select(
                    RelationshipClass.predicate_kind,
                    RelationshipClass.predicate_subkind,
                )
            )
        }
        actual_mappings = {
            (relationship_id, kind, subkind, direction)
            for relationship_id, kind, subkind, direction in connection.execute(
                sa.select(
                    RelationshipMapping.relationship_id,
                    RelationshipMapping.predicate_kind,
                    RelationshipMapping.predicate_subkind,
                    RelationshipMapping.direction,
                )
            )
        }

    assert actual_classes == expected_classes
    assert actual_mappings == expected_mappings

    cli.relationship_classification(str(tmp_path), resolved=resolved)

    with duckdb_db.committing_engine.connect() as connection:
        assert connection.scalar(
            sa.select(sa.func.count()).select_from(RelationshipClass.__table__)
        ) == len(expected_classes)
        assert connection.scalar(
            sa.select(sa.func.count()).select_from(RelationshipMapping.__table__)
        ) == len(expected_mappings)


def test_packaged_classification_loads_under_strict_mode_on_duckdb(duckdb_db):
    with ATHENA_RELATIONSHIPS.open(encoding="utf-8", newline="") as source:
        athena = list(csv.DictReader(source))
    _seed_duckdb_vocabulary(duckdb_db.committing_engine, athena)

    cli.relationship_classification(strict=True, resolved=duckdb_db.resolved)

    with duckdb_db.committing_engine.connect() as connection:
        mapped = connection.scalar(sa.select(sa.func.count()).select_from(RelationshipMapping))
        with_direction = connection.scalar(
            sa.select(sa.func.count()).where(RelationshipMapping.direction.is_not(None))
        )
        hierarchy = connection.scalar(
            sa.select(sa.func.count()).where(
                RelationshipMapping.predicate_kind == PredicateKind.HIERARCHY
            )
        )
        generalising = connection.scalar(
            sa.select(sa.func.count()).where(RelationshipMapping.generalises.is_(True))
        )
    assert mapped == len(athena)
    assert with_direction == hierarchy > 0
    assert generalising > 0


def test_an_older_duckdb_classification_raises_one_error_naming_the_columns(duckdb_db):
    _seed_duckdb_vocabulary(duckdb_db.committing_engine)
    cli.relationship_classification(resolved=duckdb_db.resolved)
    with duckdb_db.committing_engine.begin() as connection:
        connection.execute(sa.text("ALTER TABLE relationship_mapping DROP COLUMN generalises"))

    with (
        cdm_session_factory(resolved=duckdb_db.resolved)() as session,
        pytest.raises(StaleClassificationError, match="lacks the columns.*generalises"),
    ):
        load_relationship_mapping(session)
