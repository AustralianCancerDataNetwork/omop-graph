"""DuckDB integration coverage for the relationship-classification command."""

from __future__ import annotations

import sqlalchemy as sa
from oa_configurator import Role

from omop_graph import cli
from omop_graph.extensions.omop_alchemy import (
    HierarchyDirection,
    PredicateKind,
    RelationshipClass,
    RelationshipMapping,
)


def _seed_duckdb_vocabulary(engine) -> None:
    relationships = [
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
    expected_classes = {
        (PredicateKind.HIERARCHY, "Taxonomic - up", HierarchyDirection.UP),
        (PredicateKind.HIERARCHY, "Taxonomic - down", HierarchyDirection.DOWN),
    }
    expected_mappings = {
        ("Is a", PredicateKind.HIERARCHY, "Taxonomic - up"),
        ("Subsumes", PredicateKind.HIERARCHY, "Taxonomic - down"),
    }

    (tmp_path / "predicate_classification.csv").write_text(
        "class,subclass,description,semantics,inference,direction\n"
        "Hierarchy,Taxonomic - up,Parent,Inheritance,Up,up\n"
        "Hierarchy,Taxonomic - down,Child,Refinement,Down,down\n"
    )
    (tmp_path / "predicate_mapping.csv").write_text(
        "class,subclass,r_id,r_name\n"
        "Hierarchy,Taxonomic - up,Is a,Is a\n"
        "Hierarchy,Taxonomic - down,Subsumes,Subsumes\n"
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
            (kind, subkind, direction)
            for kind, subkind, direction in connection.execute(
                sa.select(
                    RelationshipClass.predicate_kind,
                    RelationshipClass.predicate_subkind,
                    RelationshipClass.direction,
                )
            )
        }
        actual_mappings = {
            (relationship_id, kind, subkind)
            for relationship_id, kind, subkind in connection.execute(
                sa.select(
                    RelationshipMapping.relationship_id,
                    RelationshipMapping.predicate_kind,
                    RelationshipMapping.predicate_subkind,
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
