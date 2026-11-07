"""DuckDB integration coverage for the relationship-classification command."""

from __future__ import annotations

import csv

import sqlalchemy as sa
from oa_configurator import Role

from omop_graph import cli
from omop_graph.extensions.omop_alchemy import (
    PredicateKind,
    RelationshipClass,
    RelationshipMapping,
)


def _read_csv(filename: str) -> list[dict[str, str]]:
    with (cli.packaged_predicate_csv_dir() / filename).open(
        encoding="utf-8", newline=""
    ) as csv_file:
        return [
            {key: value or "" for key, value in row.items() if key is not None}
            for row in csv.DictReader(csv_file)
        ]


def _seed_duckdb_vocabulary(engine, mapping_rows: list[dict[str, str]]) -> set[str]:
    relationship_table = sa.Table(
        "relationship",
        sa.MetaData(),
        sa.Column("relationship_id", sa.String(20), primary_key=True),
        schema=Role.VOCAB.value,
    )
    relationship_ids = {
        row["r_id"].strip() for row in mapping_rows if row.get("r_id", "").strip()
    }
    with engine.begin() as connection:
        relationship_table.create(connection)
        connection.execute(
            relationship_table.insert(),
            [{"relationship_id": relationship_id} for relationship_id in sorted(relationship_ids)],
        )
    return relationship_ids


def test_relationship_classification_runs_and_is_idempotent_on_duckdb(
    duckdb_db,
):
    resolved = duckdb_db.resolved
    assert resolved is not None

    classification_rows = _read_csv("predicate_classification.csv")
    mapping_rows = _read_csv("predicate_mapping.csv")
    relationship_ids = _seed_duckdb_vocabulary(duckdb_db.committing_engine, mapping_rows)

    expected_classes = {
        (row["class"], row["subclass"])
        for row in classification_rows
        if row.get("class") and row.get("subclass")
    }
    expected_mappings = {
        (row["r_id"].strip(), row["class"], row["subclass"])
        for row in mapping_rows
        if row.get("r_id", "").strip()
        and row.get("r_id", "").strip() in relationship_ids
        and row.get("class")
        and row.get("subclass")
    }
    cli.relationship_classification(resolved=resolved)

    with duckdb_db.committing_engine.connect() as connection:
        checks = connection.execute(
            sa.text(
                "SELECT constraint_text FROM duckdb_constraints() "
                "WHERE table_name = 'relationship_class' AND constraint_type = 'CHECK'"
            )
        ).scalars().all()
        assert any(
            "predicate_kind" in check
            and all(f"'{kind.value}'" in check for kind in PredicateKind)
            for check in checks
        )
        actual_classes = {
            (kind.value, subkind)
            for kind, subkind in connection.execute(
                sa.select(
                    RelationshipClass.predicate_kind,
                    RelationshipClass.predicate_subkind,
                )
            )
        }
        actual_mappings = {
            (relationship_id, kind.value, subkind)
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
    assert (
        "Category of",
        PredicateKind.HIERARCHY.value,
        "Categorical - up",
    ) in actual_mappings

    cli.relationship_classification(resolved=resolved)

    with duckdb_db.committing_engine.connect() as connection:
        assert connection.scalar(
            sa.select(sa.func.count()).select_from(RelationshipClass.__table__)
        ) == len(expected_classes)
        assert connection.scalar(
            sa.select(sa.func.count()).select_from(RelationshipMapping.__table__)
        ) == len(expected_mappings)
