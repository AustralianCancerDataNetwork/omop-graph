"""schema-provenance guard wired into relationship_classification()'s
production/CLI path (engine=None).

Monkeypatches omop_graph.cli.resolve_cdm_database (the one call site) rather
than the whole oa-configurator config chain, to point at a real, isolated
Postgres schema without touching the active on-disk config.

Only the "fires on a genuinely reconfigured schema" case is covered here.
The guard's own agree/no-op/test_only semantics are already exhaustively
covered at the primitive level in oa-configurator's own test suite; what's
worth proving per consuming repo is that this call site is actually wired
to it, and a wiring mistake would show up here too.

resolved.create_engine() below is a real, committing engine (not the
rollback-protected pg_db.connection), so every provenance row this test
writes is a genuine commit. cleanup_after_test deletes this test's own
schema_provenance rows at teardown (see Phase 10.12 in the plan).
"""

from __future__ import annotations

import dataclasses
import uuid

import pytest
import sqlalchemy as sa
from oa_configurator import Role, SchemaDriftError
from oa_configurator.domains.resources.sql import (
    SCHEMA_PROVENANCE_SCHEMA,
    _schema_provenance_table,
    record_schema_provenance,
)
from oa_configurator.testing import delete_rows_on_cleanup, isolated_test_schema

from orm_loader.helpers import Base

from omop_graph import cli as omop_graph_cli
from omop_graph.extensions.omop_alchemy import RelationshipClass

pytestmark = [pytest.mark.postgresql, pytest.mark.db_dialect]


def _resolved(pg_db, *, database_name: str, schema: str):
    """pg_db.resolved with a unique name (the guard's own key includes it),
    all three schemas pointed at schema, and connection.test_only forced
    False so the guard doesn't no-op against pg_db's own test-only marking.
    """
    patched_connection = dataclasses.replace(pg_db.resolved.connection, test_only=False)
    return dataclasses.replace(
        pg_db.resolved,
        name=database_name,
        schema_name=schema,
        vocab_schema=schema,
        results_schema=schema,
        connection=patched_connection,
        vocab_connection=patched_connection,
    )


def test_relationship_classification_guard_fires_on_reconfigured_schema(
    pg_db, monkeypatch, cleanup_after_test
):
    database_name = f"graph_guard_db_{uuid.uuid4().hex[:8]}"
    pg_engine = pg_db.connection.engine
    table = _schema_provenance_table(SCHEMA_PROVENANCE_SCHEMA)
    delete_rows_on_cleanup(
        cleanup_after_test, pg_engine, table, table.c.database_name == database_name
    )
    with (
        isolated_test_schema(pg_engine, prefix="graph_guard_a") as schema_a,
        isolated_test_schema(pg_engine, prefix="graph_guard_b") as schema_b,
    ):
        resolved_a = _resolved(pg_db, database_name=database_name, schema=schema_a)
        engine_a = resolved_a.create_engine()
        try:
            Base.metadata.create_all(bind=engine_a, checkfirst=True)
            # The line above populates the schema outside the guard's own
            # view, so an explicit baseline is needed first, mirroring what
            # a real deployment retrofitting provenance onto an
            # already-populated database would have to do.
            with engine_a.begin() as conn:
                record_schema_provenance(
                    conn,
                    database_name=resolved_a.name,
                    schema_tag=Role.PRIMARY,
                    new_physical_schema=schema_a,
                    reason="test setup baseline",
                )
            monkeypatch.setattr(omop_graph_cli, "resolve_cdm_database", lambda: resolved_a)
            omop_graph_cli.relationship_classification()
        finally:
            engine_a.dispose()

        resolved_b = _resolved(pg_db, database_name=database_name, schema=schema_b)
        engine_b = resolved_b.create_engine()
        try:
            Base.metadata.create_all(bind=engine_b, checkfirst=True)
            monkeypatch.setattr(omop_graph_cli, "resolve_cdm_database", lambda: resolved_b)
            with pytest.raises(SchemaDriftError):
                omop_graph_cli.relationship_classification()

            # This test's own setup step above already created every CDM
            # table, including these two, so the real proof the guard fired
            # before any write is that they're still empty.
            with engine_b.connect() as conn:
                count = conn.execute(
                    sa.select(sa.func.count()).select_from(RelationshipClass.__table__)
                ).scalar()
                assert count == 0
        finally:
            engine_b.dispose()
