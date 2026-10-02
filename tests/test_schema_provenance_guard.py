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

scoped_test_schema() commits real schemas, so every registry row this test
writes is a genuine commit. The Role-tag rows are reset around the test.
"""

from __future__ import annotations

import pytest
import sqlalchemy as sa
from oa_configurator import SchemaDriftError
from oa_configurator.testing import guarded_resolver, reset_schema_registry_rows, scoped_test_schema

from orm_loader.helpers import Base

from omop_graph import cli as omop_graph_cli
from omop_graph.extensions.omop_alchemy import RelationshipClass

pytestmark = [pytest.mark.postgresql, pytest.mark.db_dialect]


def test_relationship_classification_guard_fires_on_reconfigured_schema(
    pg_db, monkeypatch, cleanup_after_test
):
    reset_schema_registry_rows(cleanup_after_test, pg_db.committing_engine)
    resolver = guarded_resolver(pg_db.resolved)
    resolved = resolver.resolve_database(pg_db.resolved.name)

    with scoped_test_schema(resolved, prefix="graph_guard_a", resolver=resolver) as scoped_a:
        Base.metadata.create_all(bind=scoped_a.engine, checkfirst=True)
        monkeypatch.setattr(omop_graph_cli, "resolve_cdm_database", lambda: scoped_a.resolved)
        omop_graph_cli.relationship_classification()

    with scoped_test_schema(resolved, prefix="graph_guard_b", resolver=resolver) as scoped_b:
        Base.metadata.create_all(bind=scoped_b.engine, checkfirst=True)
        monkeypatch.setattr(omop_graph_cli, "resolve_cdm_database", lambda: scoped_b.resolved)
        with pytest.raises(SchemaDriftError):
            omop_graph_cli.relationship_classification()

        # The tables already exist, so the proof the guard fired before any
        # write is that they're still empty.
        with scoped_b.engine.connect() as conn:
            count = conn.execute(
                sa.select(sa.func.count()).select_from(RelationshipClass.__table__)
            ).scalar()
        assert count == 0
