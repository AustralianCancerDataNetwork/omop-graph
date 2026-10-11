"""Schema-provenance enforcement reaching relationship_classification()'s
production/CLI path (engine=None).

Monkeypatches omop_graph.cli.resolve_cdm_database (the one call site) rather
than the whole oa-configurator config chain, to point at a real, isolated
Postgres schema without touching the active on-disk config.

The test first runs classification on one scoped schema, then reconfigures the
primary schema and verifies that calling relationship_classification() raises
SchemaDriftError while its session engines are being built.

scoped_test_schema() commits real schemas, so every registry row this test
writes is a genuine commit. The Role-tag rows are reset around the test.
"""

from __future__ import annotations

import pytest
from oa_configurator import Role, SchemaDriftError
from oa_configurator.testing import (
    guarded_resolver,
    reset_schema_registry_rows,
    resolve_with_role_schemas,
    scoped_test_schema,
)

from orm_loader.helpers import Base

from omop_graph import cli as omop_graph_cli
from fixtures.mock_cdm import seed_relationship_vocabulary

pytestmark = [pytest.mark.postgresql, pytest.mark.db_dialect]


def test_relationship_classification_guard_fires_on_reconfigured_schema(
    pg_db, monkeypatch, cleanup_after_test
):
    reset_schema_registry_rows(cleanup_after_test, pg_db.committing_engine)
    resolver = guarded_resolver(pg_db.resolved)
    resolved = resolver.resolve_database(pg_db.resolved.name)

    with scoped_test_schema(resolved, prefix="graph_guard_a", resolver=resolver) as scoped_a:
        Base.metadata.create_all(bind=scoped_a.engine, checkfirst=True)
        seed_relationship_vocabulary(scoped_a.engine)
        monkeypatch.setattr(omop_graph_cli, "resolve_cdm_database", lambda: scoped_a.resolved)
        omop_graph_cli.relationship_classification()

    drifted = resolve_with_role_schemas(
        scoped_a.resolved,
        {Role.PRIMARY: "graph_guard_b"},
        resolver=resolver,
    )
    with pytest.raises(SchemaDriftError):
        omop_graph_cli.relationship_classification(resolved=drifted)
