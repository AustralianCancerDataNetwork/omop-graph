"""Regression test for the originally reported bug: relationship-classification
silently ignored the configured CDM schema.

Runs entirely on Phase 0's rollback-based ``pg_db`` fixture: real Postgres,
a non-default schema created inside the test's own already-open transaction,
nothing ever committed. Also covers the DROP TYPE naming-mismatch fix
(Phase 4): the enum column never set an explicit ``name=``, so the real
generated type is ``predicatekind``, not the ``predicatekindenum`` the old
raw SQL referenced. Confirmed here rather than assumed.
"""

from __future__ import annotations

import dataclasses
import uuid

import pytest
import sqlalchemy as sa
from oa_configurator import Role, SchemaDriftError, record_schema_provenance

from orm_loader.helpers import Base

from omop_graph.cli import relationship_classification
from omop_graph.extensions.omop_alchemy import RelationshipClass, RelationshipMapping

from fixtures.helpers import schema_translate_map


def _scoped_connection(pg_db, schema: str) -> sa.Connection:
    conn = pg_db.connection
    conn.execute(sa.text(f"CREATE SCHEMA {schema}"))
    return conn.execution_options(schema_translate_map=schema_translate_map(schema))


def test_relationship_classification_respects_the_configured_schema(pg_db):
    """No resolved= is passed, so guard_schema_provenance_for() no-ops here by
    design (a bare engine= caller with no resolved config behind it); this test
    is about schema routing, not provenance drift protection."""
    scoped = _scoped_connection(pg_db, "phase4_regression_test")
    Base.metadata.create_all(bind=scoped, checkfirst=True)

    relationship_classification(engine=scoped)

    n_class = scoped.execute(
        sa.select(sa.func.count()).select_from(RelationshipClass.__table__)
    ).scalar()
    n_mapping = scoped.execute(
        sa.select(sa.func.count()).select_from(RelationshipMapping.__table__)
    ).scalar()
    assert n_class and n_class > 0
    assert n_mapping and n_mapping > 0

    actual_schema = pg_db.connection.execute(
        sa.text(
            "SELECT table_schema FROM information_schema.tables "
            "WHERE table_name = 'relationship_class'"
        )
    ).scalar()
    assert actual_schema == "phase4_regression_test"

    enum_type = pg_db.connection.execute(
        sa.text("SELECT typname FROM pg_type WHERE typname = 'predicatekind'")
    ).scalar()
    assert enum_type == "predicatekind"


def test_relationship_classification_refuses_a_genuinely_split_vocab_connection(pg_db):
    """Postgres has no cross-database inline FK, so RelationshipMapping's FK
    to relationship.relationship_id (VOCAB-tagged) can never be created once
    vocab_connection is a genuinely separate connection.
    """
    resolved = dataclasses.replace(
        pg_db.resolved,
        vocab_connection=dataclasses.replace(
            pg_db.resolved.connection, name="genuinely_different", safe_url="postgresql://other/db"
        ),
    )
    with pytest.raises(RuntimeError, match="genuinely separate"):
        relationship_classification(engine=pg_db.connection, resolved=resolved)


def test_relationship_classification_is_idempotent(pg_db):
    """Re-running against the same schema, the real-world redeploy case the
    DROP TABLE/enum-drop cleanup exists for, must not fail. No resolved= here
    either, so guard_schema_provenance_for() no-ops by design, same as above."""
    scoped = _scoped_connection(pg_db, "phase4_idempotent_test")
    Base.metadata.create_all(bind=scoped, checkfirst=True)

    relationship_classification(engine=scoped)
    relationship_classification(engine=scoped)

    n_class = scoped.execute(
        sa.select(sa.func.count()).select_from(RelationshipClass.__table__)
    ).scalar()
    assert n_class and n_class > 0


def test_relationship_classification_guard_fires_on_reconfigured_schema(pg_db):
    """Unlike the two tests above, resolved= is genuinely passed here -- proves
    the guard actually detects drift for relationship_classification itself,
    not just that it no-ops correctly when a caller opts out."""
    database_name = f"guard_wiring_test_db_{uuid.uuid4().hex[:8]}"
    schema_a = "phase4_guard_wiring_a"
    schema_b = "phase4_guard_wiring_b"

    # Both connection and vocab_connection must point at the same object: relationship_classification()
    # refuses a genuinely split vocab connection, and dataclasses.replace() only overrides fields
    # explicitly passed, so leaving vocab_connection untouched would desync it from the new connection.
    single_connection = dataclasses.replace(pg_db.resolved.connection, test_only=False)
    resolved_a = dataclasses.replace(
        pg_db.resolved,
        name=database_name,
        schema_name=schema_a,
        vocab_schema=schema_a,
        results_schema=schema_a,
        connection=single_connection,
        vocab_connection=single_connection,
    )
    # Baseline must be recorded before schema_a has any tables, or the guard's own
    # first-time-population check trips on Base.metadata.create_all() below.
    scoped_a = _scoped_connection(pg_db, schema_a)
    record_schema_provenance(
        scoped_a,
        database_name=database_name,
        schema_tag=Role.PRIMARY.value,
        new_physical_schema=schema_a,
        reason="test setup",
    )
    Base.metadata.create_all(bind=scoped_a, checkfirst=True)
    relationship_classification(engine=scoped_a, resolved=resolved_a)

    resolved_b = dataclasses.replace(
        resolved_a, schema_name=schema_b, vocab_schema=schema_b, results_schema=schema_b
    )
    scoped_b = _scoped_connection(pg_db, schema_b)
    with pytest.raises(SchemaDriftError):
        relationship_classification(engine=scoped_b, resolved=resolved_b)
