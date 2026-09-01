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

import sqlalchemy as sa

from orm_loader.helpers import Base

from omop_graph.cli import relationship_classification
from omop_graph.extensions.omop_alchemy import RelationshipClass, RelationshipMapping


def _scoped_connection(pg_db, schema: str) -> sa.Connection:
    conn = pg_db.connection
    conn.execute(sa.text(f"CREATE SCHEMA {schema}"))
    return conn.execution_options(
        schema_translate_map={None: schema, "vocab": schema, "results": schema}
    )


def test_relationship_classification_respects_the_configured_schema(pg_db):
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


def test_relationship_classification_is_idempotent(pg_db):
    """Re-running against the same schema, the real-world redeploy case the
    DROP TABLE/enum-drop cleanup exists for, must not fail."""
    scoped = _scoped_connection(pg_db, "phase4_idempotent_test")
    Base.metadata.create_all(bind=scoped, checkfirst=True)

    relationship_classification(engine=scoped)
    relationship_classification(engine=scoped)

    n_class = scoped.execute(
        sa.select(sa.func.count()).select_from(RelationshipClass.__table__)
    ).scalar()
    assert n_class and n_class > 0
