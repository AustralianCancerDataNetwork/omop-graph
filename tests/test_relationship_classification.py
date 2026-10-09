"""Regression test for the originally reported bug: relationship-classification
silently ignored the configured CDM schema.

Each test runs against a committed scoped schema from ``scoped_test_schema()``,
dropped on exit. Also covers the DROP TYPE naming-mismatch fix: the enum
column never set an explicit ``name=``, so the real generated type is
``predicatekind``, not ``predicatekindenum``.
"""

from __future__ import annotations

import sqlalchemy as sa
from oa_configurator import Role
from oa_configurator.testing import scoped_test_schema

from orm_loader.backends import staging_schema_claim
from orm_loader.helpers import Base

from omop_graph.cli import relationship_classification
from omop_graph.extensions.omop_alchemy import RelationshipClass, RelationshipMapping


def test_relationship_classification_respects_the_configured_schema(pg_db):
    with scoped_test_schema(pg_db.resolved, prefix="relclass_routing", schema_claims=[staging_schema_claim()]) as scoped:
        schema = scoped.schemas[Role.PRIMARY]
        Base.metadata.create_all(bind=scoped.engine, checkfirst=True)

        relationship_classification(resolved=scoped.resolved)

        with scoped.engine.connect() as conn:
            n_class = conn.execute(
                sa.select(sa.func.count()).select_from(RelationshipClass.__table__)
            ).scalar()
            n_mapping = conn.execute(
                sa.select(sa.func.count()).select_from(RelationshipMapping.__table__)
            ).scalar()
            enum_schema = conn.execute(
                sa.text(
                    "SELECT n.nspname FROM pg_type t JOIN pg_namespace n ON n.oid = t.typnamespace "
                    "WHERE t.typname = 'predicatekind' AND n.nspname = :schema"
                ),
                {"schema": schema},
            ).scalar()
        assert n_class and n_class > 0
        assert n_mapping and n_mapping > 0
        assert sa.inspect(scoped.engine).has_table("relationship_class", schema=schema)
        assert enum_schema == schema


def test_relationship_classification_is_idempotent(pg_db):
    """Re-running against the same schema, the real-world redeploy case the
    DROP TABLE/enum-drop cleanup exists for, must not fail."""
    with scoped_test_schema(pg_db.resolved, prefix="relclass_idempotent", schema_claims=[staging_schema_claim()]) as scoped:
        Base.metadata.create_all(bind=scoped.engine, checkfirst=True)

        relationship_classification(resolved=scoped.resolved)
        relationship_classification(resolved=scoped.resolved)

        with scoped.engine.connect() as conn:
            n_class = conn.execute(
                sa.select(sa.func.count()).select_from(RelationshipClass.__table__)
            ).scalar()
        assert n_class and n_class > 0
