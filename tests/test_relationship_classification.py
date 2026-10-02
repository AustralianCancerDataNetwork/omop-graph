"""Regression test for the originally reported bug: relationship-classification
silently ignored the configured CDM schema.

Each test runs against a committed scoped schema from ``scoped_test_schema()``,
dropped on exit. Also covers the DROP TYPE naming-mismatch fix: the enum
column never set an explicit ``name=``, so the real generated type is
``predicatekind``, not ``predicatekindenum``.
"""

from __future__ import annotations

import pytest
import sqlalchemy as sa
from oa_configurator import ConnectionConfig, Resolver, Role
from oa_configurator.testing import scoped_test_schema

from orm_loader.backends import staging_schema_claim
from orm_loader.helpers import Base

from omop_graph.cli import relationship_classification
from omop_graph.extensions.omop_alchemy import RelationshipClass, RelationshipMapping


def test_relationship_classification_respects_the_configured_schema(pg_db):
    with scoped_test_schema(pg_db.resolved, prefix="relclass_routing", schema_claims=[staging_schema_claim()]) as scoped:
        schema = scoped.schemas[Role.PRIMARY]
        Base.metadata.create_all(bind=scoped.engine, checkfirst=True)

        relationship_classification(engine=scoped.engine, resolved=scoped.resolved)

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


def test_relationship_classification_refuses_a_genuinely_split_vocab_connection(pg_db):
    """Postgres has no cross-database inline FK, so RelationshipMapping's FK
    to relationship.relationship_id (VOCAB-tagged) can never be created once
    vocab_connection is a genuinely separate connection.
    """
    name = pg_db.resolved.name
    resolver = Resolver.from_active_config()
    resolved = resolver.with_overrides(
        connections={
            "genuinely_different": ConnectionConfig(dialect="postgresql+psycopg", host="other", database_name="db")
        },
        databases={name: resolver.config.databases[name].model_copy(update={"vocab_connection": "genuinely_different"})},
    ).resolve_database(name)
    with pytest.raises(RuntimeError, match="genuinely separate"):
        relationship_classification(engine=pg_db.connection, resolved=resolved)


def test_relationship_classification_is_idempotent(pg_db):
    """Re-running against the same schema, the real-world redeploy case the
    DROP TABLE/enum-drop cleanup exists for, must not fail."""
    with scoped_test_schema(pg_db.resolved, prefix="relclass_idempotent", schema_claims=[staging_schema_claim()]) as scoped:
        Base.metadata.create_all(bind=scoped.engine, checkfirst=True)

        relationship_classification(engine=scoped.engine, resolved=scoped.resolved)
        relationship_classification(engine=scoped.engine, resolved=scoped.resolved)

        with scoped.engine.connect() as conn:
            n_class = conn.execute(
                sa.select(sa.func.count()).select_from(RelationshipClass.__table__)
            ).scalar()
        assert n_class and n_class > 0
