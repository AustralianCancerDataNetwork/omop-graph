"""Regression test for the originally reported bug: relationship-classification
silently ignored the configured CDM schema.

Each test runs against a committed scoped schema from ``scoped_test_schema()``,
dropped on exit. Also covers the DROP TYPE naming-mismatch fix: the enum
column never set an explicit ``name=``, so the real generated type is
``predicatekind``, not ``predicatekindenum``.
"""

from __future__ import annotations

import logging

import sqlalchemy as sa
import sqlalchemy.orm as so
from oa_configurator.testing import scoped_test_schema

from orm_loader.backends import staging_schema_claim
from orm_loader.helpers import Base

from omop_alchemy.cdm.model.vocabulary.relationship import Relationship
from omop_graph.cli import _load_relationship_classification, relationship_classification
from omop_graph.extensions.omop_alchemy import RelationshipClass, RelationshipMapping
from fixtures.mock_cdm import seed_relationship_vocabulary

from fixtures.helpers import cdm_extension_schema_claim


def test_relationship_classification_respects_the_configured_schema(pg_db):
    """The loading body creates RelationshipClass/RelationshipMapping (and the
    predicatekind enum) wherever its engine resolves CDM_EXTENSION_SCHEMA to,
    not a hardcoded literal schema name.
    """
    with (
        cdm_extension_schema_claim(pg_db.committing_engine, prefix="relclass_routing") as ext_claim,
        scoped_test_schema(
            pg_db.resolved,
            prefix="relclass_routing",
            schema_claims=[staging_schema_claim(), ext_claim],
            register_claims=False,
        ) as scoped,
    ):
        extension_schema = ext_claim.physical_schema
        Base.metadata.create_all(bind=scoped.engine, checkfirst=True)
        seed_relationship_vocabulary(scoped.engine)

        with so.Session(scoped.engine) as session:
            relationship_ids = set(session.scalars(sa.select(Relationship.relationship_id)).all())
        _load_relationship_classification(
            None, resolved=scoped.resolved, engine=scoped.engine, relationship_ids=relationship_ids
        )

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
                {"schema": extension_schema},
            ).scalar()
        assert n_class and n_class > 0
        assert n_mapping and n_mapping > 0
        assert sa.inspect(scoped.engine).has_table("relationship_class", schema=extension_schema)
        assert enum_schema == extension_schema


def test_relationship_classification_is_idempotent(pg_db):
    """Re-running against the same schema, the real-world redeploy case the
    DROP TABLE/enum-drop cleanup exists for, must not fail."""
    with scoped_test_schema(
        pg_db.resolved,
        prefix="relclass_idempotent",
        schema_claims=[staging_schema_claim()],
        register_claims=False,
    ) as scoped:
        Base.metadata.create_all(bind=scoped.engine, checkfirst=True)
        seed_relationship_vocabulary(scoped.engine)

        relationship_classification(resolved=scoped.resolved)
        relationship_classification(resolved=scoped.resolved)

        with scoped.engine.connect() as conn:
            n_class = conn.execute(
                sa.select(sa.func.count()).select_from(RelationshipClass.__table__)
            ).scalar()
        assert n_class and n_class > 0


def test_relationship_classification_warns_and_drops_unknown_ids_colocated(
    pg_db, tmp_path, caplog
):
    (tmp_path / "predicate_classification.csv").write_text(
        "class,subclass,description,semantics,inference\n"
        "Identity,mapping,Identity mapping,identity,none\n"
    )
    (tmp_path / "predicate_mapping.csv").write_text(
        "class,subclass,r_id,r_name,classification_notes,additional_info,link\n"
        "Identity,mapping,Maps to,Maps to,,,\n"
        "Identity,mapping,unknown_rel,Unknown,,,\n"
    )

    with scoped_test_schema(
        pg_db.resolved,
        prefix="relclass_unknown",
        schema_claims=[staging_schema_claim()],
        register_claims=False,
    ) as scoped:
        Base.metadata.create_all(bind=scoped.engine, checkfirst=True)
        seed_relationship_vocabulary(scoped.engine)

        with caplog.at_level(logging.WARNING, logger="omop_graph.cli"):
            relationship_classification(str(tmp_path), resolved=scoped.resolved)

        with scoped.engine.connect() as connection:
            relationship_ids = set(
                connection.execute(
                    sa.select(RelationshipMapping.relationship_id)
                ).scalars()
            )

    assert relationship_ids == {"Maps to"}
    assert "unknown_rel" in caplog.text
