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
from fixtures.helpers import require_cdm
from fixtures.mock_cdm import load_seeded_relationship_classification, seed_relationship_vocabulary
from oa_configurator import Role
from oa_configurator.testing import scoped_test_schema
from orm_loader.backends import staging_schema_claim
from orm_loader.helpers import Base

from omop_graph.cli import relationship_classification
from omop_graph.extensions.classification_data import ClassificationDataError
from omop_graph.extensions.omop_alchemy import (
    PredicateKind,
    RelationshipClass,
    RelationshipMapping,
)


def test_relationship_classification_respects_the_configured_schema(pg_db):
    with scoped_test_schema(
        pg_db.resolved, prefix="relclass_routing", schema_claims=[staging_schema_claim()]
    ) as scoped:
        schema = scoped.schemas[Role.PRIMARY]
        Base.metadata.create_all(bind=scoped.engine, checkfirst=True)
        seed_relationship_vocabulary(scoped.engine)

        load_seeded_relationship_classification(require_cdm(scoped.resolved))

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
    with scoped_test_schema(
        pg_db.resolved, prefix="relclass_idempotent", schema_claims=[staging_schema_claim()]
    ) as scoped:
        Base.metadata.create_all(bind=scoped.engine, checkfirst=True)
        seed_relationship_vocabulary(scoped.engine)

        load_seeded_relationship_classification(require_cdm(scoped.resolved))
        load_seeded_relationship_classification(require_cdm(scoped.resolved))

        with scoped.engine.connect() as conn:
            n_class = conn.execute(
                sa.select(sa.func.count()).select_from(RelationshipClass.__table__)
            ).scalar()
        assert n_class and n_class > 0


def _write_files(directory, mapping_rows: str):
    directory.mkdir()
    (directory / "predicate_classification.csv").write_text(
        "class,subclass,description,semantics,inference\n"
        "Identity,mapping,Identity mapping,identity,none\n"
    )
    (directory / "predicate_mapping.csv").write_text("class,subclass,r_id,r_name\n" + mapping_rows)


def _loaded_ids(engine) -> set[str]:
    with engine.connect() as connection:
        return set(connection.execute(sa.select(RelationshipMapping.relationship_id)).scalars())


MAPS_TO = "Identity,mapping,Maps to,Maps to\nIdentity,mapping,Mapped from,Mapped from\n"


def test_relationship_classification_skips_unknown_ids_and_fails_on_them_when_strict(
    pg_db, tmp_path
):
    unknown = "Identity,mapping,unknown_rel,Unknown\n"
    _write_files(tmp_path / "good", MAPS_TO)
    _write_files(tmp_path / "unknown", MAPS_TO + unknown)

    with scoped_test_schema(
        pg_db.resolved, prefix="relclass_unknown", schema_claims=[staging_schema_claim()]
    ) as scoped:
        Base.metadata.create_all(bind=scoped.engine, checkfirst=True)
        seed_relationship_vocabulary(scoped.engine)
        resolved = require_cdm(scoped.resolved)
        relationship_classification(str(tmp_path / "good"), resolved=resolved)

        with pytest.raises(ClassificationDataError, match="'unknown_rel'"):
            relationship_classification(str(tmp_path / "unknown"), strict=True, resolved=resolved)
        kept = _loaded_ids(scoped.engine)

        relationship_classification(str(tmp_path / "unknown"), resolved=resolved)
        loaded = _loaded_ids(scoped.engine)

    assert kept == loaded == {"Maps to", "Mapped from"}


def test_relationship_classification_rejects_an_undefined_class_and_keeps_the_loaded_tables(
    pg_db, tmp_path
):
    _write_files(tmp_path / "good", MAPS_TO)
    _write_files(tmp_path / "bad", MAPS_TO + "Identity,nope,Mapped to,x\n")

    with scoped_test_schema(
        pg_db.resolved, prefix="relclass_undefined", schema_claims=[staging_schema_claim()]
    ) as scoped:
        Base.metadata.create_all(bind=scoped.engine, checkfirst=True)
        seed_relationship_vocabulary(scoped.engine)
        resolved = require_cdm(scoped.resolved)
        relationship_classification(str(tmp_path / "good"), resolved=resolved)

        with pytest.raises(ClassificationDataError, match="without a classification"):
            relationship_classification(str(tmp_path / "bad"), resolved=resolved)
        kept = _loaded_ids(scoped.engine)

    assert kept == {"Maps to", "Mapped from"}


def test_relationship_classification_replaces_a_predicate_kind_type_from_an_older_release(
    pg_db, tmp_path
):
    _write_files(tmp_path / "good", MAPS_TO)
    with scoped_test_schema(
        pg_db.resolved, prefix="relclass_stale_enum", schema_claims=[staging_schema_claim()]
    ) as scoped:
        schema = scoped.schemas[Role.PRIMARY]
        Base.metadata.create_all(bind=scoped.engine, checkfirst=True)
        seed_relationship_vocabulary(scoped.engine)
        with scoped.engine.begin() as connection:
            connection.execute(sa.text(f'DROP TABLE "{schema}".relationship_mapping'))
            connection.execute(sa.text(f'DROP TABLE "{schema}".relationship_class'))
            connection.execute(sa.text(f'DROP TYPE "{schema}".predicatekind'))
            connection.execute(
                sa.text(
                    f'CREATE TYPE "{schema}".predicatekind AS ENUM '
                    "('Hierarchy', 'Identity', 'Composition', 'Association')"
                )
            )
            connection.execute(
                sa.text(
                    f'CREATE TABLE "{schema}".relationship_class ('
                    f'predicate_kind "{schema}".predicatekind, predicate_subkind varchar(20), '
                    "PRIMARY KEY (predicate_kind, predicate_subkind))"
                )
            )
            connection.execute(
                sa.text(
                    f'CREATE TABLE "{schema}".relationship_mapping ('
                    f'relationship_id varchar(20), predicate_kind "{schema}".predicatekind, '
                    "predicate_subkind varchar(20))"
                )
            )

        relationship_classification(str(tmp_path / "good"), resolved=require_cdm(scoped.resolved))

        with scoped.engine.connect() as connection:
            labels = set(
                connection.execute(
                    sa.text(
                        "SELECT e.enumlabel FROM pg_enum e JOIN pg_type t ON t.oid = e.enumtypid "
                        "JOIN pg_namespace n ON n.oid = t.typnamespace "
                        "WHERE t.typname = 'predicatekind' AND n.nspname = :schema"
                    ),
                    {"schema": schema},
                ).scalars()
            )

    assert labels == {kind.value for kind in PredicateKind}
