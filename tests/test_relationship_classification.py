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
from omop_graph.extensions.omop_alchemy import RelationshipClass, RelationshipMapping


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


_CLASSIFICATION_CSV = (
    "class,subclass,description,semantics,inference\n"
    "Identity,mapping,Identity mapping,identity,none\n"
)
_MAPPING_HEADER = "class,subclass,r_id,r_name,classification_notes,additional_info,link\n"
_MAPS_TO = "Identity,mapping,Maps to,Maps to,,,\n"
_MAPPED_FROM = "Identity,mapping,Mapped from,Mapped from,,,\n"


def _write_files(directory, *mapping_rows: str) -> None:
    (directory / "predicate_classification.csv").write_text(_CLASSIFICATION_CSV)
    (directory / "predicate_mapping.csv").write_text(_MAPPING_HEADER + "".join(mapping_rows))


def test_relationship_classification_rejects_bad_files_and_keeps_the_loaded_tables(pg_db, tmp_path):
    good, bad = tmp_path / "good", tmp_path / "bad"
    good.mkdir()
    bad.mkdir()
    _write_files(good, _MAPS_TO, _MAPPED_FROM)
    _write_files(bad, _MAPS_TO, _MAPPED_FROM, "Identity,mapping,unknown_rel,Unknown,,,\n")

    with scoped_test_schema(
        pg_db.resolved,
        prefix="relclass_unknown",
        schema_claims=[staging_schema_claim()],
    ) as scoped:
        Base.metadata.create_all(bind=scoped.engine, checkfirst=True)
        seed_relationship_vocabulary(scoped.engine)
        relationship_classification(str(good), resolved=require_cdm(scoped.resolved))

        with pytest.raises(ClassificationDataError, match="'unknown_rel'"):
            relationship_classification(str(bad), resolved=require_cdm(scoped.resolved))

        with scoped.engine.connect() as connection:
            relationship_ids = set(
                connection.execute(sa.select(RelationshipMapping.relationship_id)).scalars()
            )

    assert relationship_ids == {"Maps to", "Mapped from"}
