"""q_concept_name_fulltext() must inspect the VOCAB schema, not primary.

Concept/Concept_Synonym are VOCAB-tagged; inspecting without an explicit
schema= defaults to Role.PRIMARY, so a real primary/vocab split (unlike
SQLite, where every schema tag folds to None) would raise a false
FullTextError even though the tsvector column exists.
"""

from __future__ import annotations

import pytest
import sqlalchemy as sa
from oa_configurator import Role
from oa_configurator.testing import scoped_test_schema
from omop_alchemy.backends import CONCEPT_NAME_TSVECTOR_COLUMN, FullTextError, resolve_backend
from orm_loader.helpers import Base

from omop_graph.graph.queries import q_concept_name_fulltext

from fixtures.helpers import VOCAB_TABLES

pytestmark = [pytest.mark.postgresql, pytest.mark.db_dialect]


def test_fulltext_query_finds_the_tsvector_column_in_the_vocab_schema(pg_db):
    with scoped_test_schema(pg_db.resolved, prefix="fulltext", split_roles=[Role.VOCAB]) as scoped:
        engine = scoped.engine
        Base.metadata.create_all(bind=engine, tables=VOCAB_TABLES, checkfirst=True)

        with engine.begin() as conn:
            resolve_backend(conn).install_fulltext_on_table(
                conn,
                table_name="concept",
                vector_column_name=CONCEPT_NAME_TSVECTOR_COLUMN,
                index_name="idx_concept_name_tsvector_test",
                create_indexes=True,
                fastupdate=True,
                schema_tag=Role.VOCAB.value,
            )

        # Confirmed absent from the primary schema: proves a real vocab/primary
        # split, not a lucky same-schema coincidence.
        inspector = sa.inspect(engine)
        assert not inspector.has_table("concept", schema=scoped.schemas[Role.PRIMARY])
        columns = {c["name"] for c in inspector.get_columns("concept", schema=scoped.schemas[Role.VOCAB])}
        assert CONCEPT_NAME_TSVECTOR_COLUMN in columns

        stmt = q_concept_name_fulltext("kidney cancer", engine=engine)
        assert stmt is not None


def test_fulltext_query_still_raises_when_the_column_is_genuinely_absent(pg_db):
    with scoped_test_schema(pg_db.resolved, prefix="fulltext", split_roles=[Role.VOCAB]) as scoped:
        Base.metadata.create_all(bind=scoped.engine, tables=VOCAB_TABLES, checkfirst=True)

        with pytest.raises(FullTextError):
            q_concept_name_fulltext("kidney cancer", engine=scoped.engine)
