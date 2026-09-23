"""q_concept_name_fulltext() must inspect the VOCAB schema, not primary.

Concept/Concept_Synonym are VOCAB-tagged; inspecting without an explicit
schema= defaults to Role.PRIMARY, so a real primary/vocab split (unlike
SQLite, where every schema tag folds to None) would raise a false
FullTextError even though the tsvector column exists.

Schemas are created directly on pg_db's own connection rather than via
isolated_test_schema(), which opens a second connection that can starve
against pg_db's still-open transaction.
"""

from __future__ import annotations

import uuid

import pytest
import sqlalchemy as sa
from oa_configurator import Role
from omop_alchemy.backends import CONCEPT_NAME_TSVECTOR_COLUMN, FullTextError, resolve_backend
from orm_loader.helpers import Base

from omop_graph.graph.queries import q_concept_name_fulltext

from fixtures.helpers import VOCAB_TABLES, schema_translate_map

pytestmark = [pytest.mark.postgresql, pytest.mark.db_dialect]


def _scoped(pg_db, *, primary_schema: str, vocab_schema: str) -> sa.Connection:
    conn = pg_db.connection
    conn.execute(sa.text(f"CREATE SCHEMA {primary_schema}"))
    conn.execute(sa.text(f"CREATE SCHEMA {vocab_schema}"))
    return conn.execution_options(
        schema_translate_map=schema_translate_map(primary_schema, vocab_schema=vocab_schema)
    )


def test_fulltext_query_finds_the_tsvector_column_in_the_vocab_schema(pg_db):
    primary_schema = f"fulltext_primary_{uuid.uuid4().hex[:8]}"
    vocab_schema = f"fulltext_vocab_{uuid.uuid4().hex[:8]}"
    scoped = _scoped(pg_db, primary_schema=primary_schema, vocab_schema=vocab_schema)
    Base.metadata.create_all(bind=scoped, tables=VOCAB_TABLES, checkfirst=True)

    backend = resolve_backend(scoped)
    backend.install_fulltext_on_table(
        scoped,
        table_name="concept",
        vector_column_name=CONCEPT_NAME_TSVECTOR_COLUMN,
        index_name="idx_concept_name_tsvector_test",
        create_indexes=True,
        fastupdate=True,
        schema_tag=Role.VOCAB.value,
    )

    # Confirmed absent from primary_schema: proves a real vocab/primary
    # split, not a lucky same-schema coincidence.
    assert not sa.inspect(pg_db.connection).has_table("concept", schema=primary_schema)
    columns = {
        c["name"] for c in sa.inspect(pg_db.connection).get_columns("concept", schema=vocab_schema)
    }
    assert CONCEPT_NAME_TSVECTOR_COLUMN in columns

    stmt = q_concept_name_fulltext("kidney cancer", engine=scoped)
    assert stmt is not None


def test_fulltext_query_still_raises_when_the_column_is_genuinely_absent(pg_db):
    primary_schema = f"fulltext_primary_{uuid.uuid4().hex[:8]}"
    vocab_schema = f"fulltext_vocab_{uuid.uuid4().hex[:8]}"
    scoped = _scoped(pg_db, primary_schema=primary_schema, vocab_schema=vocab_schema)
    Base.metadata.create_all(bind=scoped, tables=VOCAB_TABLES, checkfirst=True)

    with pytest.raises(FullTextError):
        q_concept_name_fulltext("kidney cancer", engine=scoped)
