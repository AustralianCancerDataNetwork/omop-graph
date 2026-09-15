"""q_concept_name_fulltext() must inspect the VOCAB schema, not primary.

Concept/Concept_Synonym are VOCAB-role tables. schema_inspect(engine) with
no role= defaults to Role.PRIMARY, so whenever vocab_schema differs from
cdm_schema (a normal same-server split, not just a same-schema deployment),
the old code inspected the wrong schema, found no tsvector column, and
raised a false FullTextError even though the column genuinely exists. The
only prior coverage (test_fulltext_optional.py) uses SQLite, where
supports_schemas() is False and every role folds to None -- masking this
completely. Real, distinct primary/vocab Postgres schemas here instead.

Schemas are created directly on pg_db's own already-open connection
(CREATE SCHEMA, rolled back automatically at teardown) rather than via
isolated_test_schema(): that opens a second, genuinely separate connection
from the same pool, which can starve waiting on pg_db's own still-open
transaction.
"""

from __future__ import annotations

import uuid

import pytest
import sqlalchemy as sa
from oa_configurator import Role
from omop_alchemy.backends import CONCEPT_NAME_TSVECTOR_COLUMN, FullTextError, resolve_backend
from omop_alchemy.cdm.model.vocabulary import Concept, Concept_Class, Domain, Vocabulary
from orm_loader.helpers import Base

from omop_graph.graph.queries import q_concept_name_fulltext

pytestmark = [pytest.mark.postgresql, pytest.mark.db_dialect]

_VOCAB_TABLES = (Domain.__table__, Vocabulary.__table__, Concept_Class.__table__, Concept.__table__)


def _scoped(pg_db, *, primary_schema: str, vocab_schema: str) -> sa.Connection:
    conn = pg_db.connection
    conn.execute(sa.text(f"CREATE SCHEMA {primary_schema}"))
    conn.execute(sa.text(f"CREATE SCHEMA {vocab_schema}"))
    return conn.execution_options(
        schema_translate_map={
            Role.PRIMARY.value: primary_schema,
            Role.VOCAB.value: vocab_schema,
            Role.RESULTS.value: primary_schema,
        }
    )


def test_fulltext_query_finds_the_tsvector_column_in_the_vocab_schema(pg_db):
    primary_schema = f"fulltext_primary_{uuid.uuid4().hex[:8]}"
    vocab_schema = f"fulltext_vocab_{uuid.uuid4().hex[:8]}"
    scoped = _scoped(pg_db, primary_schema=primary_schema, vocab_schema=vocab_schema)
    Base.metadata.create_all(bind=scoped, tables=_VOCAB_TABLES, checkfirst=True)

    backend = resolve_backend(scoped)
    backend.install_fulltext_on_table(
        scoped,
        table_name="concept",
        vector_column_name=CONCEPT_NAME_TSVECTOR_COLUMN,
        index_name="idx_concept_name_tsvector_test",
        create_indexes=True,
        fastupdate=True,
        role=Role.VOCAB,
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
    Base.metadata.create_all(bind=scoped, tables=_VOCAB_TABLES, checkfirst=True)

    with pytest.raises(FullTextError):
        q_concept_name_fulltext("kidney cancer", engine=scoped)
