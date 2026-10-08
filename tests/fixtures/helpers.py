"""Shared test-helper primitives for vocab-schema fixtures."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator, cast

import sqlalchemy as sa
from oa_configurator import SchemaClaim
from oa_configurator.testing import isolated_test_schema
from omop_alchemy.cdm.model.vocabulary import (
    Concept,
    Concept_Class,
    Domain,
    Vocabulary
)
from omop_alchemy.config import CDM_EXTENSION_SCHEMA, CDM_EXTENSION_SCHEMA_OWNER

VOCAB_TABLES = cast(
    "tuple[sa.Table, ...]",
    (Domain.__table__, Vocabulary.__table__, Concept_Class.__table__, Concept.__table__),
)
"""Domain/Vocabulary/Concept_Class/Concept: the minimal vocab bootstrap set
every split/non-split fixture needs, in FK dependency order. A consumer
needing more tables (e.g. Relationship, Concept_Relationship) extends this
tuple rather than redefining its own copy of the shared core.
"""


@contextmanager
def cdm_extension_schema_claim(engine: sa.Engine, *, prefix: str) -> Generator[SchemaClaim, None, None]:
    """A SchemaClaim for a fresh, test-isolated physical extension schema.

    Notes
    -----
    Needed before any Base.metadata.create_all() call reachable from a test,
    even one scoped via tables=: a dialect-level custom type (e.g.
    RelationshipClass's predicate_kind) is created metadata-wide, not
    scoped to tables=, so it's attempted regardless of which tables the
    call actually asked for. scoped_test_schema() doesn't scope the
    extension tag on its own (only Role tags), so this must be claimed
    and physically created separately.
    """
    with isolated_test_schema(engine, prefix=f"{prefix}_ext") as physical_schema:
        yield SchemaClaim(
            schema_tag=CDM_EXTENSION_SCHEMA,
            physical_schema=physical_schema,
            owner=CDM_EXTENSION_SCHEMA_OWNER,
        )
