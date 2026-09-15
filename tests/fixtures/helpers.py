"""Shared test-helper primitives for split/non-split, vocab-schema fixtures.

Extracted from four independently-duplicated copies across the test suite
(Phase 4.5 of the schema_translate_map fix): a vocab-table tuple, a
schema_translate_map dict builder, and an FK-trigger toggle context manager.
Schema *provisioning* strategy (rollback-based CREATE SCHEMA on an open
connection vs. a genuinely committed isolated_test_schema) stays local to
each fixture/test instead of being forced into one shared helper here --
that choice genuinely differs per test's needs, unlike the three primitives
below.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, cast

import sqlalchemy as sa
from oa_configurator import Role, qualified
from omop_alchemy.cdm.model.vocabulary import Concept, Concept_Class, Domain, Vocabulary

VOCAB_TABLES = cast(
    "tuple[sa.Table, ...]",
    (Domain.__table__, Vocabulary.__table__, Concept_Class.__table__, Concept.__table__),
)
"""Domain/Vocabulary/Concept_Class/Concept: the minimal vocab bootstrap set
every split/non-split fixture needs, in FK dependency order. A consumer
needing more tables (e.g. Relationship, Concept_Relationship) extends this
tuple rather than redefining its own copy of the shared core.
"""


def schema_translate_map(
    primary_schema: str, *, vocab_schema: str | None = None, results_schema: str | None = None
) -> dict[str, str]:
    """Build a 3-key schema_translate_map dict, defaulting vocab/results to
    primary_schema -- the common "route everything through one schema" shape,
    with an explicit override for the genuinely-split-vocab case.
    """
    return {
        Role.PRIMARY.value: primary_schema,
        Role.VOCAB.value: vocab_schema if vocab_schema is not None else primary_schema,
        Role.RESULTS.value: results_schema if results_schema is not None else primary_schema,
    }


@contextmanager
def fk_triggers_disabled(
    bindable: "sa.Engine | sa.Connection", tables: "tuple[sa.Table, ...]"
) -> Iterator[None]:
    """Disable triggers on *tables* for the duration of the block, matching
    production bulk-loads' own FK-toggle pattern.

    Several vocab tables form a genuine FK insert cycle (each reference
    row's own *_concept_id FK needs a Concept row that doesn't exist yet,
    and Concept's own domain_id/vocabulary_id/concept_class_id FKs need
    those reference rows already present), so a plain ordered insert can't
    satisfy every constraint. A no-op on any non-Postgres dialect: SQLite
    doesn't enforce FK constraints by default, so there's nothing to
    disable there, and callers that seed both dialects can call this
    unconditionally without their own dialect check. Accepts an Engine or
    an open Connection; an Engine has no ``.execute()`` of its own, so one
    short-lived connection is opened for each toggle.
    """
    if bindable.dialect.name != "postgresql":
        yield
        return

    opened_here = isinstance(bindable, sa.Engine)
    conn = bindable.connect() if opened_here else bindable
    try:
        for table in tables:
            conn.execute(sa.text(f"ALTER TABLE {qualified(conn, table.name)} DISABLE TRIGGER ALL"))
        if opened_here:
            conn.commit()
    finally:
        if opened_here:
            conn.close()

    try:
        yield
    finally:
        conn = bindable.connect() if opened_here else bindable
        try:
            for table in tables:
                conn.execute(sa.text(f"ALTER TABLE {qualified(conn, table.name)} ENABLE TRIGGER ALL"))
            if opened_here:
                conn.commit()
        finally:
            if opened_here:
                conn.close()
