"""OAK-lib adapter schema-awareness gap (Phase 4).

`omop_resource()` used to resolve a full `ResolvedCDMDatabase` internally
but discard it after extracting `database.connection.url`, so
`OMOPAlchemyImplementation`'s internally-built engine never carried
`schema_translate_map`. There are two genuinely different construction
paths here, tested separately:

1. `kg=`-injected construction (what any caller in this stack that can
   reach a `Resolver` should use): the internal `make_engine()` call still
   runs but its result is discarded, so this path was never actually
   broken by the bug. A caller building its own schema-aware engine and
   passing `kg=` already worked. Tested here anyway, since it's the
   pattern this stack's own production code should use and had no
   coverage at all.
2. Bare `engine_string=`/`resource=`-only construction (OAK-lib's own
   generic `materialize()` invocation, which only ever gets a URL string,
   never a live connection): this is the one path the bug actually broke,
   and the only one that needed the `execution_options` fix.
"""

from __future__ import annotations

from datetime import date

import sqlalchemy as sa

from oa_configurator import qualified
from oa_configurator.testing import isolated_test_schema
from omop_alchemy.cdm.model.vocabulary import Concept, Concept_Class, Domain, Vocabulary
from orm_loader.helpers import Base

from omop_graph.cli import relationship_classification
from omop_graph.db.session import make_engine
from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.oaklib_interface.omop_factory import omop_resource
from omop_graph.oaklib_interface.omop_implementation import OMOPAlchemyImplementation
from omop_graph.oaklib_interface.omop_resource import OMOPOntologyResource

_META_CONCEPT_ID = 0
_CONCEPT_ID = 1001
_TODAY = date(2020, 1, 1)
_FAR_FUTURE = date(2099, 12, 31)
_VOCAB_TABLES = (Domain.__table__, Vocabulary.__table__, Concept_Class.__table__, Concept.__table__)


def _seed_one_concept(bindable: sa.Engine | sa.Connection, *, concept_id: int, name: str) -> None:
    """Minimal, real vocab bootstrap: Domain/Vocabulary/Concept_Class/Concept
    form a genuine insert cycle in Postgres (each references-row's own
    *_concept_id FK requires a Concept row to exist, and that Concept row's
    domain_id/vocabulary_id/concept_class_id FKs require the reference rows
    to exist), the same cycle production bulk-loads handle by disabling FK
    triggers for the load, then re-enabling them. Accepts either an Engine
    or an already-open Connection: an Engine has no .execute() of its own,
    so this opens one short-lived connection for the trigger toggles.
    """
    opened_here = isinstance(bindable, sa.Engine)
    conn = bindable.connect() if opened_here else bindable
    try:
        for table in _VOCAB_TABLES:
            conn.execute(sa.text(f"ALTER TABLE {qualified(conn, table.name)} DISABLE TRIGGER ALL"))
        # Only commit a connection opened here: pg_db's own Connection is
        # already inside an explicit, rollback-based outer transaction, and
        # calling .commit() on it directly would end that transaction for
        # real, defeating the isolation the fixture exists to provide. A
        # freshly-opened connection has no such transaction to protect, and
        # DDL needs to actually persist for the Session below (a genuinely
        # separate connection from the pool) to see it.
        if opened_here:
            conn.commit()
    finally:
        if opened_here:
            conn.close()

    with sa.orm.Session(bindable) as session:
        session.add_all(
            [
                Concept(
                    concept_id=_META_CONCEPT_ID,
                    concept_name="Meta concept",
                    domain_id="Metadata",
                    vocabulary_id="OMOP",
                    concept_class_id="Metadata",
                    standard_concept="S",
                    concept_code="META",
                    valid_start_date=_TODAY,
                    valid_end_date=_FAR_FUTURE,
                ),
                Concept(
                    concept_id=concept_id,
                    concept_name=name,
                    domain_id="Metadata",
                    vocabulary_id="OMOP",
                    concept_class_id="Metadata",
                    standard_concept="S",
                    concept_code=str(concept_id),
                    valid_start_date=_TODAY,
                    valid_end_date=_FAR_FUTURE,
                ),
                Domain(domain_id="Metadata", domain_name="Metadata", domain_concept_id=_META_CONCEPT_ID),
                Vocabulary(
                    vocabulary_id="OMOP",
                    vocabulary_name="OMOP",
                    vocabulary_reference="local",
                    vocabulary_version="test",
                    vocabulary_concept_id=_META_CONCEPT_ID,
                ),
                Concept_Class(
                    concept_class_id="Metadata",
                    concept_class_name="Metadata",
                    concept_class_concept_id=_META_CONCEPT_ID,
                ),
            ]
        )
        session.commit()

    conn = bindable.connect() if opened_here else bindable
    try:
        for table in _VOCAB_TABLES:
            conn.execute(sa.text(f"ALTER TABLE {qualified(conn, table.name)} ENABLE TRIGGER ALL"))
        if opened_here:
            conn.commit()
    finally:
        if opened_here:
            conn.close()


def test_omop_resource_execution_options_carry_the_configured_schema() -> None:
    """Object inspection only: no query, no data, no database connection
    at all. omop_resource() resolves the active config's schema_translate_map
    purely from typed config data, and make_engine() with an explicit url=
    never opens a connection either (Engine construction is lazy)."""
    resource = omop_resource()

    engine = make_engine(resource.url, execution_options=resource.execution_options)

    assert resource.execution_options is not None
    assert (
        engine.get_execution_options()["schema_translate_map"]
        == resource.execution_options["schema_translate_map"]
    )


def test_kg_injection_path_resolves_against_the_configured_schema(pg_db) -> None:
    """The kg= injection path this stack's own production code should
    prefer: build a schema-aware engine externally, wrap it, pass kg=.
    The internal make_engine(engine_string, ...) call still runs but its
    result is discarded. engine_string must still be a resolvable dialect,
    just never actually connected to, so a bare "sqlite:///:memory:"
    placeholder is fine here."""
    schema = "phase4_oaklib_kg_injection"
    conn = pg_db.connection
    conn.execute(sa.text(f"CREATE SCHEMA {schema}"))
    scoped = conn.execution_options(
        schema_translate_map={None: schema, "vocab": schema, "results": schema}
    )
    Base.metadata.create_all(bind=scoped, checkfirst=True)
    _seed_one_concept(scoped, concept_id=_CONCEPT_ID, name="Test concept")
    relationship_classification(engine=scoped)

    kg = KnowledgeGraph(cdm_engine=scoped)
    adapter = OMOPAlchemyImplementation(engine_string="sqlite:///:memory:", kg=kg)

    assert adapter.label(f"OMOP:{_CONCEPT_ID}") == "Test concept"


def test_bare_engine_string_path_resolves_against_the_configured_schema(pg_db) -> None:
    """The one path that can't be dependency-injected: OAK-lib's own
    generic materialize() mechanism only ever hands a URL string to
    OMOPAlchemyImplementation, never a live connection. This is the only
    remaining legitimate use of isolated_test_schema() in this whole plan,
    since it's the only caller that genuinely can't accept pg_db's
    rolled-back Connection. Construction goes through omop_resource(),
    which needs a real, committed, independently-connectable schema."""
    with isolated_test_schema(pg_db.connection.engine, prefix="phase4_oaklib_bare") as schema:
        engine = pg_db.connection.engine.execution_options(
            schema_translate_map={None: schema, "vocab": schema, "results": schema}
        )
        Base.metadata.create_all(bind=engine, checkfirst=True)
        _seed_one_concept(engine, concept_id=_CONCEPT_ID, name="Bare-string concept")
        relationship_classification(engine=engine)

        resource = OMOPOntologyResource(
            # str(url) masks the password by default (renders "***"), and
            # this is the one place that string actually needs to be usable
            # to open a real connection, not just for display.
            url=pg_db.connection.engine.url.render_as_string(hide_password=False),
            execution_options={
                "schema_translate_map": {None: schema, "vocab": schema, "results": schema}
            },
        )
        adapter = OMOPAlchemyImplementation(resource=resource)

        assert adapter.label(f"OMOP:{_CONCEPT_ID}") == "Bare-string concept"
