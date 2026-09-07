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

from oa_configurator import (
    ResolvedCDMDatabase,
    ResolvedConnection,
    qualified,
)
from oa_configurator.testing import isolated_test_schema
from omop_alchemy.cdm.model.vocabulary import Concept, Concept_Class, Domain, Vocabulary
from orm_loader.helpers import Base

from omop_graph.cli import relationship_classification
from omop_graph.db.session import make_engine
from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.oaklib_interface.omop_factory import omop_resource
from omop_graph.oaklib_interface.omop_implementation import OMOPAlchemyImplementation
from omop_graph.oaklib_interface.omop_resource import OMOPOntologyResource
from omop_graph.config import OmopGraphConfig

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


def test_omop_resource_carries_a_configured_split_vocabulary_target(monkeypatch) -> None:
    primary = ResolvedConnection(
        name="primary",
        url="sqlite:///primary.db",
        safe_url="sqlite:///primary.db",
        _engine_url=sa.make_url("sqlite:///primary.db"),
    )
    vocabulary = ResolvedConnection(
        name="vocabulary",
        url="sqlite:///vocabulary.db",
        safe_url="sqlite:///vocabulary.db",
        _engine_url=sa.make_url("sqlite:///vocabulary.db"),
    )
    resolved = ResolvedCDMDatabase(
        name="split",
        connection=primary,
        schema_name=None,
        vocab_connection=vocabulary,
        vocab_schema=None,
        results_schema=None,
    )

    class FakeResolver:
        def resolve_package_config(self, config_type):
            assert config_type is OmopGraphConfig
            return OmopGraphConfig(cdm_db="split")

        def resolve_database(self, name):
            assert name == "split"
            return resolved

    monkeypatch.setattr(
        "omop_graph.oaklib_interface.omop_factory.Resolver.from_active_config",
        lambda: FakeResolver(),
    )

    resource = omop_resource()

    assert resource.url == primary.url
    assert resource.vocab_url == vocabulary.url
    assert resource.vocab_execution_options == resource.execution_options


def test_omop_alchemy_implementation_builds_a_genuine_vocab_engine_from_a_split_resource(
    pg_db,
) -> None:
    """The other half of the split-vocabulary wiring: omop_resource() deriving
    vocab_url/vocab_execution_options is only useful if OMOPAlchemyImplementation
    actually consumes them. KnowledgeGraph.__init__ eagerly queries via
    cdm_engine (loading relationship-mapping data), so cdm_engine needs a
    real, committed, populated schema; vocab_engine is never queried at
    construction time here, so a syntactically valid but unpopulated URL is
    enough to prove the wiring without a second real database."""
    with isolated_test_schema(pg_db.connection.engine, prefix="phase4_oaklib_split") as schema:
        engine = pg_db.connection.engine.execution_options(
            schema_translate_map={None: schema, "vocab": schema, "results": schema}
        )
        Base.metadata.create_all(bind=engine, checkfirst=True)
        _seed_one_concept(engine, concept_id=_CONCEPT_ID, name="Split-wiring concept")
        relationship_classification(engine=engine)

        resource = OMOPOntologyResource(
            url=pg_db.connection.engine.url.render_as_string(hide_password=False),
            execution_options={
                "schema_translate_map": {None: schema, "vocab": schema, "results": schema}
            },
            vocab_url="sqlite:///:memory:",
            vocab_execution_options={
                "schema_translate_map": {None: None, "vocab": None, "results": None}
            },
        )

        adapter = OMOPAlchemyImplementation(resource=resource)

        assert adapter.kg.vocab_engine is not adapter.kg.cdm_engine
        assert str(adapter.kg.vocab_engine.url) == "sqlite:///:memory:"
        assert adapter.label(f"OMOP:{_CONCEPT_ID}") == "Split-wiring concept"


def test_omop_alchemy_implementation_reuses_one_engine_when_no_split_is_configured(
    pg_db,
) -> None:
    """A resource with no vocab_url (the common case) must not build a second
    engine at all, confirming the new branch is additive, not a regression
    for every construction that isn't split."""
    with isolated_test_schema(pg_db.connection.engine, prefix="phase4_oaklib_nosplit") as schema:
        engine = pg_db.connection.engine.execution_options(
            schema_translate_map={None: schema, "vocab": schema, "results": schema}
        )
        Base.metadata.create_all(bind=engine, checkfirst=True)
        _seed_one_concept(engine, concept_id=_CONCEPT_ID, name="No-split concept")
        relationship_classification(engine=engine)

        resource = OMOPOntologyResource(
            url=pg_db.connection.engine.url.render_as_string(hide_password=False),
            execution_options={
                "schema_translate_map": {None: schema, "vocab": schema, "results": schema}
            },
        )

        adapter = OMOPAlchemyImplementation(resource=resource)

        assert adapter.kg.vocab_engine is adapter.kg.cdm_engine


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
