"""OAK-lib adapter construction.

``OMOPAlchemyImplementation`` builds its engines only through
``ResolvedCDMDatabase.create_engines()``. Three construction paths:

1. ``get_adapter("omop:<name>")``: oaklib passes a resource whose slug names
   the oa-configurator database entry.
2. ``resolved=``: an already-resolved CDM database.
3. ``kg=``: an existing ``KnowledgeGraph``, used as-is.
"""

from __future__ import annotations

from datetime import date

import pytest
import sqlalchemy as sa
import sqlalchemy.orm
from oaklib import get_adapter

from oa_configurator import Resolver
from oa_configurator.testing import scoped_test_schema
from omop_alchemy.cdm.model.vocabulary import Concept, Concept_Class, Domain, Vocabulary
from orm_loader.backends import staging_schema_claim
from orm_loader.helpers import Base, bulk_load_context

from omop_graph.cli import relationship_classification
from omop_graph.db.session import resolve_cdm_database
from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.oaklib_interface import omop_implementation
from omop_graph.oaklib_interface.omop_implementation import OMOPAlchemyImplementation

_META_CONCEPT_ID = 0
_CONCEPT_ID = 1001
_TODAY = date(2020, 1, 1)
_FAR_FUTURE = date(2099, 12, 31)


def _seed_one_concept(engine: sa.Engine, *, concept_id: int, name: str) -> None:
    """Minimal vocab bootstrap: Domain/Vocabulary/Concept_Class/Concept form an
    FK insert cycle, so FK checks are disabled around the insert.
    """
    with sa.orm.Session(engine) as session:
        with bulk_load_context(session):
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
            session.flush()
        session.commit()


def _populate(scoped, *, name: str, guarded: bool = True) -> None:
    Base.metadata.create_all(bind=scoped.engine, checkfirst=True)
    _seed_one_concept(scoped.engine, concept_id=_CONCEPT_ID, name=name)
    relationship_classification(engine=scoped.engine, resolved=scoped.resolved if guarded else None)


@pytest.mark.parametrize(("descriptor", "expected_name"), [("omop:named_db", "named_db"), ("omop:", None)])
def test_get_adapter_resolves_the_slug_as_the_database_name(
    pg_db, monkeypatch, descriptor, expected_name
) -> None:
    with scoped_test_schema(pg_db.resolved, prefix="oaklib_get_adapter", schema_claims=[staging_schema_claim()]) as scoped:
        _populate(scoped, name="Adapter concept")

        def fake_resolve_cdm_database(name):
            assert name == expected_name
            return scoped.resolved

        monkeypatch.setattr(omop_implementation, "resolve_cdm_database", fake_resolve_cdm_database)
        adapter = get_adapter(descriptor)

        assert isinstance(adapter, OMOPAlchemyImplementation)
        assert adapter.resource.scheme == "omop"
        assert adapter.label(f"OMOP:{_CONCEPT_ID}") == "Adapter concept"


def test_resolve_cdm_database_resolves_a_named_entry(pg_db) -> None:
    assert resolve_cdm_database(pg_db.resolved.name).name == pg_db.resolved.name


def test_resolved_path_reuses_one_engine_without_a_vocab_split(pg_db) -> None:
    with scoped_test_schema(pg_db.resolved, prefix="oaklib_resolved", schema_claims=[staging_schema_claim()]) as scoped:
        _populate(scoped, name="Resolved-path concept")

        adapter = OMOPAlchemyImplementation(resolved=scoped.resolved)

        assert adapter.kg.vocab_engine is adapter.kg.cdm_engine
        assert adapter.kg.cdm_engine.pool._pre_ping is True
        assert adapter.label(f"OMOP:{_CONCEPT_ID}") == "Resolved-path concept"


def test_resolved_path_builds_a_genuine_vocab_engine_for_a_split_connection(pg_db) -> None:
    name = pg_db.resolved.name
    resolver = Resolver.from_active_config()
    resolver = resolver.with_overrides(
        connections={"oaklib_split_vocab": resolver.config.connections[pg_db.resolved.connection.name]},
        databases={name: resolver.config.databases[name].model_copy(update={"vocab_connection": "oaklib_split_vocab"})},
    )
    with scoped_test_schema(resolver.resolve_database(name), prefix="oaklib_split", resolver=resolver, schema_claims=[staging_schema_claim()]) as scoped:
        # Both connections share one physical server, so the cross-schema FK
        # works; resolved= would make relationship_classification() refuse the split.
        _populate(scoped, name="Split-wiring concept", guarded=False)

        adapter = OMOPAlchemyImplementation(resolved=scoped.resolved)

        assert adapter.kg.vocab_engine is not adapter.kg.cdm_engine
        assert adapter.label(f"OMOP:{_CONCEPT_ID}") == "Split-wiring concept"


def test_kg_injection_path_uses_the_injected_knowledge_graph(pg_db) -> None:
    with scoped_test_schema(pg_db.resolved, prefix="oaklib_kg", schema_claims=[staging_schema_claim()]) as scoped:
        _populate(scoped, name="Injected concept")
        kg = KnowledgeGraph(cdm_engine=scoped.engine)

        adapter = OMOPAlchemyImplementation(kg=kg)

        assert adapter.kg is kg
        assert adapter.label(f"OMOP:{_CONCEPT_ID}") == "Injected concept"
