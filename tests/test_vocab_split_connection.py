"""Split-connection vocab routing (Phase 3.2 of the schema_translate_map fix).

``q_edges``, ``q_predicate_row_with_ancestry``, and ``q_all_predicates_with_ancestry``
join a vocab-role table (Relationship/Concept_Relationship) against
RelationshipMapping (an omop-graph extension table, not vocab-role). When
``vocab_connection`` names a physically different server than ``connection``,
a single SQL join can't span both. ``KnowledgeGraph`` fetches each side
from its own engine and merges in Python instead (see kg.py's
``_vocab_split``/``_predicate_from_rows``/``_relationship_mapping_lookup``).

Uses two genuinely distinct, real Postgres connections (``test_cdm``,
``test_orm``) standing in for "primary server" and "vocab server". Each
engine gets its own real, uniquely-named schema via
``oa_configurator.testing.isolated_test_schema()``, since rollback-based
isolation (a single already-open connection) can't stand in for two
genuinely separate physical connections.
"""

from __future__ import annotations

from datetime import date
from typing import Iterator, NamedTuple

import pytest
import sqlalchemy as sa
import sqlalchemy.orm as so

from oa_configurator.testing import isolated_test_database, isolated_test_schema
from orm_loader.config import OrmLoaderConfig
from orm_loader.helpers import Base

from omop_alchemy.cdm.model.vocabulary import (
    Concept,
    Concept_Class,
    Concept_Relationship,
    Domain,
    Relationship,
    Vocabulary,
)

from omop_graph.config import OmopGraphConfig
from omop_graph.extensions.omop_alchemy import (
    PredicateKind,
    RelationshipClass,
    RelationshipMapping,
)
from omop_graph.graph.kg import KnowledgeGraph

pytestmark = [pytest.mark.postgresql, pytest.mark.db_dialect]

META_CONCEPT_ID = 0
SUBJECT_CONCEPT_ID = 1
OBJECT_CONCEPT_ID = 2
_TODAY = date(2020, 1, 1)
_FAR_FUTURE = date(2099, 12, 31)

_VOCAB_TABLES = (
    Domain.__table__,
    Vocabulary.__table__,
    Concept_Class.__table__,
    Concept.__table__,
    Relationship.__table__,
    Concept_Relationship.__table__,
)

# Postgres has no cross-database inline FK (unlike cross-schema, which works
# fine within one database) -- RelationshipMapping's ORM-mapped FK to
# relationship.relationship_id can't be created as DDL when vocab lives on a
# genuinely different database, confirmed empirically while writing this
# test. That FK isn't what's under test here (the Python-side merge is), so
# these shadow tables reproduce RelationshipClass/RelationshipMapping's
# columns without it -- the real ORM classes read/write them identically,
# since a SELECT/INSERT only depends on column shape, not on constraint DDL.
_shadow_metadata = sa.MetaData()
_shadow_relationship_class = sa.Table(
    "relationship_class",
    _shadow_metadata,
    sa.Column(
        "predicate_kind",
        sa.Enum(PredicateKind, values_callable=lambda obj: [e.value for e in obj]),
        primary_key=True,
    ),
    sa.Column("predicate_subkind", sa.String(20), primary_key=True),
    sa.Column("description", sa.String(80), nullable=False),
    sa.Column("semantics", sa.String(40), nullable=False),
    sa.Column("inference", sa.String(40), nullable=False),
)
_shadow_relationship_mapping = sa.Table(
    "relationship_mapping",
    _shadow_metadata,
    sa.Column("relationship_id", sa.String(20), primary_key=True),
    sa.Column(
        "predicate_kind",
        sa.Enum(PredicateKind, values_callable=lambda obj: [e.value for e in obj]),
        primary_key=True,
    ),
    sa.Column("predicate_subkind", sa.String(20), primary_key=True),
)


class _Engines(NamedTuple):
    primary: sa.Engine
    vocab: sa.Engine


@pytest.fixture()
def split_engines() -> Iterator[_Engines]:
    """A primary connection (RelationshipMapping/RelationshipClass) and a
    genuinely separate physical vocab connection (Relationship/Concept/
    Concept_Relationship)."""
    with (
        isolated_test_database(OmopGraphConfig, "test_cdm_db_pg") as primary_db,
        isolated_test_database(OrmLoaderConfig, "test_orm_db_pg") as vocab_db,
    ):
        primary_raw = primary_db.connection.engine
        vocab_raw = vocab_db.connection.engine

        with (
            isolated_test_schema(primary_raw) as primary_schema,
            isolated_test_schema(vocab_raw) as vocab_schema,
        ):
            primary_engine = primary_raw.execution_options(
                schema_translate_map={None: primary_schema, "vocab": primary_schema, "results": primary_schema}
            )
            vocab_engine = vocab_raw.execution_options(
                schema_translate_map={None: vocab_schema, "vocab": vocab_schema, "results": vocab_schema}
            )

            _shadow_metadata.create_all(primary_engine)
            Base.metadata.create_all(vocab_engine, tables=_VOCAB_TABLES, checkfirst=True)

            # Domain/Vocabulary/Concept_Class/Concept form a genuine bootstrap
            # cycle (each reference row's own *_concept_id FK requires a Concept
            # row to already exist, and that Concept row's domain_id/
            # vocabulary_id/concept_class_id FKs require the reference rows to
            # already exist), the same cycle production bulk-loads handle by
            # disabling FK triggers for the load, then re-enabling them.
            with vocab_engine.begin() as conn:
                for table in _VOCAB_TABLES:
                    conn.execute(sa.text(f'ALTER TABLE "{vocab_schema}"."{table.name}" DISABLE TRIGGER ALL'))

            _seed(primary_engine, vocab_engine)

            with vocab_engine.begin() as conn:
                for table in _VOCAB_TABLES:
                    conn.execute(sa.text(f'ALTER TABLE "{vocab_schema}"."{table.name}" ENABLE TRIGGER ALL'))

            yield _Engines(primary=primary_engine, vocab=vocab_engine)


def _seed(primary_engine: sa.Engine, vocab_engine: sa.Engine) -> None:
    with so.Session(vocab_engine) as session:
        session.add_all(
            [
                Concept(
                    concept_id=META_CONCEPT_ID,
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
                    concept_id=SUBJECT_CONCEPT_ID,
                    concept_name="Subject concept",
                    domain_id="Condition",
                    vocabulary_id="SNOMED",
                    concept_class_id="Clinical Finding",
                    standard_concept="S",
                    concept_code="SUBJ",
                    valid_start_date=_TODAY,
                    valid_end_date=_FAR_FUTURE,
                ),
                Concept(
                    concept_id=OBJECT_CONCEPT_ID,
                    concept_name="Object concept",
                    domain_id="Condition",
                    vocabulary_id="SNOMED",
                    concept_class_id="Clinical Finding",
                    standard_concept="S",
                    concept_code="OBJ",
                    valid_start_date=_TODAY,
                    valid_end_date=_FAR_FUTURE,
                ),
                Domain(domain_id="Metadata", domain_name="Metadata", domain_concept_id=META_CONCEPT_ID),
                Domain(domain_id="Condition", domain_name="Condition", domain_concept_id=META_CONCEPT_ID),
                Vocabulary(
                    vocabulary_id="OMOP",
                    vocabulary_name="OMOP",
                    vocabulary_reference="local",
                    vocabulary_version="test",
                    vocabulary_concept_id=META_CONCEPT_ID,
                ),
                Vocabulary(
                    vocabulary_id="SNOMED",
                    vocabulary_name="SNOMED",
                    vocabulary_reference="local",
                    vocabulary_version="test",
                    vocabulary_concept_id=META_CONCEPT_ID,
                ),
                Concept_Class(
                    concept_class_id="Metadata",
                    concept_class_name="Metadata",
                    concept_class_concept_id=META_CONCEPT_ID,
                ),
                Concept_Class(
                    concept_class_id="Clinical Finding",
                    concept_class_name="Clinical Finding",
                    concept_class_concept_id=META_CONCEPT_ID,
                ),
                Relationship(
                    relationship_id="maps to",
                    relationship_name="Maps to",
                    is_hierarchical="0",
                    defines_ancestry="0",
                    reverse_relationship_id="mapped from",
                    relationship_concept_id=META_CONCEPT_ID,
                ),
                Relationship(
                    relationship_id="mapped from",
                    relationship_name="Mapped from",
                    is_hierarchical="0",
                    defines_ancestry="0",
                    reverse_relationship_id="maps to",
                    relationship_concept_id=META_CONCEPT_ID,
                ),
                Concept_Relationship(
                    concept_id_1=SUBJECT_CONCEPT_ID,
                    concept_id_2=OBJECT_CONCEPT_ID,
                    relationship_id="maps to",
                    valid_start_date=_TODAY,
                    valid_end_date=_FAR_FUTURE,
                    invalid_reason=None,
                ),
            ]
        )
        session.commit()

    with so.Session(primary_engine) as session:
        session.add_all(
            [
                RelationshipClass(
                    predicate_kind=PredicateKind.IDENTITY,
                    predicate_subkind="mapping",
                    description="Identity mapping",
                    semantics="identity",
                    inference="none",
                ),
                RelationshipMapping(
                    relationship_id="maps to",
                    predicate_kind=PredicateKind.IDENTITY,
                    predicate_subkind="mapping",
                ),
                RelationshipMapping(
                    relationship_id="mapped from",
                    predicate_kind=PredicateKind.IDENTITY,
                    predicate_subkind="mapping",
                ),
            ]
        )
        session.commit()


def _split_kg(engines: _Engines) -> KnowledgeGraph:
    return KnowledgeGraph(cdm_engine=engines.primary, vocab_engine=engines.vocab)


def test_predicate_merges_across_split_connections(split_engines: _Engines) -> None:
    kg = _split_kg(split_engines)
    predicate = kg.predicate("maps to")

    assert predicate.relationship_id == "maps to"
    assert predicate.reverse_id == "mapped from"
    assert predicate.predicate_kind == PredicateKind.IDENTITY
    assert predicate.predicate_subkind == "mapping"


def test_predicates_merges_across_split_connections(split_engines: _Engines) -> None:
    kg = _split_kg(split_engines)
    by_id = {p.relationship_id: p for p in kg.predicates()}

    assert set(by_id) == {"maps to", "mapped from"}
    assert by_id["maps to"].predicate_kind == PredicateKind.IDENTITY
    assert by_id["maps to"].predicate_subkind == "mapping"


def test_edges_merges_across_split_connections(split_engines: _Engines) -> None:
    kg = _split_kg(split_engines)
    edges = kg.edges(
        concept_ids=SUBJECT_CONCEPT_ID,
        direction="out",
        active_only=False,
        within_domain=False,
    )

    assert len(edges) == 1
    edge = edges[0]
    assert edge.subject_id == SUBJECT_CONCEPT_ID
    assert edge.object_id == OBJECT_CONCEPT_ID
    assert edge.predicate_id == "maps to"
    assert edge.predicate_kind == PredicateKind.IDENTITY
    assert edge.predicate_subkind == "mapping"


def test_edges_predicate_kinds_filter_applies_after_merge(split_engines: _Engines) -> None:
    kg = _split_kg(split_engines)
    edges = kg.edges(
        concept_ids=SUBJECT_CONCEPT_ID,
        direction="out",
        active_only=False,
        within_domain=False,
        predicate_kinds=frozenset({PredicateKind.HIERARCHY}),
    )

    assert edges == ()
