"""Split-connection vocab routing.

q_edges/q_predicate_row_with_ancestry/q_all_predicates_with_ancestry join a
vocab-tagged table against RelationshipMapping. A genuinely separate 
vocab_connection can't do it in one SQL join. Instead, KnowledgeGraph fetches each side 
from its own engine and merges in Python (see kg.py's
_vocab_split/_predicate_from_rows/_relationship_mapping_lookup).

Uses two real, distinct Postgres connections (test_cdm, test_orm) standing
in for primary/vocab servers, each with its own schema via
isolated_test_schema(). Note: rollback-based isolation can't stand in for two
genuinely separate physical connections.
"""

from __future__ import annotations

from datetime import date
from typing import Iterator, NamedTuple

import pytest
import sqlalchemy as sa
import sqlalchemy.orm as so

from oa_configurator import Role
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

from fixtures.helpers import VOCAB_TABLES, fk_triggers_disabled, schema_translate_map

pytestmark = [pytest.mark.postgresql, pytest.mark.db_dialect]

META_CONCEPT_ID = 0
SUBJECT_CONCEPT_ID = 1
OBJECT_CONCEPT_ID = 2
_TODAY = date(2020, 1, 1)
_FAR_FUTURE = date(2099, 12, 31)

# Relationship/Concept_Relationship aren't in the shared minimal vocab
# bootstrap, but are exactly what the split-connection merge is testing.
_VOCAB_TABLES = VOCAB_TABLES + (Relationship.__table__, Concept_Relationship.__table__)

# Postgres has no cross-database inline FK, so RelationshipMapping's FK to
# relationship.relationship_id can't be created as DDL across genuinely
# separate databases. These shadow tables reproduce the real columns
# without it, since a SELECT/INSERT only depends on column shape.
_shadow_metadata = sa.MetaData()
_shadow_relationship_class = sa.Table(
    "relationship_class",
    _shadow_metadata,
    sa.Column(
        "predicate_kind",
        sa.Enum(
            PredicateKind,
            values_callable=lambda obj: [e.value for e in obj],
            schema=Role.PRIMARY.value,
        ),
        primary_key=True,
    ),
    sa.Column("predicate_subkind", sa.String(20), primary_key=True),
    sa.Column("description", sa.String(80), nullable=False),
    sa.Column("semantics", sa.String(40), nullable=False),
    sa.Column("inference", sa.String(40), nullable=False),
    # Must match RelationshipClass's schema, or SQLAlchemy silently builds a second, unlinked Table object.
    schema=Role.PRIMARY.value,
)
_shadow_relationship_mapping = sa.Table(
    "relationship_mapping",
    _shadow_metadata,
    sa.Column("relationship_id", sa.String(20), primary_key=True),
    sa.Column(
        "predicate_kind",
        sa.Enum(
            PredicateKind,
            values_callable=lambda obj: [e.value for e in obj],
            schema=Role.PRIMARY.value,
        ),
        primary_key=True,
    ),
    sa.Column("predicate_subkind", sa.String(20), primary_key=True),
    schema=Role.PRIMARY.value,
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
                schema_translate_map=schema_translate_map(primary_schema)
            )
            vocab_engine = vocab_raw.execution_options(
                schema_translate_map=schema_translate_map(vocab_schema)
            )

            _shadow_metadata.create_all(primary_engine)
            Base.metadata.create_all(vocab_engine, tables=_VOCAB_TABLES, checkfirst=True)

            # Domain/Vocabulary/Concept_Class/Concept form a genuine FK bootstrap
            # cycle, handled the same way production bulk-loads do: disable FK
            # triggers for the load, then re-enable them.
            with fk_triggers_disabled(vocab_engine, _VOCAB_TABLES):
                _seed(primary_engine, vocab_engine)

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


def test_concept_view_resolves_against_the_vocab_connection(split_engines: _Engines) -> None:
    kg = _split_kg(split_engines)
    view = kg.concept_view(SUBJECT_CONCEPT_ID)

    assert view.concept_id == SUBJECT_CONCEPT_ID
    assert view.concept_name == "Subject concept"


def test_concept_id_by_code_resolves_against_the_vocab_connection(split_engines: _Engines) -> None:
    kg = _split_kg(split_engines)

    assert kg.concept_id_by_code("SNOMED", "SUBJ") == SUBJECT_CONCEPT_ID


def test_concept_ids_by_label_resolves_against_the_vocab_connection(split_engines: _Engines) -> None:
    kg = _split_kg(split_engines)

    assert kg.concept_ids_by_label("Subject concept") == (SUBJECT_CONCEPT_ID,)


def test_predicate_name_resolves_against_the_vocab_connection(split_engines: _Engines) -> None:
    kg = _split_kg(split_engines)

    assert kg.predicate_name("maps to") == "Maps to"


def test_valid_domains_and_vocabularies_resolve_against_the_vocab_connection(
    split_engines: _Engines,
) -> None:
    kg = _split_kg(split_engines)

    assert {"Metadata", "Condition"} <= kg._valid_domains
    assert {"OMOP", "SNOMED"} <= kg._valid_vocabularies


def test_entities_resolves_against_the_vocab_connection(split_engines: _Engines) -> None:
    kg = _split_kg(split_engines)
    ids = tuple(kg.entities(domain="Condition"))

    assert set(ids) == {SUBJECT_CONCEPT_ID, OBJECT_CONCEPT_ID}


def test_relationships_resolves_against_the_vocab_connection(split_engines: _Engines) -> None:
    kg = _split_kg(split_engines)
    triples = tuple(kg.relationships(subjects=(SUBJECT_CONCEPT_ID,), predicates=None, objects=None))

    assert triples == ((SUBJECT_CONCEPT_ID, "maps to", OBJECT_CONCEPT_ID),)


def test_relationships_invert_swaps_subjects_and_objects(split_engines: _Engines) -> None:
    kg = _split_kg(split_engines)
    triples = tuple(
        kg.relationships(subjects=(OBJECT_CONCEPT_ID,), predicates=None, objects=None, invert=True)
    )

    assert triples == ((OBJECT_CONCEPT_ID, "maps to", SUBJECT_CONCEPT_ID),)
