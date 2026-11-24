"""Builders for the small in-memory CDMs the oaklib tests describe row by row."""

from __future__ import annotations

from collections.abc import Generator, Iterable
from contextlib import contextmanager
from datetime import date
from typing import Any, cast

import sqlalchemy as sa
from omop_alchemy.cdm.model.vocabulary.concept import Concept
from omop_alchemy.cdm.model.vocabulary.concept_ancestor import Concept_Ancestor
from omop_alchemy.cdm.model.vocabulary.concept_relationship import Concept_Relationship
from omop_alchemy.cdm.model.vocabulary.concept_synonym import Concept_Synonym
from omop_alchemy.cdm.model.vocabulary.relationship import Relationship
from orm_loader.helpers import Base
from sqlalchemy.orm import Session, sessionmaker

from omop_graph.extensions.omop_alchemy import (
    HierarchyDirection,
    PredicateKind,
    RelationshipClass,
    RelationshipMapping,
)
from omop_graph.oaklib_interface.codec import OMOPLanguage

START = date(2020, 1, 1)
END = date(2099, 12, 31)


def concept(
    concept_id: int,
    name: str | None = None,
    *,
    vocabulary_id: str = "SNOMED",
    standard_concept: str | None = "S",
    concept_code: str | None = None,
    **fields: Any,
) -> Concept:
    """A valid Condition concept, named ``Concept <id>`` and coded ``C<id>`` unless told otherwise."""
    return Concept(
        concept_id=concept_id,
        concept_name=name or f"Concept {concept_id}",
        concept_code=concept_code or f"C{concept_id}",
        domain_id="Condition",
        vocabulary_id=vocabulary_id,
        concept_class_id="Clinical Finding",
        standard_concept=standard_concept,
        valid_start_date=START,
        valid_end_date=END,
        **fields,
    )


def relationship(relationship_id: str, reverse: str) -> Relationship:
    """A relationship named after its id, without hierarchy or ancestry."""
    return Relationship(
        relationship_id=relationship_id,
        relationship_name=relationship_id,
        is_hierarchical="0",
        defines_ancestry="0",
        reverse_relationship_id=reverse,
        relationship_concept_id=0,
    )


def edge(
    subject: int, relationship_id: str, object_: int, *, invalid_reason: str | None = None
) -> Concept_Relationship:
    """A concept relationship valid for the whole CDM, deprecated with *invalid_reason*."""
    return Concept_Relationship(
        concept_id_1=subject,
        relationship_id=relationship_id,
        concept_id_2=object_,
        valid_start_date=START,
        valid_end_date=END,
        invalid_reason=invalid_reason,
    )


def edge_pair(
    subject: int,
    relationship_id: str,
    reverse: str,
    object_: int,
    *,
    invalid_reason: str | None = None,
) -> list[Concept_Relationship]:
    """A relationship and its reverse, as OMOP stores each pair."""
    return [
        edge(subject, relationship_id, object_, invalid_reason=invalid_reason),
        edge(object_, reverse, subject, invalid_reason=invalid_reason),
    ]


def ancestor(ancestor_id: int, descendant_id: int, levels: int = 1) -> Concept_Ancestor:
    """A ``concept_ancestor`` row with the same minimum and maximum separation."""
    return Concept_Ancestor(
        ancestor_concept_id=ancestor_id,
        descendant_concept_id=descendant_id,
        min_levels_of_separation=levels,
        max_levels_of_separation=levels,
    )


def synonym(concept_id: int, name: str, language: OMOPLanguage) -> Concept_Synonym:
    """A synonym of a concept in a language."""
    return Concept_Synonym(
        concept_id=concept_id,
        concept_synonym_name=name,
        language_concept_id=language.concept_id,
    )


def relationship_class(
    kind: PredicateKind, subkind: str, direction: HierarchyDirection | None = None
) -> RelationshipClass:
    """A classification row for a subkind, with placeholder descriptions."""
    return RelationshipClass(
        predicate_kind=kind,
        predicate_subkind=subkind,
        direction=direction,
        description=subkind,
        semantics=subkind,
        inference="none",
    )


def classified(relationship_id: str, kind: PredicateKind, subkind: str) -> RelationshipMapping:
    """The classification of a relationship under a kind and subkind."""
    return RelationshipMapping(
        relationship_id=relationship_id, predicate_kind=kind, predicate_subkind=subkind
    )


@contextmanager
def sqlite_cdm(
    models: Iterable[type[Base]], rows: Iterable[Any]
) -> Generator[sessionmaker[Session]]:
    """
    Create an in-memory SQLite database with the tables of *models* and the *rows*.

    Yields the session factory, and disposes of the engine on exit.
    """
    engine = sa.create_engine("sqlite+pysqlite:///:memory:").execution_options(
        schema_translate_map={"primary": None, "vocab": None}
    )
    Base.metadata.create_all(engine, tables=[cast(sa.Table, model.__table__) for model in models])
    sessions = sessionmaker(bind=engine)
    with sessions() as session:
        session.add_all(list(rows))
        session.commit()
    try:
        yield sessions
    finally:
        engine.dispose()
