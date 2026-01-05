from __future__ import annotations

from sqlalchemy import select, func, case, literal
from sqlalchemy.sql import Select

from omop_alchemy.cdm.model.vocabulary import (
    Concept,
    Concept_Ancestor,
    Concept_Relationship,
    Relationship,
    Concept_Synonym,
)

def q_concept_view(concept_id: int) -> Select:
    return (
        select(
            Concept.concept_id,
            Concept.concept_name,
            Concept.concept_code,
            Concept.vocabulary_id,
            Concept.domain_id,
            Concept.concept_class_id,
            Concept.standard_concept,
            Concept.valid_start_date,
            Concept.valid_end_date,
            Concept.invalid_reason,
        )
        .where(Concept.concept_id == concept_id)
    )

def q_concept_id_by_code(vocabulary_id: str, concept_code: str) -> Select:
    return (
        select(Concept.concept_id)
        .where(
            Concept.vocabulary_id == vocabulary_id,
            Concept.concept_code == concept_code,
        )
    )

def q_concept_name() -> Select:
    return (
        select(
            Concept.concept_id,
            Concept.concept_name,
            case(
                (Concept.standard_concept.in_(['S', 'C']), literal(True)),
                else_=literal(False),
            ).label('is_standard'),
            case(
                (Concept.invalid_reason.in_(['D', 'U']), literal(False)),
                else_=literal(True),
            ).label('is_active'),
        )
    )

def q_concept_name_match(name: str) -> Select:
    return (
        q_concept_name()
        .where(func.lower(Concept.concept_name) == func.lower(name))
    )

def q_concept_name_ilike(term: str) -> Select:
    return (
        q_concept_name()
        .where(Concept.concept_name.ilike(f"%{term}%"))
    )

def q_concept_synonym() -> Select:
    return (
        select(
            Concept.concept_id,
            Concept_Synonym.concept_synonym_name,
            case(
                (Concept.standard_concept.in_(['S', 'C']), literal(True)),
                else_=literal(False),
            ).label('is_standard'),
            case(
                (Concept.invalid_reason.in_(['D', 'U']), literal(False)),
                else_=literal(True),
            ).label('is_active'),
        )
        .join(Concept, Concept.concept_id == Concept_Synonym.concept_id)
    )

def q_concept_synonym_match(label: str) -> Select:
    return (
        q_concept_synonym()
        .where(func.lower(Concept_Synonym.concept_synonym_name) == func.lower(label))
    )

def q_concept_synonym_ilike(label: str) -> Select:
    return (
        q_concept_synonym()
        .where(Concept_Synonym.concept_synonym_name.ilike(f"%{label}%"))
    )

def q_predicate_name(relationship_id: str) -> Select:
    return (
        select(Relationship.relationship_name)
        .where(Relationship.relationship_id == relationship_id)
    )


def q_predicate_row(relationship_id: str) -> Select:
    return (
        select(
            Relationship.relationship_id,
            Relationship.relationship_name,
            Relationship.reverse_relationship_id,
            Relationship.is_hierarchical,
            Relationship.defines_ancestry,
        )
        .where(Relationship.relationship_id == relationship_id)
    )


def q_outgoing_edges(concept_id: int, relationship_id: str | None = None) -> Select:
    stmt = (
        select(
            Concept_Relationship.concept_id_1,
            Concept_Relationship.relationship_id,
            Concept_Relationship.concept_id_2,
            Concept_Relationship.valid_start_date,
            Concept_Relationship.valid_end_date,
            Concept_Relationship.invalid_reason,
        )
        .where(Concept_Relationship.concept_id_1 == concept_id)
    )
    if relationship_id is not None:
        stmt = stmt.where(Concept_Relationship.relationship_id == relationship_id)
    return stmt


def q_outgoing_edges_batch(concept_ids: list[int], relationship_id: str | None = None) -> Select:
    stmt = (
        select(
            Concept_Relationship.concept_id_1,
            Concept_Relationship.relationship_id,
            Concept_Relationship.concept_id_2,
            Concept_Relationship.valid_start_date,
            Concept_Relationship.valid_end_date,
            Concept_Relationship.invalid_reason,
        )
        .where(Concept_Relationship.concept_id_1.in_(concept_ids))
    )
    if relationship_id is not None:
        stmt = stmt.where(Concept_Relationship.relationship_id == relationship_id)
    return stmt


def q_incoming_edges(concept_id: int, relationship_id: str | None = None) -> Select:
    stmt = (
        select(
            Concept_Relationship.concept_id_1,
            Concept_Relationship.relationship_id,
            Concept_Relationship.concept_id_2,
            Concept_Relationship.valid_start_date,
            Concept_Relationship.valid_end_date,
            Concept_Relationship.invalid_reason,
        )
        .where(Concept_Relationship.concept_id_2 == concept_id)
    )
    if relationship_id is not None:
        stmt = stmt.where(Concept_Relationship.relationship_id == relationship_id)
    return stmt


def q_parents(concept_id: int) -> Select:
    return (
        select(Concept_Ancestor.ancestor_concept_id)
        .where(
            Concept_Ancestor.descendant_concept_id == concept_id,
            Concept_Ancestor.min_levels_of_separation == 1,
        )
    )


def q_children(concept_id: int) -> Select:
    return (
        select(Concept_Ancestor.descendant_concept_id)
        .where(
            Concept_Ancestor.ancestor_concept_id == concept_id,
            Concept_Ancestor.min_levels_of_separation == 1,
        )
    )


def q_ancestors(concept_id: int) -> Select:
    return (
        select(Concept_Ancestor.ancestor_concept_id)
        .where(Concept_Ancestor.descendant_concept_id == concept_id)
    )
