from __future__ import annotations

from sqlalchemy import select, literal, case, and_, func, exists
from sqlalchemy.orm import aliased
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

def q_concept_views(concept_ids: tuple[int, ...]) -> Select:
    order_map = {cid: index for index, cid in enumerate(concept_ids)}
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
        .where(Concept.concept_id.in_(concept_ids))
        .order_by(case(order_map, value=Concept.concept_id))
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

def q_all_predicates():
    return select(
        Relationship.relationship_id,
        Relationship.relationship_name,
        Relationship.reverse_relationship_id,
        Relationship.is_hierarchical,
        Relationship.defines_ancestry,
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

def q_predicate_row_with_ancestry(relationship_id: str) -> Select:
    """Same as q_predicate_row but also queries the reverse predicate to determine directionality
    of the predicate if either is `defines_ancestry`
    
    Returns
    --------
    Select statement returning columns:
    - relationship_id: str (unique string label for the relationship, e.g. "is a", "maps to", etc.)
    - relationship_name: str (name of the relationship)
    - reverse_relationship_id: Optional[str] (unique string label for the reverse relationship if it exists, e.g. "has" is reverse of "is a")
    - is_hierarchical: str (0 or 1)
    - defines_ancestry: str (0 or 1)
    - is_upward: str (0 or 1)
    - is_downward: str (0 or 1)
    """

    Rel = Relationship
    Rev = aliased(Relationship)

    return (
        select(
            Rel.relationship_id,  # This is not an id but a unique label string...
            Rel.relationship_name,
            Rel.reverse_relationship_id,
            Rel.is_hierarchical,
            Rel.defines_ancestry.label("anc_down"),
            Rev.defines_ancestry.label("anc_up"),
        )
        .join(Rev, Rel.reverse_relationship_id == Rev.relationship_id)  # This is not really joining IDs but matching the relationship_id string to the reverse_relationship_id string
        .where(Rel.relationship_id == relationship_id)
    )

def q_all_predicates_with_ancestry():
    Rel = Relationship
    Rev = aliased(Relationship)
    return (
        select(
            Rel.relationship_id,  # This is not an id but a unique label string...
            Rel.relationship_name,
            Rel.reverse_relationship_id,
            Rel.is_hierarchical,
            Rel.defines_ancestry.label("anc_down"),
            Rev.defines_ancestry.label("anc_up"),
        )
        .join(Rev, Rel.reverse_relationship_id == Rev.relationship_id)
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


def q_outgoing_edges_batch(concept_ids: tuple[int, ...], relationship_id: str | None = None) -> Select:
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

def q_incoming_edges_batch(concept_ids: tuple[int, ...], relationship_id: str | None = None) -> Select:
    stmt = (
        select(
            Concept_Relationship.concept_id_1,
            Concept_Relationship.relationship_id,
            Concept_Relationship.concept_id_2,
            Concept_Relationship.valid_start_date,
            Concept_Relationship.valid_end_date,
            Concept_Relationship.invalid_reason,
        )
        .where(Concept_Relationship.concept_id_2.in_(concept_ids))
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


def q_concept_filtered(vocabulary_id: str | None = None, domain_id: str | None = None) -> Select:
    stmt = (
        select(Concept.concept_id)
        .where(Concept.standard_concept.is_not(None))
    )
    if domain_id:
        stmt = stmt.where(Concept.domain_id == domain_id)
    if vocabulary_id:
        stmt = stmt.where(Concept.vocabulary_id == vocabulary_id)
    return stmt

def q_concept_synonym_filtered(concept_id: int) -> Select:
    stmt = q_concept_synonym()
    return (
        stmt
        .where(Concept_Synonym.concept_id == concept_id)
    )


def q_singletons(*, vocabulary_id: str | None = None, domain_id: str | None = None) -> Select:
    stmt = q_concept_filtered(vocabulary_id=vocabulary_id, domain_id=domain_id)

    return (
        stmt
        .where(
            ~exists(
                select(1).where(
                    and_(
                        Concept_Ancestor.descendant_concept_id == Concept.concept_id,
                        Concept_Ancestor.min_levels_of_separation == 1,
                    )
                )
            )
        )
        .where(
            ~exists(
                select(1).where(
                    and_(
                        Concept_Ancestor.ancestor_concept_id == Concept.concept_id,
                        Concept_Ancestor.min_levels_of_separation == 1,
                    )
                )
            )
        )
    )

def q_roots(*, vocabulary_id: str | None = None, domain_id: str | None = None) -> Select:
    stmt = q_concept_filtered(vocabulary_id=vocabulary_id, domain_id=domain_id)
    return (
        stmt
        .where(
            ~exists(
                select(1).where(
                    and_(
                        Concept_Ancestor.descendant_concept_id == Concept.concept_id,
                        Concept_Ancestor.min_levels_of_separation == 1,
                    )
                )
            )
        )
    )

def q_leaves(*, vocabulary_id: str | None = None, domain_id: str | None = None) -> Select:
    stmt = q_concept_filtered(vocabulary_id=vocabulary_id, domain_id=domain_id)
    return (
        stmt
        .where(
            ~exists(
                select(1).where(
                    and_(
                        Concept_Ancestor.ancestor_concept_id == Concept.concept_id,
                        Concept_Ancestor.min_levels_of_separation == 1,
                    )
                )
            )
        )
    )