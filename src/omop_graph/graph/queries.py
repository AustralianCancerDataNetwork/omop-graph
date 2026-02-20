"""
SQLAlchemy query generators for the OMOP Knowledge Graph.

This module contains factory functions that produce SQLAlchemy `Select` statements.
These statements are executed by the `KnowledgeGraph` class against the OMOP CDM database.

The queries cover:
* Concept retrieval (by ID, code, or list).
* Label and Synonym matching (exact, fuzzy, full-text).
* Predicate (Relationship) definitions.
* Graph Traversal (outgoing/incoming edges).
* Hierarchy analysis (parents, children, ancestors, roots, leaves).
"""

from __future__ import annotations

from typing import Optional, Tuple, Type

from sqlalchemy import and_, case, exists, func, literal, select
from sqlalchemy.orm import aliased
from sqlalchemy.sql import Select

from omop_alchemy.cdm.model.vocabulary import (
    Concept,
    Concept_Ancestor,
    Concept_Relationship,
    Concept_Synonym,
    Relationship,
)
from omop_alchemy.cdm.base.embeddings import ModelRegistry, EmbeddingBase

from .constraints import SearchConstraintConcept


def q_concept_view(concept_id: int) -> Select:
    """
    Query for a single concept by its ID.

    Parameters
    ----------
    concept_id : int
        The OMOP Concept ID.

    Returns
    -------
    Select
        A statement selecting standard concept columns.
    """
    return select(
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
    ).where(Concept.concept_id == concept_id)


def q_concept_views(concept_ids: Tuple[int, ...]) -> Select:
    """
    Query for multiple concepts by their IDs, preserving the input order.

    Parameters
    ----------
    concept_ids : tuple[int, ...]
        A tuple of OMOP Concept IDs.

    Returns
    -------
    Select
        A statement selecting standard concept columns ordered by the input list.
    """
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
    """
    Query for a Concept ID given its source vocabulary and code.

    Parameters
    ----------
    vocabulary_id : str
        The vocabulary ID (e.g., 'SNOMED').
    concept_code : str
        The source code (e.g., '12345').

    Returns
    -------
    Select
        A statement selecting just the concept_id.
    """
    return (
        select(Concept.concept_id)
        .where(
            Concept.vocabulary_id == vocabulary_id,
            Concept.concept_code == concept_code,
        )
    )


def q_concept_name() -> Select:
    """
    Base query for concept names and their status.

    Returns
    -------
    Select
        A statement selecting ID, name, standard status, and active status.
    """
    return select(
        Concept.concept_id,
        Concept.concept_name,
        case(
            (Concept.standard_concept.in_(["S", "C"]), literal(True)),
            else_=literal(False),
        ).label("is_standard"),
        case(
            (Concept.invalid_reason.in_(["D", "U"]), literal(False)),
            else_=literal(True),
        ).label("is_active"),
    )


def q_concept_name_match(
    name: str, search_constraint: Optional[SearchConstraintConcept] = None
) -> Select:
    """
    Query for exact case-insensitive matches on concept names.

    Parameters
    ----------
    name : str
        The concept name to match.
    search_constraint : SearchConstraintConcept, optional
        Additional filters (domain, vocab).

    Returns
    -------
    Select
        The query statement.
    """
    base_stmt = q_concept_name().where(
        func.lower(Concept.concept_name) == func.lower(name)
    )
    if search_constraint:
        if not isinstance(search_constraint, SearchConstraintConcept):
            raise TypeError(
                "search_constraint must be an instance of SearchConstraintConcept"
            )
        base_stmt = search_constraint.apply(base_stmt)
    return base_stmt


def q_concept_name_ilike(
    term: str, search_constraint: Optional[SearchConstraintConcept] = None
) -> Select:
    """
    Query for partial matches on concept names using ILIKE.

    Parameters
    ----------
    term : str
        The search term (without wildcards; wildcards are added automatically).
    search_constraint : SearchConstraintConcept, optional
        Additional filters.

    Returns
    -------
    Select
        The query statement.
    """
    base_stmt = q_concept_name().where(Concept.concept_name.ilike(f"%{term}%"))
    if search_constraint:
        if not isinstance(search_constraint, SearchConstraintConcept):
            raise TypeError(
                "search_constraint must be an instance of SearchConstraintConcept"
            )
        base_stmt = search_constraint.apply(base_stmt)
    return base_stmt


def q_concept_synonym() -> Select:
    """
    Base query for concept synonyms joined with concept status.

    Returns
    -------
    Select
        A statement selecting ID, synonym name, and concept status flags.
    """
    return (
        select(
            Concept.concept_id,
            Concept_Synonym.concept_synonym_name,
            case(
                (Concept.standard_concept.in_(["S", "C"]), literal(True)),
                else_=literal(False),
            ).label("is_standard"),
            case(
                (Concept.invalid_reason.in_(["D", "U"]), literal(False)),
                else_=literal(True),
            ).label("is_active"),
        )
        .join(Concept, Concept.concept_id == Concept_Synonym.concept_id)
    )


def q_concept_synonym_match(
    label: str, search_constraint: Optional[SearchConstraintConcept] = None
) -> Select:
    """
    Query for exact case-insensitive matches on concept synonyms.
    """
    base_stmt = q_concept_synonym().where(
        func.lower(Concept_Synonym.concept_synonym_name) == func.lower(label)
    )
    if search_constraint:
        if not isinstance(search_constraint, SearchConstraintConcept):
            raise TypeError(
                "search_constraint must be an instance of SearchConstraintConcept"
            )
        base_stmt = search_constraint.apply(base_stmt)
    return base_stmt


def q_concept_synonym_ilike(
    label: str, search_constraint: Optional[SearchConstraintConcept] = None
) -> Select:
    """
    Query for partial matches on concept synonyms using ILIKE.
    """
    base_stmt = q_concept_synonym().where(
        Concept_Synonym.concept_synonym_name.ilike(f"%{label}%")
    )
    if search_constraint:
        if not isinstance(search_constraint, SearchConstraintConcept):
            raise TypeError(
                "search_constraint must be an instance of SearchConstraintConcept"
            )
        base_stmt = search_constraint.apply(base_stmt)
    return base_stmt


def q_concept_name_fulltext(
    term: str, search_constraint: Optional[SearchConstraintConcept] = None
) -> Select:
    """
    Query for concept names using PostgreSQL full-text search (tsvector).

    Parameters
    ----------
    term : str
        The search string (passed to plainto_tsquery).
    search_constraint : SearchConstraintConcept, optional
        Additional filters.

    Returns
    -------
    Select
        The query statement ordered by rank.
    """
    vector = func.to_tsvector("english", func.coalesce(Concept.concept_name, ""))
    query = func.plainto_tsquery("english", term)
    stmt = (
        q_concept_name()
        .where(vector.op("@@")(query))  # The Match Operator
        .order_by(func.ts_rank(vector, query).desc())
    )

    if search_constraint:
        stmt = search_constraint.apply(stmt)

    return stmt


def q_concept_synonym_fulltext(
    term: str, search_constraint: Optional[SearchConstraintConcept] = None
) -> Select:
    """
    Query for concept synonyms using PostgreSQL full-text search (tsvector).
    """
    vector = func.to_tsvector(
        "english", func.coalesce(Concept_Synonym.concept_synonym_name, "")
    )
    query = func.plainto_tsquery("english", term)
    stmt = (
        q_concept_synonym()
        .where(vector.op("@@")(query))  # The Match Operator
        .order_by(func.ts_rank(vector, query).desc())
    )

    if search_constraint:
        stmt = search_constraint.apply(stmt)

    return stmt


def q_predicate_name(relationship_id: str) -> Select:
    """Query for the human-readable name of a relationship."""
    return select(Relationship.relationship_name).where(
        Relationship.relationship_id == relationship_id
    )


def q_all_predicates() -> Select:
    """Query for all defined relationships."""
    return select(
        Relationship.relationship_id,
        Relationship.relationship_name,
        Relationship.reverse_relationship_id,
        Relationship.is_hierarchical,
        Relationship.defines_ancestry,
    )


def q_predicate_row(relationship_id: str) -> Select:
    """Query for a specific relationship definition."""
    return select(
        Relationship.relationship_id,
        Relationship.relationship_name,
        Relationship.reverse_relationship_id,
        Relationship.is_hierarchical,
        Relationship.defines_ancestry,
    ).where(Relationship.relationship_id == relationship_id)


def q_predicate_row_with_ancestry(relationship_id: str) -> Select:
    """
    Query a predicate and its reverse to determine directionality.

    This joins the Relationship table with itself to determine if the relationship
    points 'up' (towards ancestors) or 'down' (towards descendants).

    Returns
    -------
    Select
        Columns: relationship_id, relationship_name, reverse_relationship_id,
        is_hierarchical, anc_down, anc_up.
    """
    Rel = Relationship
    Rev = aliased(Relationship)

    return (
        select(
            Rel.relationship_id,
            Rel.relationship_name,
            Rel.reverse_relationship_id,
            Rel.is_hierarchical,
            Rel.defines_ancestry.label("anc_down"),
            Rev.defines_ancestry.label("anc_up"),
        )
        .join(
            Rev, Rel.reverse_relationship_id == Rev.relationship_id
        )  # Match string IDs
        .where(Rel.relationship_id == relationship_id)
    )


def q_all_predicates_with_ancestry() -> Select:
    """Query all predicates with derived ancestry direction flags."""
    Rel = Relationship
    Rev = aliased(Relationship)
    return select(
        Rel.relationship_id,
        Rel.relationship_name,
        Rel.reverse_relationship_id,
        Rel.is_hierarchical,
        Rel.defines_ancestry.label("anc_down"),
        Rev.defines_ancestry.label("anc_up"),
    ).join(Rev, Rel.reverse_relationship_id == Rev.relationship_id)


def q_outgoing_edges(concept_id: int, relationship_id: Optional[str] = None) -> Select:
    """
    Query edges starting from a specific concept ID.

    Parameters
    ----------
    concept_id : int
        The subject concept ID.
    relationship_id : str, optional
        Filter by relationship type.

    Returns
    -------
    Select
        Statement returning edges (subj, rel, obj, validity).
    """
    stmt = select(
        Concept_Relationship.concept_id_1,
        Concept_Relationship.relationship_id,
        Concept_Relationship.concept_id_2,
        Concept_Relationship.valid_start_date,
        Concept_Relationship.valid_end_date,
        Concept_Relationship.invalid_reason,
    ).where(Concept_Relationship.concept_id_1 == concept_id)

    if relationship_id is not None:
        stmt = stmt.where(Concept_Relationship.relationship_id == relationship_id)
    return stmt


def q_outgoing_edges_batch(
    concept_ids: Tuple[int, ...], relationship_id: Optional[str] = None
) -> Select:
    """Query outgoing edges for a batch of concept IDs."""
    stmt = select(
        Concept_Relationship.concept_id_1,
        Concept_Relationship.relationship_id,
        Concept_Relationship.concept_id_2,
        Concept_Relationship.valid_start_date,
        Concept_Relationship.valid_end_date,
        Concept_Relationship.invalid_reason,
    ).where(Concept_Relationship.concept_id_1.in_(concept_ids))

    if relationship_id is not None:
        stmt = stmt.where(Concept_Relationship.relationship_id == relationship_id)
    return stmt


def q_incoming_edges(concept_id: int, relationship_id: Optional[str] = None) -> Select:
    """
    Query edges ending at a specific concept ID.

    Parameters
    ----------
    concept_id : int
        The object concept ID.
    relationship_id : str, optional
        Filter by relationship type.

    Returns
    -------
    Select
        Statement returning edges.
    """
    stmt = select(
        Concept_Relationship.concept_id_1,
        Concept_Relationship.relationship_id,
        Concept_Relationship.concept_id_2,
        Concept_Relationship.valid_start_date,
        Concept_Relationship.valid_end_date,
        Concept_Relationship.invalid_reason,
    ).where(Concept_Relationship.concept_id_2 == concept_id)

    if relationship_id is not None:
        stmt = stmt.where(Concept_Relationship.relationship_id == relationship_id)
    return stmt


def q_incoming_edges_batch(
    concept_ids: Tuple[int, ...], relationship_id: Optional[str] = None
) -> Select:
    """Query incoming edges for a batch of concept IDs."""
    stmt = select(
        Concept_Relationship.concept_id_1,
        Concept_Relationship.relationship_id,
        Concept_Relationship.concept_id_2,
        Concept_Relationship.valid_start_date,
        Concept_Relationship.valid_end_date,
        Concept_Relationship.invalid_reason,
    ).where(Concept_Relationship.concept_id_2.in_(concept_ids))

    if relationship_id is not None:
        stmt = stmt.where(Concept_Relationship.relationship_id == relationship_id)
    return stmt


def q_parents(concept_id: int) -> Select:
    """Query immediate parents (min_levels_of_separation=1)."""
    return select(Concept_Ancestor.ancestor_concept_id).where(
        Concept_Ancestor.descendant_concept_id == concept_id,
        Concept_Ancestor.min_levels_of_separation == 1,
    )


def q_children(concept_id: int) -> Select:
    """Query immediate children (min_levels_of_separation=1)."""
    return select(Concept_Ancestor.descendant_concept_id).where(
        Concept_Ancestor.ancestor_concept_id == concept_id,
        Concept_Ancestor.min_levels_of_separation == 1,
    )


def q_ancestors(concept_id: int) -> Select:
    """Query all ancestors."""
    return select(Concept_Ancestor.ancestor_concept_id).where(
        Concept_Ancestor.descendant_concept_id == concept_id
    )


def q_concept_filtered(
    vocabulary_id: Optional[str] = None, domain_id: Optional[str] = None
) -> Select:
    """Helper query for selecting standard concepts filtered by vocab/domain."""
    stmt = select(Concept.concept_id).where(Concept.standard_concept.is_not(None))
    if domain_id:
        stmt = stmt.where(Concept.domain_id == domain_id)
    if vocabulary_id:
        stmt = stmt.where(Concept.vocabulary_id == vocabulary_id)
    return stmt


def q_concept_synonym_filtered(concept_id: int) -> Select:
    """Query synonyms for a specific concept."""
    stmt = q_concept_synonym()
    return stmt.where(Concept_Synonym.concept_id == concept_id)


def q_singletons(
    *, vocabulary_id: Optional[str] = None, domain_id: Optional[str] = None
) -> Select:
    """
    Query for singleton concepts (no parents, no children).
    Uses NOT EXISTS subqueries on the ancestor table.
    """
    stmt = q_concept_filtered(vocabulary_id=vocabulary_id, domain_id=domain_id)

    return (
        stmt.where(
            ~exists(
                select(1).where(
                    and_(
                        Concept_Ancestor.descendant_concept_id == Concept.concept_id,
                        Concept_Ancestor.min_levels_of_separation == 1,
                    )
                )
            )
        ).where(
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


def q_roots(
    *, vocabulary_id: Optional[str] = None, domain_id: Optional[str] = None
) -> Select:
    """Query for root concepts (no parents)."""
    stmt = q_concept_filtered(vocabulary_id=vocabulary_id, domain_id=domain_id)
    return stmt.where(
        ~exists(
            select(1).where(
                and_(
                    Concept_Ancestor.descendant_concept_id == Concept.concept_id,
                    Concept_Ancestor.min_levels_of_separation == 1,
                )
            )
        )
    )


def q_leaves(
    *, vocabulary_id: Optional[str] = None, domain_id: Optional[str] = None
) -> Select:
    """Query for leaf concepts (no children)."""
    stmt = q_concept_filtered(vocabulary_id=vocabulary_id, domain_id=domain_id)
    return stmt.where(
        ~exists(
            select(1).where(
                and_(
                    Concept_Ancestor.ancestor_concept_id == Concept.concept_id,
                    Concept_Ancestor.min_levels_of_separation == 1,
                )
            )
        )
    )


def q_concept_domain_ids() -> Select:
    """Query distinct domain IDs."""
    return select(Concept.domain_id).distinct().where(Concept.domain_id.is_not(None))


def q_concept_vocabulary_ids() -> Select:
    """Query distinct vocabulary IDs."""
    return (
        select(Concept.vocabulary_id)
        .distinct()
        .where(Concept.vocabulary_id.is_not(None))
    )


def q_concept_potential_ancestor(child_id: int, parent_id: int) -> Select:
    """
    Check if a parent is an ancestor of a child (separation > 1).
    """
    return select(
        Concept_Ancestor.ancestor_concept_id,
        Concept_Ancestor.descendant_concept_id,
        Concept_Ancestor.min_levels_of_separation,
    ).where(
        and_(
            Concept_Ancestor.ancestor_concept_id == parent_id,
            Concept_Ancestor.descendant_concept_id == child_id,
            Concept_Ancestor.min_levels_of_separation > 1,
        )
    )


def q_concept_num_ancestors(concept_ids: Tuple[int, ...]) -> Select:
    """
    Count the number of ancestors for each concept in the batch.
    """
    return (
        select(
            Concept.concept_id,
            func.count(Concept_Ancestor.descendant_concept_id).label("num_ancestors"),
        )
        .join(
            Concept_Ancestor, Concept.concept_id == Concept_Ancestor.ancestor_concept_id
        )
        .where(Concept.concept_id.in_(concept_ids))
        .group_by(Concept.concept_id)
    )

def q_embedding_model_table_name(model_name: str) -> Select:
    """Query to get the table name of an embedding model in the database."""
    return select(ModelRegistry.table_name).where(ModelRegistry.name == model_name)

def q_embedding_cosine_similarity(
    embedding_table: Type[EmbeddingBase],
    text_embedding: list[float],
    concept_ids: Optional[Tuple[int, ...]] = None,
    limit: int = 10
):
    
    distance = embedding_table.embedding.cosine_distance(text_embedding)
    stmt = (
        select(
            Concept.concept_id,
            (1 - distance).label("similarity")
        )
        .join(embedding_table, Concept.concept_id == embedding_table.concept_id)  # type: ignore
        .order_by(distance)
    )

    # 4. Add your optional where clause
    if concept_ids:
        stmt = stmt.where(Concept.concept_id.in_(concept_ids))
        limit = len(concept_ids)

    # Limit the number of results return to the top N most similar concepts
    stmt = stmt.limit(limit)
    return stmt