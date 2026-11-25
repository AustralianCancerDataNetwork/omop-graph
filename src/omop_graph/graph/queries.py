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

import re
from collections.abc import Callable
from typing import Literal, cast

from omop_alchemy.backends import (
    resolve_backend,
)
from omop_alchemy.cdm.model import InvalidReasonFlag, normalised_flag_expr
from omop_alchemy.cdm.model.vocabulary import (
    Concept,
    Concept_Ancestor,
    Concept_Relationship,
    Concept_Synonym,
    Domain,
    Relationship,
    Vocabulary,
)
from omop_alchemy.cdm.query import ConceptFilter
from sqlalchemy import (
    Connection,
    Engine,
    Table,
    and_,
    case,
    exists,
    func,
    literal,
    or_,
    select,
    true,
    union_all,
)
from sqlalchemy.orm import aliased
from sqlalchemy.sql import ColumnElement, CompoundSelect, Select

from ..extensions.omop_alchemy import HierarchyDirection, RelationshipMapping
from .edges import EdgeFilter


def _concept_has_standardness_expr():
    """True for concepts carrying a standardness flag: ``'S'`` or ``'C'``.

    Composed locally from omop-alchemy's two atomic predicates. The library
    deliberately publishes only ``is_standard`` (``'S'``) and
    ``is_classification`` (``'C'``); omop-graph is the one consumer that wants
    the union, for hierarchy traversal and match ranking where a classification
    node is a legitimate waypoint even though it is not a valid mapping target.

    Both predicates are two-valued in omop-alchemy >= 1.1, so no ``coalesce``
    compensation is required and ``NOT`` over this returns the true complement.
    """
    return or_(Concept.is_standard_expr(), Concept.is_classification_expr())


def _concept_match_order_terms(name_expr, rank_expr=None):
    """
    Build a stable ordering for concept label matches.

    Ordering logic:
    - Basic ordering
        1. Concepts carrying a standardness flag first
           (``_concept_has_standardness_expr()``: ``'S'`` or ``'C'``, tolerating
           blank/whitespace-only values as unset). Classification concepts share
           tier 0 with standard ones deliberately — this orders *match
           candidates*, where a classification node is a legitimate result, not
           mapping targets.
        2. Active concepts first (``Concept.is_valid_expr()``: invalid_reason unset,
           i.e. NULL or blank/whitespace-only). CDM v5.4 permits only "D"/"U"/NULL,
           for which this matches the previous ``invalid_reason in ("D", "U")``
           test; out-of-spec values now rank as inactive rather than active.
        3. Shorter label length first (LENGTH(name))
        4. Lower concept_id as final tie-breaker
    - If rank_expr is provided (e.g. FTS rank):
        0. rank_expr (e.g. FTS: Highest full-text relevance score (ts_rank))

    This ensures that, for all search types, the most relevant, standard, active, and concise concepts are ranked first.
    """
    terms = []
    if rank_expr is not None:
        terms.append(rank_expr.desc())
    terms.extend(
        (
            case(
                (_concept_has_standardness_expr(), literal(0)),
                else_=literal(1),
            ),
            case(
                (Concept.is_valid_expr(), literal(0)),
                else_=literal(1),
            ),
            func.length(name_expr),
            Concept.concept_id,
        )
    )
    return tuple(terms)


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
        Concept.is_standard_expr().label("standard_concept"),
        Concept.is_classification_expr().label("classification_concept"),
        Concept.valid_start_date,
        Concept.valid_end_date,
        Concept.invalid_reason,
    ).where(Concept.concept_id == concept_id)


def q_concept_views(concept_ids: tuple[int, ...], sort: bool = True) -> Select:
    """
    Query for multiple concepts by their IDs.

    Parameters
    ----------
    concept_ids : tuple[int, ...]
        A tuple of OMOP Concept IDs.
    sort : bool, default True
        If True, preserve the input order using a CASE statement.
        If False, do not add an ORDER BY clause (DB returns arbitrary order).

    Returns
    -------
    Select
        A statement selecting standard concept columns, optionally ordered by input list.
    """
    stmt = select(
        Concept.concept_id,
        Concept.concept_name,
        Concept.concept_code,
        Concept.vocabulary_id,
        Concept.domain_id,
        Concept.concept_class_id,
        Concept.is_standard_expr().label("standard_concept"),
        Concept.is_classification_expr().label("classification_concept"),
        Concept.valid_start_date,
        Concept.valid_end_date,
        Concept.invalid_reason,
    ).where(Concept.concept_id.in_(concept_ids))
    if sort:
        stmt = stmt.order_by(*_concept_match_order_terms(Concept.concept_name))
    return stmt


def q_concepts_by_code(vocabulary_id: str, concept_code: str) -> Select:
    """
    Query the id and validity of every concept with a vocabulary and code.

    Parameters
    ----------
    vocabulary_id : str
        The vocabulary ID (e.g., 'SNOMED').
    concept_code : str
        The source code (e.g., '12345').

    Returns
    -------
    Select
        A statement selecting ``concept_id`` and ``is_valid``, ordered by concept id.
    """
    return (
        select(Concept.concept_id, Concept.is_valid_expr().label("is_valid"))
        .where(
            Concept.vocabulary_id == vocabulary_id,
            Concept.concept_code == concept_code,
        )
        .order_by(Concept.concept_id)
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
        Concept.concept_name.label("name"),
        _concept_has_standardness_expr().label("is_standard"),
        Concept.is_valid_expr().label("is_active"),
    )


def q_concept_synonym() -> Select:
    """
    Base query for concept synonyms joined with concept status.

    Returns
    -------
    Select
        A statement selecting ID, synonym name, and concept status flags.
    """
    return select(
        Concept.concept_id,
        Concept_Synonym.concept_synonym_name.label("name"),
        _concept_has_standardness_expr().label("is_standard"),
        Concept.is_valid_expr().label("is_active"),
    ).join(Concept, Concept.concept_id == Concept_Synonym.concept_id)


def q_concept_name_match(
    query_concept_name: str,
    search_constraint: ConceptFilter | None = None,
    synonym: bool = False,
    sort: bool = True,
) -> Select:
    """
    Query for exact case-insensitive matches on concept names.

    Parameters
    ----------
    query_concept_name : str
        The concept name to match.
    search_constraint : ConceptFilter, optional
        Additional filters (domain, vocab).
    synonym : bool, optional
        Whether to search in synonyms instead of concept names.

    Returns
    -------
    Select
        The query statement.
    """
    name_expr = Concept_Synonym.concept_synonym_name if synonym else Concept.concept_name

    if synonym:
        base_stmt = q_concept_synonym().where(
            func.lower(name_expr) == func.lower(query_concept_name)
        )
    else:
        base_stmt = q_concept_name().where(func.lower(name_expr) == func.lower(query_concept_name))
    if search_constraint:
        if not isinstance(search_constraint, ConceptFilter):
            raise TypeError("search_constraint must be an instance of ConceptFilter")
        base_stmt = search_constraint.apply(base_stmt)
    if sort:
        base_stmt = base_stmt.order_by(*_concept_match_order_terms(name_expr))
    return base_stmt


def q_concept_name_ilike(
    query_concept_name: str,
    search_constraint: ConceptFilter | None = None,
    synonym: bool = False,
    sort: bool = True,
    anchored: bool = False,
) -> Select:
    """
    Query for partial matches on concept names using ILIKE.

    Parameters
    ----------
    query_concept_name : str
        The concept name to search for. ``%``, ``_`` and ``\\`` match themselves.
    search_constraint : ConceptFilter, optional
        Additional filters.
    synonym : bool, optional
        Whether to search in synonyms instead of concept names.
    anchored : bool, optional
        If True, the name must start with the query instead of containing it.

    Returns
    -------
    Select
        The query statement.
    """
    name_expr = Concept_Synonym.concept_synonym_name if synonym else Concept.concept_name

    escaped = re.sub(r"([\\%_])", r"\\\1", query_concept_name)
    pattern = f"{escaped}%" if anchored else f"%{escaped}%"
    base_stmt = (q_concept_synonym() if synonym else q_concept_name()).where(
        name_expr.ilike(pattern, escape="\\")
    )
    if search_constraint:
        if not isinstance(search_constraint, ConceptFilter):
            raise TypeError("search_constraint must be an instance of ConceptFilter")
        base_stmt = search_constraint.apply(base_stmt)
    if sort:
        base_stmt = base_stmt.order_by(*_concept_match_order_terms(name_expr))
    return base_stmt


def q_concept_name_fulltext(
    query_concept_name: str,
    *,
    engine: Engine | Connection,
    search_constraint: ConceptFilter | None = None,
    synonym: bool = False,
    sort: bool = True,
) -> Select:
    """
    Query for concept names using PostgreSQL full-text search via the
    pre-computed tsvector columns and GIN indices.

    Requires the stored tsvector columns from ``omop-alchemy fulltext install``
    and ``omop-alchemy fulltext populate``. If they are absent, this raises
    ``FullTextError`` instead of falling back to on-demand tsvector
    generation.

    Parameters
    ----------
    query_concept_name : str
        The concept name to search for.
    engine : Engine or Connection
        Bindable whose schema_translate_map locates the vocabulary schema.
    search_constraint : ConceptFilter, optional
        Additional filters (domain, vocab).
    synonym : bool, optional
        Whether to search in synonyms instead of concept names.
    sort : bool, optional
        Whether to order by match quality.

    Raises
    ------
    FullTextError
        If the stored tsvector column is missing.
    """
    name_expr = Concept_Synonym.concept_synonym_name if synonym else Concept.concept_name
    target_table = Concept_Synonym if synonym else Concept
    stmt = q_concept_synonym() if synonym else q_concept_name()
    vector = resolve_backend(engine).fulltext_vector_column(
        engine, cast(Table, target_table.__table__)
    )
    query = func.plainto_tsquery("english", query_concept_name)

    stmt = stmt.where(vector.op("@@")(query))  # Hits the GIN index instantly

    if search_constraint:
        stmt = search_constraint.apply(stmt)

    if sort:
        stmt = stmt.order_by(
            *_concept_match_order_terms(name_expr, rank_expr=func.ts_rank(vector, query))
        )

    return stmt


def q_predicate_name(relationship_id: str) -> Select:
    """Query for the human-readable name of a relationship."""
    return select(Relationship.relationship_name).where(
        Relationship.relationship_id == relationship_id
    )


def q_all_predicates() -> Select:
    """Query all defined relationships."""
    return select(
        Relationship.relationship_id,
        Relationship.relationship_name,
        Relationship.reverse_relationship_id,
    )


def q_predicate_row(relationship_id: str) -> Select:
    """Query for a specific relationship definition."""
    return q_all_predicates().where(Relationship.relationship_id == relationship_id)


def _filter_concept_relationships(stmt: Select, edge_filter: EdgeFilter) -> Select:
    """Apply the validity, date, domain and relationship id conditions of *edge_filter*.

    The kind and subkind fields are not applied here: the caller resolves them to
    ``predicate_ids`` first, since the classification is not in the SQL tables.
    """
    if edge_filter.active_only:
        stmt = stmt.where(Concept_Relationship.is_valid_expr())

    if edge_filter.on is not None:
        stmt = stmt.where(
            and_(
                Concept_Relationship.valid_start_date <= edge_filter.on,
                Concept_Relationship.valid_end_date >= edge_filter.on,
            )
        )

    if edge_filter.within_domain:
        Subj = aliased(Concept)
        Obj = aliased(Concept)
        stmt = stmt.join(Subj, Concept_Relationship.concept_id_1 == Subj.concept_id)
        stmt = stmt.join(Obj, Concept_Relationship.concept_id_2 == Obj.concept_id)
        stmt = stmt.where(Subj.domain_id == Obj.domain_id)

    if edge_filter.predicate_ids is not None:
        stmt = stmt.where(Concept_Relationship.relationship_id.in_(edge_filter.predicate_ids))

    return stmt


def q_edges(
    concept_ids: tuple[int, ...] | int,
    direction: Literal["in", "out"],
    edge_filter: EdgeFilter,
    ordered: bool = False,
) -> Select:
    """Query edges for a batch of concept IDs, in either direction.

    Parameters
    ----------
    direction : {"in", "out"}
        Whether to query incoming or outgoing edges for ``concept_ids``.
    edge_filter : EdgeFilter
        Conditions on the edges. Its kind and subkind fields are ignored, see
        :func:`_filter_concept_relationships`.
    ordered : bool, default=False
        If True, order by source concept, target concept, and relationship ID
        to make results reproducible. If False, row order is unspecified and
        may differ between runs and backends.
    """
    if isinstance(concept_ids, int):
        concept_ids = (concept_ids,)

    stmt = select(
        Concept_Relationship.concept_id_1.label("subject_id"),
        Concept_Relationship.relationship_id.label("predicate_id"),
        Concept_Relationship.concept_id_2.label("object_id"),
        Concept_Relationship.valid_start_date,
        Concept_Relationship.valid_end_date,
        Concept_Relationship.invalid_reason,
    )
    stmt = _filter_concept_relationships(stmt, edge_filter)

    if direction == "in":
        stmt = stmt.where(Concept_Relationship.concept_id_2.in_(concept_ids))
    elif direction == "out":
        stmt = stmt.where(Concept_Relationship.concept_id_1.in_(concept_ids))

    if ordered:
        stmt = stmt.order_by(
            Concept_Relationship.concept_id_1,
            Concept_Relationship.concept_id_2,
            Concept_Relationship.relationship_id,
        )

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


def q_closure(
    concept_ids: tuple[int, ...], direction: HierarchyDirection, edge_filter: EdgeFilter
) -> Select:
    """
    Query the transitive closure of the edges of *edge_filter* as (origin, reached) pairs.

    Follows the selected edges repeatedly from each concept. ``UP`` goes from the subject of
    an edge to its object, as from a concept to its ``Is a`` ancestors, and ``DOWN`` the
    other way. A concept is never reached from itself. Its kind and subkind fields are
    ignored, see :func:`_filter_concept_relationships`.
    """
    near, far = (
        (Concept_Relationship.concept_id_1, Concept_Relationship.concept_id_2)
        if direction is HierarchyDirection.UP
        else (Concept_Relationship.concept_id_2, Concept_Relationship.concept_id_1)
    )
    seed = _filter_concept_relationships(
        select(near.label("origin"), far.label("reached")).where(near.in_(concept_ids)),
        edge_filter,
    )
    closure = seed.cte("closure", recursive=True)
    step = _filter_concept_relationships(
        select(closure.c.origin, far)
        .select_from(closure)
        .join(Concept_Relationship, near == closure.c.reached),
        edge_filter,
    )
    closure = closure.union(step)
    return select(closure.c.origin, closure.c.reached).where(closure.c.origin != closure.c.reached)


def q_mappings(
    concept_ids: tuple[int, ...] | None, edge_filter: EdgeFilter, source: str | None = None
) -> Select | CompoundSelect:
    """
    Query the edges of *edge_filter* with the vocabularies of their concepts.

    Parameters
    ----------
    concept_ids : tuple[int, ...] | None
        Only edges starting from these concepts, or all edges when None.
    edge_filter : EdgeFilter
        Conditions on the edges, see :func:`q_edges`. Edges from a concept to itself are
        left out.
    source : str, optional
        Only edges with this vocabulary on either side. It is two queries, one for each
        side, since a condition on either side of the join cannot use an index.
    """

    def edges(side: Callable[[type[Concept], type[Concept]], ColumnElement[bool]]) -> Select:
        Subject = aliased(Concept)
        Object = aliased(Concept)
        stmt = (
            select(
                Concept_Relationship.concept_id_1.label("subject_id"),
                Concept_Relationship.relationship_id.label("predicate_id"),
                Concept_Relationship.concept_id_2.label("object_id"),
                Concept_Relationship.valid_start_date,
                Subject.vocabulary_id.label("subject_source"),
                Object.vocabulary_id.label("object_source"),
            )
            .join(Subject, Subject.concept_id == Concept_Relationship.concept_id_1)
            .join(Object, Object.concept_id == Concept_Relationship.concept_id_2)
            .where(Concept_Relationship.concept_id_1 != Concept_Relationship.concept_id_2)
            .where(side(Subject, Object))
        )
        stmt = _filter_concept_relationships(stmt, edge_filter)
        if concept_ids is not None:
            stmt = stmt.where(Concept_Relationship.concept_id_1.in_(concept_ids))
        return stmt

    if source is None:
        return edges(lambda subject, object_: true())
    return union_all(
        edges(lambda subject, object_: subject.vocabulary_id == source),
        edges(
            lambda subject, object_: and_(
                object_.vocabulary_id == source, subject.vocabulary_id != source
            )
        ),
    )


def q_relationships(
    subjects: tuple[int, ...] | None,
    objects: tuple[int, ...] | None,
    edge_filter: EdgeFilter,
) -> Select:
    """Query (subject, relationship_id, object) triples. See :func:`q_edges` for *edge_filter*."""
    stmt = select(
        Concept_Relationship.concept_id_1,
        Concept_Relationship.relationship_id,
        Concept_Relationship.concept_id_2,
    )
    stmt = _filter_concept_relationships(stmt, edge_filter)

    if subjects is not None:
        stmt = stmt.where(Concept_Relationship.concept_id_1.in_(subjects))

    if objects is not None:
        stmt = stmt.where(Concept_Relationship.concept_id_2.in_(objects))

    return stmt


DEFAULT_CONCEPT_FILTER = ConceptFilter(
    require_standard=True, include_classification=True, require_active=True
)
"""Concepts the graph enumerators list by default: active, standard or classification."""


def q_entities(concept_filter: ConceptFilter = DEFAULT_CONCEPT_FILTER) -> Select:
    """Query concept IDs using OMOP Alchemy's canonical concept semantics."""
    return concept_filter.apply(select(Concept.concept_id))


def q_synonyms(concept_ids: tuple[int, ...]) -> Select:
    """Query the synonyms of *concept_ids* with their language concept, ordered by name."""
    return (
        select(
            Concept_Synonym.concept_id,
            Concept_Synonym.concept_synonym_name.label("name"),
            Concept_Synonym.language_concept_id,
        )
        .where(Concept_Synonym.concept_id.in_(concept_ids))
        .order_by(Concept_Synonym.concept_id, Concept_Synonym.concept_synonym_name)
    )


def q_synonym_language_ids() -> Select:
    """Query the distinct language concept ids used by synonyms."""
    return select(Concept_Synonym.language_concept_id).distinct()


def q_vocabularies() -> Select:
    """Query all vocabularies."""
    return select(
        Vocabulary.vocabulary_id,
        Vocabulary.vocabulary_name,
        Vocabulary.vocabulary_reference,
        Vocabulary.vocabulary_version,
        Vocabulary.vocabulary_concept_id,
    )


def q_domains() -> Select:
    """Query all domains with their concept id."""
    return select(Domain.domain_id, Domain.domain_concept_id)


def q_obsolete_concepts(
    include_updated: bool = True, concept_filter: ConceptFilter | None = None
) -> Select:
    """
    Query the ids of deleted concepts, and of updated ones when *include_updated*.

    A *concept_filter* narrows them, by its vocabularies and domains and its other fields.
    """
    reasons = [InvalidReasonFlag.DELETED]
    if include_updated:
        reasons.append(InvalidReasonFlag.UPDATED)
    stmt = select(Concept.concept_id).where(
        normalised_flag_expr(Concept.invalid_reason).in_(reasons)
    )
    return concept_filter.apply(stmt) if concept_filter else stmt


def _has_parent(edge_filter: EdgeFilter | None) -> ColumnElement[bool]:
    """Correlated test that a concept has a parent: in the closure, or by *edge_filter* edges."""
    if edge_filter is None:
        return exists(
            select(1).where(
                Concept_Ancestor.descendant_concept_id == Concept.concept_id,
                Concept_Ancestor.min_levels_of_separation == 1,
            )
        )
    return exists(
        _filter_concept_relationships(
            select(1)
            .select_from(Concept_Relationship)
            .where(Concept_Relationship.concept_id_1 == Concept.concept_id),
            edge_filter,
        )
    )


def _has_child(edge_filter: EdgeFilter | None) -> ColumnElement[bool]:
    """Correlated test that a concept has a child: in the closure, or by *edge_filter* edges."""
    if edge_filter is None:
        return exists(
            select(1).where(
                Concept_Ancestor.ancestor_concept_id == Concept.concept_id,
                Concept_Ancestor.min_levels_of_separation == 1,
            )
        )
    return exists(
        _filter_concept_relationships(
            select(1)
            .select_from(Concept_Relationship)
            .where(Concept_Relationship.concept_id_2 == Concept.concept_id),
            edge_filter,
        )
    )


def q_roots(
    concept_filter: ConceptFilter = DEFAULT_CONCEPT_FILTER, edge_filter: EdgeFilter | None = None
) -> Select:
    """
    Query for root concepts (no parents).

    Parents are the closure's level 1 ancestors, or with *edge_filter* the targets of
    outgoing edges it selects.
    """
    return q_entities(concept_filter).where(~_has_parent(edge_filter))


def q_leaves(
    concept_filter: ConceptFilter = DEFAULT_CONCEPT_FILTER, edge_filter: EdgeFilter | None = None
) -> Select:
    """
    Query for leaf concepts (no children).

    Children are the closure's level 1 descendants, or with *edge_filter* the sources of
    incoming edges it selects.
    """
    return q_entities(concept_filter).where(~_has_child(edge_filter))


def q_singletons(
    concept_filter: ConceptFilter = DEFAULT_CONCEPT_FILTER, edge_filter: EdgeFilter | None = None
) -> Select:
    """Query for singleton concepts (no parents and no children). See :func:`q_roots`."""
    return q_entities(concept_filter).where(~_has_parent(edge_filter), ~_has_child(edge_filter))


def q_concept_domain_ids() -> Select:
    """Query distinct domain IDs."""
    return select(Concept.domain_id).distinct().where(Concept.domain_id.is_not(None))


def q_concept_vocabulary_ids() -> Select:
    """Query distinct vocabulary IDs."""
    return select(Concept.vocabulary_id).distinct().where(Concept.vocabulary_id.is_not(None))


def q_concept_potential_ancestor(child_id: int, parent_id: int) -> Select:
    """
    Check if a parent is an ancestor of a child (including immediate parent).
    """
    return select(
        Concept_Ancestor.ancestor_concept_id,
        Concept_Ancestor.descendant_concept_id,
        Concept_Ancestor.min_levels_of_separation,
    ).where(
        and_(
            Concept_Ancestor.ancestor_concept_id == parent_id,
            Concept_Ancestor.descendant_concept_id == child_id,
            Concept_Ancestor.min_levels_of_separation > 0,
        )
    )


def q_concept_potential_ancestors_batch(
    child_ids: tuple[int, ...], parent_ids: tuple[int, ...]
) -> Select:
    """
    Check which of several candidate parents are ancestors of one or more children.

    Parameters
    ----------
    child_ids : tuple of int
        A tuple of descendant concept IDs for batch mode.
    parent_ids : tuple of int
        Candidate ancestor concept IDs to check.
    """
    return select(
        Concept_Ancestor.ancestor_concept_id,
        Concept_Ancestor.descendant_concept_id,
        Concept_Ancestor.min_levels_of_separation,
    ).where(
        and_(
            Concept_Ancestor.ancestor_concept_id.in_(parent_ids),
            Concept_Ancestor.descendant_concept_id.in_(child_ids),
            Concept_Ancestor.min_levels_of_separation > 0,
        )
    )


def q_concept_num_ancestors(concept_ids: tuple[int, ...]) -> Select:
    """
    Count the number of ancestors for each concept in the batch.
    """
    return (
        select(
            Concept.concept_id,
            func.count(Concept_Ancestor.ancestor_concept_id).label("num_ancestors"),
        )
        .join(
            Concept_Ancestor,
            Concept.concept_id == Concept_Ancestor.descendant_concept_id,
        )
        .where(Concept.concept_id.in_(concept_ids))
        .where(Concept_Ancestor.min_levels_of_separation > 0)
        .group_by(Concept.concept_id)
    )


def q_concept_num_descendants(concept_ids: tuple[int, ...]) -> Select:
    """
    Count the number of descendants for each concept in the batch.
    """
    return (
        select(
            Concept.concept_id,
            func.count(Concept_Ancestor.descendant_concept_id).label("num_descendants"),
        )
        .join(Concept_Ancestor, Concept.concept_id == Concept_Ancestor.ancestor_concept_id)
        .where(Concept.concept_id.in_(concept_ids))
        .where(Concept_Ancestor.min_levels_of_separation > 0)
        .group_by(Concept.concept_id)
    )


def q_relationship_class(relationship_id: str) -> Select:
    return select(RelationshipMapping.predicate_kind).where(
        RelationshipMapping.relationship_id == relationship_id
    )
