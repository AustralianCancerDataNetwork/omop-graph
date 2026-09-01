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

from typing import Optional, Tuple, Literal, Union
from datetime import date

from sqlalchemy import (
    and_,
    case,
    exists,
    func,
    literal,
    or_,
    select,
    Engine,
    column,
)
from sqlalchemy.orm import aliased
from sqlalchemy.sql import Select

from oa_configurator import schema_inspect

from omop_alchemy.backends import (
    CONCEPT_NAME_TSVECTOR_COLUMN,
    CONCEPT_SYNONYM_NAME_TSVECTOR_COLUMN,
    FullTextError,
)
from omop_alchemy.cdm.model.vocabulary import (
    Concept,
    Concept_Ancestor,
    Concept_Relationship,
    Concept_Synonym,
    Relationship,
)
from omop_alchemy.cdm.query import ConceptFilter

from ..extensions.omop_alchemy import RelationshipMapping, PredicateKind


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


def q_concept_views(concept_ids: Tuple[int, ...], sort: bool = True) -> Select:
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
    return select(Concept.concept_id).where(
        Concept.vocabulary_id == vocabulary_id,
        Concept.concept_code == concept_code,
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
    search_constraint: Optional[ConceptFilter] = None,
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
    name_expr = (
        Concept_Synonym.concept_synonym_name if synonym else Concept.concept_name
    )

    if synonym:
        base_stmt = q_concept_synonym().where(
            func.lower(name_expr) == func.lower(query_concept_name)
        )
    else:
        base_stmt = q_concept_name().where(
            func.lower(name_expr) == func.lower(query_concept_name)
        )
    if search_constraint:
        if not isinstance(search_constraint, ConceptFilter):
            raise TypeError(
                "search_constraint must be an instance of ConceptFilter"
            )
        base_stmt = search_constraint.apply(base_stmt)
    if sort:
        base_stmt = base_stmt.order_by(*_concept_match_order_terms(name_expr))
    return base_stmt


def q_concept_name_ilike(
    query_concept_name: str,
    search_constraint: Optional[ConceptFilter] = None,
    synonym: bool = False,
    sort: bool = True,
) -> Select:
    """
    Query for partial matches on concept names using ILIKE.

    Parameters
    ----------
    query_concept_name : str
        The concept name to search for.
    search_constraint : ConceptFilter, optional
        Additional filters.
    synonym : bool, optional
        Whether to search in synonyms instead of concept names.

    Returns
    -------
    Select
        The query statement.
    """
    name_expr = (
        Concept_Synonym.concept_synonym_name if synonym else Concept.concept_name
    )

    if "%" in query_concept_name:
        raise ValueError("query_concept_name should not contain wildcards like '%'.")

    if synonym:
        base_stmt = q_concept_synonym().where(
            name_expr.ilike(f"%{query_concept_name}%")
        )
    else:
        base_stmt = q_concept_name().where(name_expr.ilike(f"%{query_concept_name}%"))
    if search_constraint:
        if not isinstance(search_constraint, ConceptFilter):
            raise TypeError(
                "search_constraint must be an instance of ConceptFilter"
            )
        base_stmt = search_constraint.apply(base_stmt)
    if sort:
        base_stmt = base_stmt.order_by(*_concept_match_order_terms(name_expr))
    return base_stmt


def q_concept_name_fulltext(
    query_concept_name: str,
    *,
    engine: Engine,
    search_constraint: Optional[ConceptFilter] = None,
    synonym: bool = False,
    sort: bool = True,
) -> Select:
    """
    Query for concept names using PostgreSQL full-text search via optional
    pre-computed tsvector columns and GIN indices.

    This query only works when the stored tsvector columns have been installed
    and registered in the ORM metadata via ``omop-maint fulltext install`` and
    ``omop-maint fulltext populate``. If those columns are absent, this raises
    ``FullTextError`` instead of falling back to on-demand tsvector
    generation.

    Parameters
    ----------
    query_concept_name : str
        The concept name to search for.
    search_constraint : ConceptFilter, optional
        Additional filters (domain, vocab).
    synonym : bool, optional
        Whether to search in synonyms instead of concept names.

    """
    name_expr = (
        Concept_Synonym.concept_synonym_name if synonym else Concept.concept_name
    )

    inspector = schema_inspect(engine)
    target_table = Concept_Synonym if synonym else Concept
    target_col = (
        CONCEPT_SYNONYM_NAME_TSVECTOR_COLUMN
        if synonym
        else CONCEPT_NAME_TSVECTOR_COLUMN
    )
    stmt = q_concept_synonym() if synonym else q_concept_name()

    tsvector_col = next(
        (
            c["name"]
            for c in inspector.get_columns(target_table.__tablename__)
            if c["name"] == target_col
        ),
        None,
    )

    if tsvector_col is None:
        raise FullTextError(
            f"Full-text search column '{target_col}' not found in table '{target_table.__tablename__}'. "
            "Make sure to run 'omop-maint fulltext install' and 'omop-maint fulltext populate' to set up full-text search."
        )

    vector = column(tsvector_col)
    query = func.plainto_tsquery("english", query_concept_name)

    stmt = stmt.where(vector.op("@@")(query))  # Hits the GIN index instantly

    if search_constraint:
        stmt = search_constraint.apply(stmt)

    if sort:
        stmt = stmt.order_by(
            *_concept_match_order_terms(
                name_expr, rank_expr=func.ts_rank(vector, query)
            )
        )

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
        Relationship.is_hierarchical_relationship_expr().label("is_hierarchical"),
        Relationship.is_ancestry_defining_expr().label("defines_ancestry"),
    )


def q_predicate_row(relationship_id: str) -> Select:
    """Query for a specific relationship definition."""
    return select(
        Relationship.relationship_id,
        Relationship.relationship_name,
        Relationship.reverse_relationship_id,
        Relationship.is_hierarchical_relationship_expr().label("is_hierarchical"),
        Relationship.is_ancestry_defining_expr().label("defines_ancestry"),
    ).where(Relationship.relationship_id == relationship_id)


def q_predicate_row_with_ancestry(
    relationship_id: str, *, include_classification: bool = True
) -> Select:
    """
    Query a predicate and its reverse to determine directionality.

    This joins the Relationship table with itself to determine if the relationship
    points 'up' (towards ancestors) or 'down' (towards descendants).

    Parameters
    ----------
    include_classification : bool, optional
        Join in RelationshipMapping's predicate_kind/predicate_subkind. Set to
        False for a split-connection deployment (Relationship is vocab-role,
        RelationshipMapping is not, so they can live on different physical
        connections). The caller fetches RelationshipMapping separately via
        :func:`q_relationship_mapping_row` and merges in Python.

    Returns
    -------
    Select
        Columns: relationship_id, relationship_name, reverse_relationship_id,
        is_hierarchical, anc_down, anc_up, plus predicate_kind/predicate_subkind
        when include_classification is True.
    """
    Rel = Relationship
    Rev = aliased(Relationship)

    stmt = select(
        Rel.relationship_id,
        Rel.relationship_name,
        Rel.reverse_relationship_id,
        Rel.is_hierarchical_relationship_expr().label("is_hierarchical"),
        Rel.is_ancestry_defining_expr().label("anc_down"),
        Rev.is_ancestry_defining_expr().label("anc_up"),
    ).join(
        Rev,
        Rel.reverse_relationship_id == Rev.relationship_id,
    )

    if include_classification:
        Rm = aliased(RelationshipMapping)
        stmt = stmt.add_columns(Rm.predicate_kind, Rm.predicate_subkind).join(
            Rm, Rel.relationship_id == Rm.relationship_id
        )

    return stmt.where(Rel.relationship_id == relationship_id)


def q_all_predicates_with_ancestry(*, include_classification: bool = True) -> Select:
    """Query all predicates with derived ancestry direction flags and classification.

    Parameters
    ----------
    include_classification : bool, optional
        See :func:`q_predicate_row_with_ancestry`.
    """
    Rel = Relationship
    Rev = aliased(Relationship)

    stmt = select(
        Rel.relationship_id,
        Rel.relationship_name,
        Rel.reverse_relationship_id,
        Rel.is_hierarchical_relationship_expr().label("is_hierarchical"),
        Rel.is_ancestry_defining_expr().label("anc_down"),
        Rev.is_ancestry_defining_expr().label("anc_up"),
    ).join(Rev, Rel.reverse_relationship_id == Rev.relationship_id)

    if include_classification:
        Rm = aliased(RelationshipMapping)
        stmt = stmt.add_columns(Rm.predicate_kind, Rm.predicate_subkind).join(
            Rm, Rel.relationship_id == Rm.relationship_id
        )

    return stmt


def q_relationship_mapping_row(relationship_id: str) -> Select:
    """Query one RelationshipMapping row by relationship_id.

    The primary-role half of a split-connection predicate lookup, pairing
    with :func:`q_predicate_row_with_ancestry`'s ``include_classification=False``.
    """
    return select(
        RelationshipMapping.relationship_id,
        RelationshipMapping.predicate_kind,
        RelationshipMapping.predicate_subkind,
    ).where(RelationshipMapping.relationship_id == relationship_id)


def q_relationship_mapping_all() -> Select:
    """Query every RelationshipMapping row, keyed by relationship_id.

    The primary-role half of a split-connection edges/predicates lookup.
    RelationshipMapping is a small reference table, so callers merge it as a
    plain dict rather than joining across connections.
    """
    return select(
        RelationshipMapping.relationship_id,
        RelationshipMapping.predicate_kind,
        RelationshipMapping.predicate_subkind,
    )


def q_edges(
    concept_ids: Union[Tuple[int, ...], int],
    direction: Literal["in", "out"],
    predicate_ids: Optional[frozenset[str]] = None,
    predicate_kinds: Optional[frozenset[PredicateKind]] = None,
    active_only: bool = False,
    on: Optional[date] = None,
    within_domain: bool = False,
    include_classification: bool = True,
) -> Select:
    """Query outgoing edges for a batch of concept IDs.

    Parameters
    ----------
    include_classification : bool, optional
        Join in RelationshipMapping's predicate_kind/predicate_subkind.
        Concept_Relationship is vocab-role, RelationshipMapping is not, so
        for a split-connection deployment set this to False and merge
        RelationshipMapping (via :func:`q_relationship_mapping_all`)
        in Python instead. ``predicate_kinds`` cannot be applied in SQL
        when this is False (the column isn't joined); the caller must
        filter after merging.
    """
    if isinstance(concept_ids, int):
        concept_ids = (concept_ids,)

    if not include_classification and predicate_kinds:
        raise ValueError(
            "predicate_kinds requires include_classification=True; filter "
            "after merging RelationshipMapping in Python instead."
        )

    Subj = aliased(Concept)
    Obj = aliased(Concept)

    stmt = select(
        Concept_Relationship.concept_id_1.label("subject_id"),
        Concept_Relationship.relationship_id.label("predicate_id"),
        Concept_Relationship.concept_id_2.label("object_id"),
        Concept_Relationship.valid_start_date,
        Concept_Relationship.valid_end_date,
        Concept_Relationship.invalid_reason,
    )

    if include_classification:
        stmt = stmt.add_columns(
            RelationshipMapping.predicate_kind, RelationshipMapping.predicate_subkind
        ).join(
            RelationshipMapping,
            Concept_Relationship.relationship_id
            == RelationshipMapping.relationship_id,
        )

    if active_only:
        stmt = stmt.where(Concept_Relationship.is_valid_expr())
        if on is not None:
            stmt = stmt.where(
                and_(
                    Concept_Relationship.valid_start_date <= on,
                    Concept_Relationship.valid_end_date >= on,
                )
            )

    if within_domain:
        stmt = stmt.join(Subj, Concept_Relationship.concept_id_1 == Subj.concept_id)
        stmt = stmt.join(Obj, Concept_Relationship.concept_id_2 == Obj.concept_id)
        stmt = stmt.where(Subj.domain_id == Obj.domain_id)

    if direction == "in":
        stmt = stmt.where(Concept_Relationship.concept_id_2.in_(concept_ids))
    elif direction == "out":
        stmt = stmt.where(Concept_Relationship.concept_id_1.in_(concept_ids))

    if predicate_ids:  # Exact ID's
        stmt = stmt.where(Concept_Relationship.relationship_id.in_(predicate_ids))
    if predicate_kinds:  # Global categories
        stmt = stmt.where(RelationshipMapping.predicate_kind.in_(predicate_kinds))

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


def q_relationships(
    subjects: Optional[tuple[int, ...]],
    predicates: Optional[tuple[str, ...]],
    objects: Optional[tuple[int, ...]],
) -> Select:

    stmt = select(
        Concept_Relationship.concept_id_1,
        Concept_Relationship.relationship_id,
        Concept_Relationship.concept_id_2,
    )

    if subjects:
        stmt = stmt.where(Concept_Relationship.concept_id_1.in_(subjects))

    if predicates:
        stmt = stmt.where(Concept_Relationship.relationship_id.in_(predicates))

    if objects:
        stmt = stmt.where(Concept_Relationship.concept_id_2.in_(objects))

    return stmt


def q_entities(
    domain: str | None, standard_only: bool = True, filter_obsoletes: bool = True
) -> Select:
    """Query concept IDs using OMOP Alchemy's canonical concept semantics."""
    return ConceptFilter(
        domains=(domain,) if domain else None,
        require_standard=standard_only,
        include_classification=True,
        require_active=filter_obsoletes,
    ).apply(select(Concept.concept_id))


def q_concept_filtered(
    vocabulary_id: Optional[str] = None, domain_id: Optional[str] = None
) -> Select:
    """Helper query for selecting standard concepts filtered by vocab/domain."""
    return ConceptFilter(
        domains=(domain_id,) if domain_id else None,
        vocabularies=(vocabulary_id,) if vocabulary_id else None,
        require_standard=True,
        include_classification=True,
    ).apply(select(Concept.concept_id))


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

    return stmt.where(
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
    child_ids: Tuple[int, ...], parent_ids: Tuple[int, ...]
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


def q_concept_num_ancestors(concept_ids: Tuple[int, ...]) -> Select:
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


def q_concept_num_descendants(concept_ids: Tuple[int, ...]) -> Select:
    """
    Count the number of descendants for each concept in the batch.
    """
    return (
        select(
            Concept.concept_id,
            func.count(Concept_Ancestor.descendant_concept_id).label("num_descendants"),
        )
        .join(
            Concept_Ancestor, Concept.concept_id == Concept_Ancestor.ancestor_concept_id
        )
        .where(Concept.concept_id.in_(concept_ids))
        .where(Concept_Ancestor.min_levels_of_separation > 0)
        .group_by(Concept.concept_id)
    )


def q_relationship_class(relationship_id: str) -> Select:
    return select(RelationshipMapping.predicate_kind).where(
        RelationshipMapping.relationship_id == relationship_id
    )
