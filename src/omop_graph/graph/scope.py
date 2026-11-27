"""
Concept filters used as scopes, and the helpers that derive one scope from another.

A ``ConceptFilter`` given to a graph method is a scope, a membership rule that decides which
concepts the method reports or counts. ``limit`` truncates a list and decides no membership,
so a scope that sets it is refused, and the methods that truncate take a ``limit`` parameter.
A scope never changes which concepts a walk passes through. It decides what is reported,
also of the concepts the caller named, and only a relationship asserted to a concept is
reported regardless. 

Important
---------
The ids of a scope are one condition of the query and are never
batched, so a scope with more than about 65,000 ids fails on PostgreSQL.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

from omop_alchemy.cdm.model.vocabulary import Concept
from omop_alchemy.cdm.query import ConceptFilter
from sqlalchemy import Select

ALL_CONCEPTS = ConceptFilter()
"""A scope that selects every concept."""


def _require_scope(concept_filter: ConceptFilter) -> ConceptFilter:
    """
    Check that a concept filter can be used as a scope.

    Parameters
    ----------
    concept_filter : ConceptFilter
        The filter given as a scope.

    Returns
    -------
    ConceptFilter
        The same filter.

    Raises
    ------
    ValueError
        If it sets ``limit``, which truncates a list and so cannot decide membership. The
        methods that truncate take a ``limit`` parameter of their own.
    """
    if concept_filter.limit is not None:
        raise ValueError(
            "A concept_filter is a scope and must not set limit, which truncates a result "
            "and does not decide which concepts belong. Pass limit to the method instead."
        )
    return concept_filter


def _apply_scope(stmt: Select, concept_filter: ConceptFilter) -> Select:
    """
    Restrict a statement to the concepts a scope selects.

    Parameters
    ----------
    stmt : Select
        A statement whose FROM clause includes ``Concept``.
    concept_filter : ConceptFilter
        The scope.

    Returns
    -------
    Select
        The statement with the conditions of the scope.

    Raises
    ------
    ValueError
        If the scope sets ``limit``.
    """
    return _require_scope(concept_filter).apply(stmt)


def limited(stmt: Select, limit: int | None) -> Select:
    """
    Truncate a statement to the concepts with the lowest ids.

    Parameters
    ----------
    stmt : Select
        A statement whose FROM clause includes ``Concept``.
    limit : int, optional
        The most rows to read. None reads them all, in no particular order.

    Returns
    -------
    Select
        The statement ordered by concept id and cut to ``limit`` rows, so the same rows
        are read on every run.
    """
    return stmt if limit is None else stmt.order_by(Concept.concept_id).limit(limit)


def restricted_to_ids(concept_filter: ConceptFilter, concept_ids: Iterable[int]) -> ConceptFilter:
    """
    Restrict a scope to some concepts, keeping only those it already allows.

    Parameters
    ----------
    concept_filter : ConceptFilter
        The scope to restrict. Its own ``concept_ids``, when set, are kept as a limit on the ids.
    concept_ids : Iterable[int]
        The concepts to keep.

    Returns
    -------
    ConceptFilter
        ``concept_filter`` with the ids both name, in the order of concept_ids.
    """
    ids = tuple(dict.fromkeys(concept_ids))
    if concept_filter.concept_ids is not None:
        allowed = set(concept_filter.concept_ids)
        ids = tuple(concept_id for concept_id in ids if concept_id in allowed)
    return replace(concept_filter, concept_ids=ids)


def every_concept(concept_filter: ConceptFilter) -> ConceptFilter:
    """
    The vocabularies and domains of ``concept_filter``, whatever the standardness or validity of a concept.

    Parameters
    ----------
    concept_filter : ConceptFilter
        The scope to widen.

    Returns
    -------
    ConceptFilter
        ``concept_filter`` without its standard, classification and active requirements.
    """
    return replace(
        concept_filter, require_standard=False, include_classification=False, require_active=False
    )


def _narrowed(allowed: tuple[str, ...] | None, requested: Iterable[str]) -> tuple[str, ...]:
    """The requested values that allowed admits, in their order, all of them when it is None."""
    return tuple(value for value in dict.fromkeys(requested) if allowed is None or value in allowed)


def narrowed_to_vocabularies(
    concept_filter: ConceptFilter, vocabularies: Iterable[str]
) -> ConceptFilter:
    """
    Restrict a scope to some vocabularies, keeping only those it already allows.

    Parameters
    ----------
    concept_filter : ConceptFilter
        The scope to narrow.
    vocabularies : Iterable[str]
        Vocabulary ids to keep.

    Returns
    -------
    ConceptFilter
        ``concept_filter`` with the vocabularies it shares with vocabularies, in their order. A scope
        without vocabularies allows all of them.
    """
    return replace(
        concept_filter, vocabularies=_narrowed(concept_filter.vocabularies, vocabularies)
    )


def narrowed_to_domains(concept_filter: ConceptFilter, domains: Iterable[str]) -> ConceptFilter:
    """
    Restrict a scope to some domains, keeping only those it already allows.

    Parameters
    ----------
    concept_filter : ConceptFilter
        The scope to narrow.
    domains : Iterable[str]
        Domain ids to keep.

    Returns
    -------
    ConceptFilter
        ``concept_filter`` with the domains it shares with domains, in their order. A scope
        without domains allows all of them.
    """
    return replace(concept_filter, domains=_narrowed(concept_filter.domains, domains))
