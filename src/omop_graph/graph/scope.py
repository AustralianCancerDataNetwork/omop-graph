"""
Concept filters used as scopes, and the helpers that derive one scope from another.

A ``ConceptFilter`` plays two roles. As a query, it selects and truncates concepts, so
``concept_ids`` and ``limit`` both mean something. As a scope, it is a membership rule that
decides which concepts a method reports or counts, so ``limit``, which truncates a list,
means nothing. A scope never changes which concepts a walk passes through, and never removes
a concept the caller asked about.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Iterable
from dataclasses import replace

from omop_alchemy.cdm.query import ConceptFilter

ALL_CONCEPTS = ConceptFilter()
"""A scope that selects every concept."""

SCOPE_ONLY_MARKER = "__scope_only__"
"""Attribute set on the methods that take their ``concept_filter`` as a scope."""


def require_scope(concept_filter: ConceptFilter) -> ConceptFilter:
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
        If it sets ``limit``, which truncates a list and so cannot decide membership.
    """
    if concept_filter.limit is not None:
        raise ValueError(
            "A concept_filter used as a scope must not set limit, which truncates a result "
            "and does not decide which concepts belong."
        )
    return concept_filter


def scope_only[**P, R](method: Callable[P, R]) -> Callable[P, R]:
    """
    Mark a method whose ``concept_filter`` argument is a scope, and check it on every call.

    The check runs when the method is called, also for a generator method, so a bad scope
    fails at the call and not at the first item.
    """
    signature = inspect.signature(method)

    @functools.wraps(method)
    def checked(*args: P.args, **kwargs: P.kwargs) -> R:
        value = signature.bind(*args, **kwargs).arguments.get("concept_filter")
        if value is not None:
            require_scope(value)
        return method(*args, **kwargs)

    setattr(checked, SCOPE_ONLY_MARKER, True)
    return checked


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
        ``concept_filter`` with the ids both name, in the order of *concept_ids*.
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
    """The requested values that *allowed* admits, in their order, all of them when it is None."""
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
        ``concept_filter`` with the vocabularies it shares with *vocabularies*, in their order. A scope
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
        ``concept_filter`` with the domains it shares with *domains*, in their order. A scope
        without domains allows all of them.
    """
    return replace(concept_filter, domains=_narrowed(concept_filter.domains, domains))
