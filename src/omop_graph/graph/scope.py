"""Helpers that derive one concept scope from another."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

from omop_alchemy.cdm.query import ConceptFilter

ALL_CONCEPTS = ConceptFilter()
"""A scope that selects every concept."""


def every_concept(scope: ConceptFilter) -> ConceptFilter:
    """
    The vocabularies and domains of *scope*, whatever the standardness or validity of a concept.

    Parameters
    ----------
    scope : ConceptFilter
        The scope to widen.

    Returns
    -------
    ConceptFilter
        *scope* without its standard, classification and active requirements.
    """
    return replace(
        scope, require_standard=False, include_classification=False, require_active=False
    )


def narrowed_to_vocabularies(scope: ConceptFilter, vocabularies: Iterable[str]) -> ConceptFilter:
    """
    Restrict *scope* to some vocabularies, keeping only those it already allows.

    Parameters
    ----------
    scope : ConceptFilter
        The scope to narrow.
    vocabularies : Iterable[str]
        Vocabulary ids to keep.

    Returns
    -------
    ConceptFilter
        *scope* with the vocabularies it shares with *vocabularies*, in their order. A scope
        without vocabularies allows all of them.
    """
    return replace(
        scope,
        vocabularies=tuple(
            vocabulary
            for vocabulary in dict.fromkeys(vocabularies)
            if scope.vocabularies is None or vocabulary in scope.vocabularies
        ),
    )
