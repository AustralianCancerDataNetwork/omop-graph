"""Helpers that derive one concept scope from another."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

from omop_alchemy.cdm.query import ConceptFilter

ALL_CONCEPTS = ConceptFilter()
"""A scope that selects every concept."""


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


def narrowed_to_vocabularies(
    concept_filter: ConceptFilter, vocabularies: Iterable[str]
) -> ConceptFilter:
    """
    Restrict ``concept_filter`` to some vocabularies, keeping only those it already allows.

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
        concept_filter,
        vocabularies=tuple(
            vocabulary
            for vocabulary in dict.fromkeys(vocabularies)
            if concept_filter.vocabularies is None or vocabulary in concept_filter.vocabularies
        ),
    )
