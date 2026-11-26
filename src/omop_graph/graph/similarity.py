"""Types for the structural similarity of concepts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ICSource(Enum):
    """Where the descendants of a concept are counted for its information content."""

    IS_A = "is_a"
    """Walk the strict ``Is a`` hierarchy within the scope. Exact, and costly for broad concepts."""

    CONCEPT_ANCESTOR = "concept_ancestor"
    """Read the precomputed ``concept_ancestor`` table. Fast in bulk, but it also holds
    classification and cohort ancestry, and ignores the scope."""


@dataclass(frozen=True, slots=True)
class ConceptSimilarity:
    """
    The structural similarity of two concepts.

    Parameters
    ----------
    concept_a : int
        The first concept.
    concept_b : int
        The second concept.
    common_ancestors : frozenset[int]
        The concepts that are, or are above, both concepts.
    most_recent_common_ancestors : frozenset[int]
        The common ancestors that have no descendant among the common ancestors.
    jaccard : float
        Common ancestors over all ancestors of either concept, both counting the concept itself.
    ancestor_id : int | None
        The most recent common ancestor with the highest information content, the lowest
        id on a tie, or None when the concepts have no common ancestor.
    ancestor_information_content : float
        The information content of ``ancestor_id``, zero when there is none.
    information_content_a : float
        The information content of the first concept.
    information_content_b : float
        The information content of the second concept.
    phenodigm : float
        The geometric mean of ``jaccard`` and ``ancestor_information_content``.
    """

    concept_a: int
    concept_b: int
    common_ancestors: frozenset[int]
    most_recent_common_ancestors: frozenset[int]
    jaccard: float
    ancestor_id: int | None
    ancestor_information_content: float
    information_content_a: float
    information_content_b: float
    phenodigm: float
