"""Counts over the concepts, relationships and synonyms of a graph."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Grouping(Enum):
    """What concepts are grouped by when they are counted together."""

    VOCABULARY = "vocabulary"
    DOMAIN = "domain"


@dataclass(frozen=True, slots=True)
class ConceptCount:
    """
    The concepts that share a vocabulary, a domain and a validity.

    Parameters
    ----------
    vocabulary_id : str
        The vocabulary of the concepts.
    domain_id : str
        The domain of the concepts.
    deprecated : bool
        Whether the concepts are deprecated, that is deleted or updated into another.
    count : int
        How many concepts.
    """

    vocabulary_id: str
    domain_id: str
    deprecated: bool
    count: int


@dataclass(frozen=True, slots=True)
class ConceptCounts:
    """
    The concepts of a scope, counted by vocabulary, domain and validity.

    Parameters
    ----------
    rows : tuple[ConceptCount, ...]
        One count for each combination that occurs.
    """

    rows: tuple[ConceptCount, ...]

    @property
    def total(self) -> int:
        """All the concepts."""
        return sum(row.count for row in self.rows)

    @property
    def deprecated(self) -> int:
        """The deprecated concepts."""
        return sum(row.count for row in self.rows if row.deprecated)

    def by_vocabulary(self) -> dict[str, int]:
        """The concepts of each vocabulary."""
        counts: dict[str, int] = {}
        for row in self.rows:
            counts[row.vocabulary_id] = counts.get(row.vocabulary_id, 0) + row.count
        return counts

    def by_domain(self) -> dict[str, int]:
        """The concepts of each domain."""
        counts: dict[str, int] = {}
        for row in self.rows:
            counts[row.domain_id] = counts.get(row.domain_id, 0) + row.count
        return counts


@dataclass(frozen=True, slots=True)
class SynonymCounts:
    """
    The synonyms of a scope.

    Parameters
    ----------
    statements : int
        All synonyms, each concept and name counted once.
    distinct : int
        The different synonym names.
    """

    statements: int
    distinct: int


@dataclass(frozen=True, slots=True)
class GroupStatistics:
    """
    The relationships and synonyms of the concepts of one vocabulary or one domain.

    Parameters
    ----------
    relationships : dict[str, int]
        The active relationships of each relationship id, counted by the group of their subject.
    synonyms : SynonymCounts
        The synonyms of the concepts of the group.
    """

    relationships: dict[str, int]
    synonyms: SynonymCounts
