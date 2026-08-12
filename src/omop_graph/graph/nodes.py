"""
Node definitions for the OMOP Knowledge Graph.

This module defines lightweight data structures (Views) representing entities
in the OMOP graph, such as Concepts and search matches.

These classes are primarily used for:
1.  Holding data returned by database queries.
2.  Rendering rich representations in Jupyter notebooks via `_repr_html_`.
3.  Ranking and sorting search results.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from enum import Enum
from html import escape
from itertools import chain
from typing import Dict, Iterable, List, Optional, Tuple

from sqlalchemy import Row

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConceptView:
    """
    A lightweight, immutable view of an OMOP Concept.

    This class represents a row from the `concept` table, optimized for
    read-only access and visualization.

    Parameters
    ----------
    concept_id : int
        The unique OMOP identifier.
    concept_name : str
        The human-readable name of the concept.
    concept_code : str
        The source code in the original vocabulary.
    vocabulary_id : str
        The vocabulary identifier (e.g., 'SNOMED', 'RxNorm').
    domain_id : str
        The domain identifier (e.g., 'Condition', 'Drug').
    concept_class_id : str
        The class of the concept (e.g., 'Clinical Finding').
    standard_concept : bool
        True if this is a Standard Concept ('S'), False otherwise.
    valid_start_date : date
        The start date of validity.
    valid_end_date : date
        The end date of validity.
    invalid_reason : str, optional
        The reason for invalidation (e.g., 'D', 'U'), or None if valid.
    """

    concept_id: int
    concept_name: str
    concept_code: str
    vocabulary_id: str
    domain_id: str
    concept_class_id: str
    standard_concept: bool
    valid_start_date: date
    valid_end_date: date
    invalid_reason: Optional[str]

    @property
    def is_active(self) -> bool:
        """Whether the concept is active under OMOP Alchemy's flag semantics."""
        value = self.invalid_reason.strip() if self.invalid_reason is not None else ""
        return not value

    def __repr__(self) -> str:
        return (
            f"ConceptView("
            f"id={self.concept_id}, "
            f"{self.vocabulary_id}:{self.concept_code}, "
            f"name={self.concept_name!r})"
        )

    def _repr_html_(self) -> str:
        """
        Render a rich HTML representation for Jupyter notebooks.
        """
        std_badge = ""
        if self.standard_concept:
            std_badge = (
                "<span style='background:#2b7; color:white; padding:2px 6px; "
                "border-radius:4px; font-size:0.75em; margin-left:6px;'>standard</span>"
            )

        inactive_badge = ""
        if self.invalid_reason:
            inactive_badge = (
                "<span style='background:#c33; color:white; padding:2px 6px; "
                "border-radius:4px; font-size:0.75em; margin-left:6px;'>inactive</span>"
            )

        return f"""
        <div style="border:1px solid #ddd; border-radius:8px; padding:8px; max-width:520px;">
          <div style="font-weight:600; font-size:1.05em;">
            {escape(self.concept_name)}
            {std_badge}
            {inactive_badge}
          </div>

          <div style="font-family:monospace; color:#555; margin-top:2px;">
            {escape(self.vocabulary_id)}:{escape(self.concept_code)}
          </div>

          <table style="margin-top:6px; font-size:0.85em;">
            <tr><th align="left">concept_id</th><td>{self.concept_id}</td></tr>
            <tr><th align="left">domain</th><td>{escape(self.domain_id)}</td></tr>
            <tr><th align="left">class</th><td>{escape(self.concept_class_id)}</td></tr>
            <tr><th align="left">valid</th>
                <td>{self.valid_start_date} → {self.valid_end_date}</td></tr>
          </table>
        </div>
        """

    @classmethod
    def from_row(cls, row: Row) -> ConceptView:
        """
        Create a ConceptView from a SQLAlchemy Row.

        The query layer projects OMOP Alchemy's canonical standardness expression
        into the ``standard_concept`` field as a boolean.

        Parameters
        ----------
        row : Row
            A row object returned by a SQLAlchemy query selecting from the Concept table.

        Returns
        -------
        ConceptView
            The instantiated view.
        """
        data = dict(row._mapping)
        data["standard_concept"] = bool(data["standard_concept"])
        return cls(**data)


class LabelMatchKind(Enum):
    """
    Classification of how a label matched a concept.
    Value order defines priority (lower is better):

    Notes
    -----
    Supported match kinds include:
    - EXACT: Direct case-insensitive match on concept_name.
    - FTS: Full-text search match (fuzzy).
    - PARTIAL: Partial match (fuzzy) substrings with ILIKE.
    - EMBEDDING: Match based on vector similarity.

    It does not use synonym vs. concept_name as a ranking signal,
    as these are treated as identical quality.
    The ``LabelMatch.synonym`` field carries that distinction for callers that need it.
    """

    EXACT = 0
    FTS = 1
    PARTIAL = 2
    EMBEDDING = 3

    def __lt__(self, other: "LabelMatchKind") -> bool:
        return self.value < other.value


@dataclass(frozen=True)
class LabelMatch:
    """
    A single result from a text search/grounding operation.

    Parameters
    ----------
    input_query : str
        The original text that was searched.
    matched_concept_label : str
        The text in the database that matched (concept name or synonym).
    matched_concept_id : int
        The OMOP Concept ID of the matched concept.
    match_kind : LabelMatchKind
        How the match was found (Exact, FTS, Partial, Embedding).
    is_standard : bool
        Whether the matched concept is Standard.
    is_active : bool
        Whether the matched concept is currently valid.
    synonym : bool
        True if the match came from the ``concept_synonym`` table rather than
        the primary ``concept_name`` field.  This is informational only and does
        not affect priority ordering. See ``LabelMatchKind`` for ranking.
    """

    input_query: str
    matched_concept_label: str
    matched_concept_id: int

    match_kind: LabelMatchKind
    is_standard: bool
    is_active: bool
    synonym: bool

    def _repr_html_(self) -> str:
        """
        Render a rich HTML representation for Jupyter notebooks.
        """
        kind_badges = {
            LabelMatchKind.EXACT: "<span style='background:#2b7; color:white; padding:2px 6px; border-radius:4px; font-size:0.75em;'>exact</span>",
            LabelMatchKind.FTS: "<span style='background:#27a; color:white; padding:2px 6px; border-radius:4px; font-size:0.75em;'>fulltext</span>",
            LabelMatchKind.PARTIAL: "<span style='background:#888; color:white; padding:2px 6px; border-radius:4px; font-size:0.75em;'>partial</span>",
            LabelMatchKind.EMBEDDING: "<span style='background:#a72; color:white; padding:2px 6px; border-radius:4px; font-size:0.75em;'>embedding</span>",
        }
        kind_badge = kind_badges[self.match_kind]

        std_badge = (
            "<span style='background:#2b7; color:white; padding:2px 6px; border-radius:4px; font-size:0.75em;'>standard</span>"
            if self.is_standard
            else "<span style='background:#aaa; color:white; padding:2px 6px; border-radius:4px; font-size:0.75em;'>non-standard</span>"
        )

        active_badge = (
            "<span style='background:#2b7; color:white; padding:2px 6px; border-radius:4px; font-size:0.75em;'>active</span>"
            if self.is_active
            else "<span style='background:#c33; color:white; padding:2px 6px; border-radius:4px; font-size:0.75em;'>inactive</span>"
        )

        return f"""
        <div style="border-bottom:1px solid #eee; padding:4px 0;">
          <div>
            <code>{escape(self.matched_concept_label)}</code>
            → concept_id <b>{self.matched_concept_id}</b>
          </div>
          <div style="font-size:0.8em; margin-top:2px;">
            {kind_badge} {std_badge} {active_badge}
          </div>
        </div>
        """


@dataclass(frozen=True)
class LabelMatchGroupView:
    """
    A grouped collection of LabelMatch results.

    Aggregates matches by Concept ID to present a unified view of
    why a specific concept was selected (e.g., matched via name AND synonym).

    Parameters
    ----------
    groups : dict[int, tuple[LabelMatch, ...]]
        A dictionary mapping Concept ID to a tuple of sorted LabelMatches.
    """

    groups: Dict[int, Tuple[LabelMatch, ...]]

    @classmethod
    def from_matches(cls, matches: Iterable[LabelMatch]) -> LabelMatchGroupView:
        """
        Construct a group view from a flat iterable of matches.

        Matches are grouped by Concept ID.

        Notes
        -----
        This could also be performed in SQL during the query phase.

        Parameters
        ----------
        matches : Iterable[LabelMatch]
            The raw search results.

        Returns
        -------
        LabelMatchGroupView
            The grouped view.
        """
        grouped: Dict[int, List[LabelMatch]] = defaultdict(list)
        for m in matches:
            grouped[m.matched_concept_id].append(m)

        grouped_tuple = {
            cid: tuple(sorted(ms, key=lambda m: m.match_kind.value))
            for cid, ms in grouped.items()
        }
        return cls(groups=grouped_tuple)

    def __iter__(self):
        """Iterate over all matches flattened."""
        return chain.from_iterable(self.groups.values())

    def __repr__(self) -> str:
        parts = []
        for cid, ms in self.groups.items():
            best = ms[0]
            parts.append(
                f"{cid}("
                f"{best.match_kind.name.lower()}, "
                f"{'std' if best.is_standard else 'non-std'}, "
                f"{'active' if best.is_active else 'inactive'})"
            )
        return "LabelMatchGroupView(" + ", ".join(parts) + ")"

    def _repr_html_(self) -> str:
        """
        Render a summary table for Jupyter notebooks.
        """
        rows = []

        for cid, ms in self.groups.items():
            best = ms[0]

            reasons = []
            # Determine match kind
            if best.match_kind is LabelMatchKind.EXACT:
                reasons.append("direct name match")
            elif best.match_kind is LabelMatchKind.PARTIAL:
                reasons.append("synonym match")
            elif best.match_kind is LabelMatchKind.FTS:
                reasons.append("fulltext match")
            elif best.match_kind is LabelMatchKind.EMBEDDING:
                reasons.append("embedding match")
            else:
                reasons.append("unknown match")

            # Determine concept status
            reasons.append("standard" if best.is_standard else "non-standard")
            reasons.append("active" if best.is_active else "inactive")

            # Collect other matched synonyms
            other_labels = ", ".join(escape(m.matched_concept_label) for m in ms[1:])

            rows.append(f"""
              <tr>
                <td><code>{cid}</code></td>
                <td>{escape(best.matched_concept_label)}</td>
                <td>{escape(", ".join(reasons))}</td>
                <td style="font-size:0.85em; color:#666;">
                  {other_labels if other_labels else "—"}
                </td>
              </tr>
            """)

        return f"""
        <div style="border:1px solid #ddd; border-radius:8px; padding:8px;">
          <div style="font-weight:600; margin-bottom:6px;">
            Label match summary
          </div>
          <table style="border-collapse:collapse; width:100%;">
            <thead>
              <tr style="border-bottom: 1px solid #eee; text-align:left;">
                <th style="padding:4px;">concept_id</th>
                <th style="padding:4px;">best match</th>
                <th style="padding:4px;">details</th>
                <th style="padding:4px;">other matched labels</th>
              </tr>
            </thead>
            <tbody>
              {"".join(rows)}
            </tbody>
          </table>
        </div>
        """


@dataclass(frozen=True)
class AncestorMatch:
    """
    Represents a successful hierarchy check between two concepts.

    Parameters
    ----------
    ancestor_concept_id : int
        The concept ID of the ancestor.
    descendant_concept_id : int
        The concept ID of the descendant.
    min_levels_of_separation : int
        The shortest path distance between them (0 = self).
    """

    ancestor_concept_id: int
    descendant_concept_id: int
    min_levels_of_separation: int
