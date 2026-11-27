"""Label matches: how a label matched a concept, and the matches grouped by concept."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum, unique
from html import escape
from itertools import chain
from typing import Self


@unique
class LabelMatchKind(Enum):
    """
    Classification of how a label matched a concept.

    The value is the priority, where lower is better, and no two members share one. Sort a
    list of kinds to get the order a search tries them in. Each member also carries the
    badge text, badge colour and reason shown in notebook renderings.

    Attributes
    ----------
    badge : str
        Short text of the badge.
    colour : str
        Background colour of the badge.
    reason : str
        Why the label matched, for summaries.

    Notes
    -----
    Supported match kinds include:
    - EXACT: Direct case-insensitive match on concept_name.
    - STARTS_WITH: The label starts with the query, case-insensitively.
    - PARTIAL: Partial match (fuzzy) substrings with ILIKE.
    - FTS: Full-text search match (fuzzy).
    - EMBEDDING: Match based on vector similarity.

    It does not use synonym vs. concept_name as a ranking signal,
    as these are treated as identical quality.
    The ``LabelMatch.synonym`` field carries that distinction for callers that need it.
    """

    EXACT = (0, "exact", "#2b7", "direct name match")
    STARTS_WITH = (1, "starts with", "#84b", "label starts with the query")
    PARTIAL = (2, "partial", "#888", "label contains the query")
    FTS = (3, "fulltext", "#27a", "fulltext match")
    EMBEDDING = (4, "embedding", "#a72", "embedding match")

    badge: str
    colour: str
    reason: str

    def __new__(cls, priority: int, badge: str, colour: str, reason: str) -> Self:
        member = object.__new__(cls)
        member._value_ = priority
        member.badge = badge
        member.colour = colour
        member.reason = reason
        return member

    def __lt__(self, other: LabelMatchKind) -> bool:
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
        kind_badge = (
            f"<span style='background:{self.match_kind.colour}; color:white; padding:2px 6px; "
            f"border-radius:4px; font-size:0.75em;'>{self.match_kind.badge}</span>"
        )

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

    groups: dict[int, tuple[LabelMatch, ...]]

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
        grouped: dict[int, list[LabelMatch]] = defaultdict(list)
        for m in matches:
            grouped[m.matched_concept_id].append(m)

        grouped_tuple = {
            cid: tuple(sorted(ms, key=lambda m: m.match_kind.value)) for cid, ms in grouped.items()
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

            reasons = [best.match_kind.reason]

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
                  {other_labels if other_labels else "&mdash;"}
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
