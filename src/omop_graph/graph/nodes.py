
from dataclasses import dataclass
from datetime import date
from typing import Optional, Iterable
from enum import Enum, auto
from html import escape
from collections import defaultdict
from itertools import chain

from sqlalchemy import Row


@dataclass(frozen=True)
class ConceptView:
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

    def __repr__(self):
        return (
            f"ConceptView("
            f"id={self.concept_id}, "
            f"{self.vocabulary_id}:{self.concept_code}, "
            f"name={self.concept_name!r})"
        )
    

    def _repr_html_(self) -> str:
        std_badge = (
            f"<span style='background:#2b7; color:white; padding:2px 6px; "
            f"border-radius:4px; font-size:0.75em; margin-left:6px;'>standard</span>"
            if self.standard_concept == "S"
            else ""
        )

        inactive_badge = (
            f"<span style='background:#c33; color:white; padding:2px 6px; "
            f"border-radius:4px; font-size:0.75em; margin-left:6px;'>inactive</span>"
            if self.invalid_reason
            else ""
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
    def from_row(cls, row: Row) -> "ConceptView":
        data = dict(row._mapping)
        data['standard_concept'] = data.pop('standard_concept') == "S"
        return cls(**data)
    
class LabelMatchKind(Enum):
    # Order matters as it ranks the kinds of matches
    DIRECT = auto()
    SYNONYM = auto()
    FULLTEXT = auto()

@dataclass(frozen=True)
class LabelMatch:
    input_label: str
    matched_label: str
    concept_id: int

    match_kind: LabelMatchKind
    is_standard: bool
    is_active: bool


    def _repr_html_(self) -> str:
        kind_badge = {
            LabelMatchKind.DIRECT: "<span style='background:#2b7; color:white; padding:2px 6px; border-radius:4px; font-size:0.75em;'>direct</span>",
            LabelMatchKind.SYNONYM: "<span style='background:#888; color:white; padding:2px 6px; border-radius:4px; font-size:0.75em;'>synonym</span>",
            LabelMatchKind.FULLTEXT: "<span style='background:#27a; color:white; padding:2px 6px; border-radius:4px; font-size:0.75em;'>fulltext</span>",
        }[self.match_kind]

        std_badge = (
            "<span style='background:#2b7; color:white; padding:2px 6px; border-radius:4px; font-size:0.75em;'>standard</span>"
            if self.is_standard else
            "<span style='background:#aaa; color:white; padding:2px 6px; border-radius:4px; font-size:0.75em;'>non-standard</span>"
        )

        active_badge = (
            "<span style='background:#2b7; color:white; padding:2px 6px; border-radius:4px; font-size:0.75em;'>active</span>"
            if self.is_active else
            "<span style='background:#c33; color:white; padding:2px 6px; border-radius:4px; font-size:0.75em;'>inactive</span>"
        )

        return f"""
        <div style="border-bottom:1px solid #eee; padding:4px 0;">
          <div>
            <code>{escape(self.matched_label)}</code>
            → concept_id <b>{self.concept_id}</b>
          </div>
          <div style="font-size:0.8em; margin-top:2px;">
            {kind_badge} {std_badge} {active_badge}
          </div>
        </div>
        """

    def __lt__(self, other: "LabelMatch") -> bool:
        return label_match_rank(self) < label_match_rank(other)

def label_match_rank(m: LabelMatch) -> tuple:
    """
    Lower is better.
    """
    return (
        not m.is_standard,          # prefer standard
        not m.is_active,            # prefer active
        m.match_kind.value,  # prefer direct
        len(m.matched_label), # prefer shorter matches
    )



@dataclass(frozen=True)
class LabelMatchGroupView:
    """
    Grouped, human-readable view over a set of LabelMatch results.
    """
    groups: dict[int, tuple[LabelMatch, ...]]

    @classmethod
    def from_matches(cls, matches: Iterable[LabelMatch]) -> "LabelMatchGroupView":
        grouped: dict[int, list[LabelMatch]] = defaultdict(list)
        for m in matches:
            grouped[m.concept_id].append(m)

        # sort each group by match quality
        groups_sorted = {
            cid: tuple(sorted(ms))
            for cid, ms in grouped.items()
        }
        return cls(groups=groups_sorted)

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
    
    def __iter__(self):
        return chain.from_iterable(self.groups.values())

    def _repr_html_(self) -> str:
        rows = []

        for cid, ms in self.groups.items():
            best = ms[0]

            reasons = []
            if best.match_kind is LabelMatchKind.DIRECT:
                reasons.append("direct name match")
            elif best.match_kind is LabelMatchKind.SYNONYM:
                reasons.append("synonym match")
            elif best.match_kind is LabelMatchKind.FULLTEXT:
                reasons.append("fulltext match")
            else:
                raise ValueError(f"Unknown match kind: {best.match_kind}")

            if best.is_standard:
                reasons.append("standard")
            else:
                reasons.append("non-standard")

            if best.is_active:
                reasons.append("active")
            else:
                reasons.append("inactive")

            other_labels = ", ".join(
                escape(m.matched_label) for m in ms[1:]
            )

            rows.append(f"""
              <tr>
                <td><code>{cid}</code></td>
                <td>{escape(best.matched_label)}</td>
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
          <table style="border-collapse:collapse;">
            <thead>
              <tr>
                <th align="left">concept_id</th>
                <th align="left">best match</th>
                <th align="left">why this matched</th>
                <th align="left">other labels</th>
              </tr>
            </thead>
            <tbody>
              {''.join(rows)}
            </tbody>
          </table>
        </div>
        """
    
@dataclass(frozen=True)
class AncestorMatch:
    ancestor_concept_id: int
    descendant_concept_id: int
    min_levels_of_separation: int