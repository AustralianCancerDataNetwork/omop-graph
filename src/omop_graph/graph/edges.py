from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from enum import Enum, auto
from typing import Optional, TYPE_CHECKING, Mapping
from html import escape
if TYPE_CHECKING:
    from .kg import KnowledgeGraph

"""
Definitions for graph edges and predicates.

Edges represent relationships between concepts in the knowledge graph.

Scope: Lightweight data structures only. No graph algorithms here. 
i.e. What is an edge or predicate, and how do I classify or filter them?

Supported relationships include:
    * mapping (semantic equivalence)
    * versioning (replaced by / replaces)
    * ontological (is a / subclass of)
    * attribute (has attribute)
    * metadata (additional information)
"""

class PredicateKind(Enum):
    ONTOLOGICAL = auto()
    ATTRIBUTE = auto()
    MAPPING = auto()
    VERSIONING = auto()
    METADATA = auto()

    def label(self) -> str:
        return {
            PredicateKind.ONTOLOGICAL: "ontological relationship (preferred structure)",
            PredicateKind.MAPPING: "mapping relationship (cross-vocabulary)",
            PredicateKind.ATTRIBUTE: "attribute enrichment",
            PredicateKind.VERSIONING: "versioning relationship",
            PredicateKind.METADATA: "metadata relationship (low semantic value)",
        }[self]

@dataclass(frozen=True)
class EdgeView:
    subject_id: int
    predicate_id: str
    object_id: int
    valid_start_date: Optional[date]
    valid_end_date: Optional[date]
    invalid_reason: Optional[str]


    def __repr__(self) -> str:
        return (
            f"Edge({self.subject_id} -[{self.predicate_id}]-> {self.object_id})"
        )
    

    def pretty(self, kg: "KnowledgeGraph") -> str:
        s = kg.concept_view(self.subject_id)
        o = kg.concept_view(self.object_id)
        pred = kg.predicate(self.predicate_id)

        return (
            f"{s.concept_name} "
            f"-[{pred.name}]-> "
            f"{o.concept_name}"
        )


@dataclass(frozen=True)
class Predicate:
    relationship_id: str
    name: str
    reverse_id: Optional[str]
    is_hierarchical: bool
    defines_ancestry: bool

    def classify_predicate(self, *, kg) -> PredicateKind:
        if self.defines_ancestry:
            return PredicateKind.ONTOLOGICAL

        name = self.name.lower()

        if "maps to" in name or "mapped from" in name or "equivalent" in name:
            return PredicateKind.MAPPING

        if "replaced" in name or "replaces" in name:
            return PredicateKind.VERSIONING

        if name.startswith("has "):
            return PredicateKind.ATTRIBUTE

        if self.reverse_id:
            rev = kg.predicate(self.reverse_id)
            if rev.name.lower().startswith("has "):
                return PredicateKind.METADATA

        return PredicateKind.METADATA
    

    def __repr__(self) -> str:
        flags = []
        if self.is_hierarchical:
            flags.append("hierarchical")
        if self.defines_ancestry:
            flags.append("ancestry")
        if self.reverse_id:
            flags.append(f"reverse={self.reverse_id}")

        flag_str = f" | {', '.join(flags)}" if flags else ""

        return f"Predicate({self.relationship_id!r}: {self.name!r}{flag_str})"

def is_active(
    start: date | None,
    end: date | None,
    invalid_reason: str | None,
    *,
    on: date | None = None,
) -> bool:
    if on is None:
        return invalid_reason is None
    if start and on < start:
        return False
    if end and on > end:
        return False
    return invalid_reason is None


@dataclass(frozen=True)
class PredicateSummary:
    groups: Mapping[PredicateKind, tuple[Predicate, ...]]

    def __repr__(self) -> str:
        parts = []
        for kind in PredicateKind:
            preds = self.groups.get(kind, ())
            parts.append(f"{kind.name}: {len(preds)}")
        return "PredicateSummary(" + ", ".join(parts) + ")"

    def _repr_html_(self) -> str:
        blocks = []

        for kind in PredicateKind:
            preds = self.groups.get(kind, ())
            if not preds:
                continue

            pred_list = "".join(
                f"<li><code>{escape(p.relationship_id)}</code>: {escape(p.name)}</li>"
                for p in sorted(preds, key=lambda p: p.relationship_id)
            )

            blocks.append(f"""
              <details style="margin-bottom:6px;">
                <summary style="cursor:pointer; font-weight:600;">
                  {escape(kind.name)} 
                  <span style="color:#666; font-weight:normal;">
                    ({len(preds)}) — {escape(kind.label())}
                  </span>
                </summary>
                <ul style="margin:6px 0 0 16px;">
                  {pred_list}
                </ul>
              </details>
            """)

        return f"""
        <div style="border:1px solid #ddd; border-radius:8px; padding:10px;">
          <div style="font-weight:600; margin-bottom:8px;">
            Predicate summary
          </div>
          {''.join(blocks)}
        </div>
        """
