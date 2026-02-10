from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from enum import Enum, auto
from typing import Optional, TYPE_CHECKING, Mapping
from html import escape
if TYPE_CHECKING:
    from .kg import KnowledgeGraph

import logging
logger = logging.getLogger(__name__)

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
    # Vertical Hierarchy
    ONTO_UP = auto()
    ONTO_DOWN = auto()
    
    # Horizontal & Translation
    MAPPING = auto()
    VERSIONING = auto()
    
    # Semantic Enrichment
    COMPOSITION = auto()
    INTERACTION = auto()
    ATTRIBUTE = auto()
    
    # Noise
    METADATA = auto()

    def label(self) -> str:
        return {
            PredicateKind.ONTO_UP: "ontological relationship (upwards, generalization)",
            PredicateKind.ONTO_DOWN: "ontological relationship (downwards, specialization)",
            
            PredicateKind.MAPPING: "mapping relationship (cross-vocabulary translation)",
            PredicateKind.VERSIONING: "versioning relationship (lifecycle/deprecation)",
            
            PredicateKind.COMPOSITION: "compositional relationship (part-whole structure)",
            PredicateKind.INTERACTION: "interaction relationship (causal/clinical logic)",
            PredicateKind.ATTRIBUTE: "attribute enrichment (descriptive property)",
            
            PredicateKind.METADATA: "metadata relationship (administrative/low semantic value)",
        }[self]

HIERARCHICAL_PREDICATE_KINDS = frozenset({
    PredicateKind.ONTO_UP, 
    PredicateKind.ONTO_DOWN, 
    PredicateKind.MAPPING,
    PredicateKind.VERSIONING,
    PredicateKind.COMPOSITION
})

PREDICATE_VERSIONING_KEYWORDS = frozenset([
    "replaced", "replaces", "revision", "discontinued", "invalid", "was_a"
])

PREDICATE_MAPPING_KEYWORDS = frozenset([
    "maps to", "mapped from", "equivalent", " eq", "same_as", 
    "alt_to", "poss_eq", " - ", " to ",
    "brand", "tradename"
])

PREDICATE_COMPOSITION_KEYWORDS = frozenset([
    "component", "consist", "constitut", "contain",
    "part of", "ingredient", " ing", "panel", "includes"
])

PREDICATE_INTERACTION_KEYWORDS = frozenset([
    "causes", "caused", "due to", "induces", "induced", 
    "treat", "prevent", "contraindicat", " ci ", " ci",
    "interact", "affected", "etiology", "manifestation"
])

PREDICATE_ATTRIBUTE_KEYWORDS = frozenset([
    "property", "value", "unit", "range", "measure", "scale", "method", "mode"
])

PREDICATE_METADATA_KEYWORDS = frozenset([
    "asso with",
    "occurs after",
    "occurs before",
    "followed by",
    "follows"

])


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
    relationship_id: str  # Not really an ID but a unique string label for the relationship, e.g. "is a", "maps to", etc.
    name: str
    reverse_id: Optional[str]  # Same here, unique string label for the reverse relationship if it exists, e.g. "has" is reverse of "is a"
    is_hierarchical: bool
    upwards: bool
    downwards: bool

    @property
    def defines_ancestry(self) -> bool:
        return self.upwards or self.downwards

    def classify_predicate(self, *, kg) -> PredicateKind:
        # 1. Structural Hierarchy (The Spine)
        if self.upwards and self.downwards:
            raise ValueError(f"Predicate {self.relationship_id} cannot be both upwards and downwards")
        if self.upwards:
            return PredicateKind.ONTO_UP
        elif self.downwards:
            return PredicateKind.ONTO_DOWN
        
        rid = self.relationship_id.lower()

        if any(kw in rid for kw in PREDICATE_VERSIONING_KEYWORDS):
            return PredicateKind.VERSIONING

        if any(kw in rid for kw in PREDICATE_MAPPING_KEYWORDS):
            return PredicateKind.MAPPING

        if any(kw in rid for kw in PREDICATE_COMPOSITION_KEYWORDS):
            return PredicateKind.COMPOSITION

        if any(kw in rid for kw in PREDICATE_INTERACTION_KEYWORDS):
            return PredicateKind.INTERACTION

        if rid.startswith("has ") or rid.endswith(" of") or any(kw in rid for kw in PREDICATE_ATTRIBUTE_KEYWORDS):
            return PredicateKind.ATTRIBUTE
        
        if any(kw in rid for kw in PREDICATE_METADATA_KEYWORDS):
            return PredicateKind.METADATA

        logger.debug(f"Predicate classified as METADATA: {self.relationship_id}")
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
