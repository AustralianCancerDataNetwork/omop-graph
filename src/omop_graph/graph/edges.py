from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from enum import Enum, auto
from typing import Optional, TYPE_CHECKING, Mapping, ClassVar
from html import escape
import re
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
    MAPS_TO = auto()
    MAPS_FROM = auto()
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
            
            PredicateKind.MAPS_TO: "mapping relationship (cross-vocabulary translation)",
            PredicateKind.MAPS_FROM: "reverse mapping relationship (cross-vocabulary translation)",
            PredicateKind.VERSIONING: "versioning relationship (lifecycle/deprecation)",
            
            PredicateKind.COMPOSITION: "compositional relationship (part-whole structure)",
            PredicateKind.INTERACTION: "interaction relationship (causal/clinical logic)",
            PredicateKind.ATTRIBUTE: "attribute enrichment (descriptive property)",
            
            PredicateKind.METADATA: "metadata relationship (administrative/low semantic value)",
        }[self]

HIERARCHICAL_PREDICATE_KINDS = frozenset({
    PredicateKind.ONTO_UP, 
    PredicateKind.ONTO_DOWN, 
    PredicateKind.MAPS_TO,
    PredicateKind.VERSIONING,
    PredicateKind.COMPOSITION
})


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

# Can be adapted using self.kg.predicate_summary() as it queries all unique predicates
PREDICATE_RULES: tuple[tuple[PredicateKind, re.Pattern], ...] = (
    # "ATC - RxNorm eq", "ATC to NDFRT eq", "RxNorm - CVX"
    (PredicateKind.ONTO_UP, re.compile(r".*\b(is a)\b.*", re.I)),
    (PredicateKind.ONTO_DOWN, re.compile(r"(subsumes)$", re.I)),
    (PredicateKind.MAPS_TO, re.compile(r"^[A-Z0-9 -/]+ (to|-) [A-Z0-9 -/]+( (eq|name))?$", re.I)),
    (PredicateKind.MAPS_TO, re.compile(r".*(maps to|same_as to|alt_to to|poss_eq to)$", re.I)),
    (PredicateKind.MAPS_FROM, re.compile(r".*(mapped from|same_as from|alt_to from|poss_eq from)$", re.I)),
    (PredicateKind.VERSIONING, re.compile(r".*(replaced|replaces|revision|discontinued|invalid|was_a|historic).*", re.I)),
    (PredicateKind.COMPOSITION, re.compile(r".*(component|consist|constitut|contain|part of|ingredient|\bing\b|panel|includes).*", re.I)),
    (PredicateKind.INTERACTION, re.compile(r".*(cause|due to|induce|treat|prevent|contraindicat|\bci\b|interact|affected|etiology|manifestation|inhibit|diagnose|acts on).*", re.I)),
    (PredicateKind.ATTRIBUTE, re.compile(r".*\b(has | of$|property|value|unit|range|measure|scale|method|mode|available|sterile|dose form|character).*", re.I)),
    (PredicateKind.METADATA, re.compile(r".*(asso with|occurs |follow|reformulated|physiol effect|during|before|after|towards|temp related).*", re.I)),
    (PredicateKind.METADATA, re.compile(r".*\b(using|used|uses)\b.*", re.I)),
)

@dataclass(frozen=True)
class Predicate:
    relationship_id: str  # string ID (e.g. `maps to`)
    name: str # human readable label (e.g. `Non-standard to Standard Mapping`)
    reverse_id: Optional[str]  # string ID reverse relationship (e.g. `mapped from`)
    is_hierarchical: bool
    anc_up: bool
    anc_down: bool


    @property
    def defines_ancestry(self) -> bool:
        return self.anc_up or self.anc_down

    def classify_predicate(self) -> PredicateKind:       
        
        rid = self.relationship_id.strip()
        predicate_kind = self._get_regex_kind(rid)

        # Defines ancestry is not really used so could be removed eventually
        # NOTE: This is for debugging only
        #if self.defines_ancestry:
        #    if predicate_kind in (PredicateKind.ONTO_UP, PredicateKind.ONTO_DOWN):
        #        return predicate_kind
        #    logger.debug(f"Predicate {self.relationship_id} [{self.name}] has ancestry and is of type {predicate_kind}")

        if predicate_kind is not None:
            return predicate_kind

        logger.debug(f"Defaults to METADATA: {rid}")
        return PredicateKind.METADATA
    
    def _get_regex_kind(self, rid) -> Optional[PredicateKind]:
        for kind, pattern in PREDICATE_RULES:
            if pattern.match(rid):
                return kind
        return None
    

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
