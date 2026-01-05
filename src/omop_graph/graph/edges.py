from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from enum import Enum, auto
from typing import Optional, TYPE_CHECKING
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

@dataclass(frozen=True)
class Predicate:
    relationship_id: str
    name: str
    reverse_id: Optional[str]
    is_hierarchical: bool
    defines_ancestry: bool

    def classify_predicate(self, *, kg) -> PredicateKind:
        if self.defines_ancestry or self.is_hierarchical:
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

def _pred_id(pred: Predicate | str | None) -> str | None:
    if pred is None:
        return None
    if isinstance(pred, Predicate):
        return pred.relationship_id
    if isinstance(pred, str):
        return pred
    raise TypeError(f"Unsupported predicate type: {type(pred)}")


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
