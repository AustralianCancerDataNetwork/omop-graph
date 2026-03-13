"""
Definitions for graph edges and predicates.

This module defines the lightweight data structures representing the
relationships (edges) between concepts in the OMOP Knowledge Graph.

It focuses on data definitions and classification logic, not graph traversal algorithms.

Supported Relationships
-----------------------
* **Mapping:** Semantic equivalence (e.g., source code to standard concept).
* **Versioning:** Lifecycle tracking (e.g., 'replaced by', 'is a').
* **Ontological:** Hierarchical structure (e.g., 'is a', 'subsumes').
* **Attribute:** Descriptive properties (e.g., 'has dose form').
* **Metadata:** Administrative or low-semantic value connections.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date
from enum import Enum, auto
from html import escape
from typing import TYPE_CHECKING, ClassVar, Mapping, Optional

if TYPE_CHECKING:
    from .kg import KnowledgeGraph

logger = logging.getLogger(__name__)



class PredicateKind(Enum):
    """
    Categorization of edge types for filtering and reasoning.
    """

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

    METADATA = auto()

    # Catch-all for uncategorised predicates
    UNCATEGORISED = auto()

    def label(self) -> str:
        """
        Get a human-readable description of the predicate kind.
        """
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
            PredicateKind.UNCATEGORISED: "uncategorised relationship (requires review)",
        }[self]


# These predicates map horizontally between concepts, i.e. equivalence to standard concepts
HORIZONTAL_PREDICATE_KINDS = frozenset(
    {
        PredicateKind.MAPS_TO,
        PredicateKind.MAPS_FROM,
        PredicateKind.VERSIONING,
    }
)

GROUNDING_PREDICATE_KINDS = frozenset(
    {
        *HORIZONTAL_PREDICATE_KINDS,
        PredicateKind.ONTO_UP,      # Sometimes, non-standard to standard is achieved through vertical links
        PredicateKind.ONTO_DOWN,    # NOTE: Not sure if that one is required
        #PredicateKind.COMPOSITION,
    }
)


@dataclass(frozen=True)
class EdgeView:
    """
    A lightweight, immutable view of an edge in the Knowledge Graph.

    Parameters
    ----------
    subject_id : int
        The OMOP Concept ID of the source.
    predicate_id : str
        The relationship ID (e.g., 'is a', 'mapped from').
    object_id : int
        The OMOP Concept ID of the target.
    valid_start_date : date, optional
        The date the relationship became valid.
    valid_end_date : date, optional
        The date the relationship became invalid.
    invalid_reason : str, optional
        The reason for invalidation (e.g., 'D' for deleted), if applicable.
    """

    subject_id: int
    predicate_id: str
    object_id: int
    valid_start_date: Optional[date]
    valid_end_date: Optional[date]
    invalid_reason: Optional[str]

    def __repr__(self) -> str:
        return f"Edge({self.subject_id} -[{self.predicate_id}]-> {self.object_id})"

    def pretty(self, kg: KnowledgeGraph) -> str:
        """
        Return a human-readable string representation of the edge using concept names.

        Parameters
        ----------
        kg : KnowledgeGraph
            The graph instance used to look up concept names.

        Returns
        -------
        str
            A string in the format 'Subject Name -[predicate]-> Object Name'.
        """
        s = kg.concept_view(self.subject_id)
        o = kg.concept_view(self.object_id)
        pred = kg.predicate(self.predicate_id)

        return f"{s.concept_name} -[{pred.name}]-> {o.concept_name}"


# Regex rules for classifying predicates based on their relationship ID.
PREDICATE_RULES: dict[PredicateKind, tuple[re.Pattern, ...]] = {
    PredicateKind.ONTO_UP: (re.compile(r".*\b(is a)\b.*", re.I),),  # rdfs:subClassOf, skos:broader
    PredicateKind.ONTO_DOWN: (re.compile(r"(subsumes)$", re.I),),   # Inverse of rdfs:subClassOf, skos:narrower
    PredicateKind.MAPS_TO: (    # skos:exactMatch, skos:closeMatch
        re.compile(r"^[A-Z0-9 -/]+ (to|-) [A-Z0-9 -/]+( (eq|name))?$", re.I),
        re.compile(r".*(maps to|same_as to|alt_to to|poss_eq to)$", re.I),
    ),
    PredicateKind.MAPS_FROM: (
        re.compile(r".*(mapped from|same_as from|alt_to from|poss_eq from)$", re.I),
    ),
    PredicateKind.VERSIONING: (
        re.compile(
            r".*(replaced|replaces|revision|discontinued|invalid|was_a|historic).*",
            re.I,
        ),
    ),
    PredicateKind.COMPOSITION: (
        re.compile(
            r".*(component|consist|constitut|contain|part of|ingredient|\bing\b|panel|includes).*",
            re.I,
        ),
    ),
    PredicateKind.INTERACTION: (
        re.compile(
            r".*(cause|due to|induce|treat|prevent|contraindicat|\bci\b|interact|affected|etiology|manifestation|inhibit|diagnose|acts on).*",
            re.I,
        ),
    ),
    PredicateKind.ATTRIBUTE: (
        re.compile(
            r".*\b(has | of$|property|value|unit|range|measure|scale|method|mode|available|sterile|dose form|character).*",
            re.I,
        ),
    ),
    PredicateKind.METADATA: (
        re.compile(
            r".*(asso with|occurs |follow|reformulated|physiol effect|during|before|after|towards|temp related).*",
            re.I,
        ),
        re.compile(r".*\b(using|used|uses)\b.*", re.I),
    ),
}
    


@dataclass(frozen=True)
class Predicate:
    """
    Definition of a Relationship Type in the OMOP CDM.

    Parameters
    ----------
    relationship_id : str
        The unique string identifier (e.g. `maps to`).
    name : str
        The human-readable label (e.g. `Non-standard to Standard Mapping`).
    reverse_id : str, optional
        The relationship_id of the inverse relationship (e.g. `mapped from`).
    is_hierarchical : bool
        Whether OMOP defines this as a hierarchical relationship.
    anc_up : bool
        Whether this relationship defines 'defines_ancestry' upwards (deprecated logic).
    anc_down : bool
        Whether this relationship defines 'defines_ancestry' downwards (deprecated logic).
    """

    relationship_id: str
    name: str
    reverse_id: Optional[str]
    is_hierarchical: bool
    anc_up: bool
    anc_down: bool

    @property
    def defines_ancestry(self) -> bool:
        """
        Check if this predicate is involved in defining ancestry.
        """
        return self.anc_up or self.anc_down

    def classify_predicate(self) -> PredicateKind:
        """
        Classify the predicate into a specific `PredicateKind` based on regex rules.

        Returns
        -------
        PredicateKind
            The classification of the relationship (e.g. ONTO_UP, METADATA).
        """
        rid = self.relationship_id.strip()
        
        # Helper logic to match regex
        for kind, pattern_tuple in PREDICATE_RULES.items():
            for pattern in pattern_tuple:
                if pattern.match(rid):
                    return kind

        # Default fallback
        return PredicateKind.UNCATEGORISED

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
    start: Optional[date],
    end: Optional[date],
    invalid_reason: Optional[str],
    *,
    on: Optional[date] = None,
) -> bool:
    """
    Check if a relationship is active on a given date.

    Parameters
    ----------
    start : date, optional
        The start date of the relationship.
    end : date, optional
        The end date of the relationship.
    invalid_reason : str, optional
        The invalid reason code (e.g. 'D', 'U'). None implies valid.
    on : date, optional
        The reference date to check against. If None, only checks `invalid_reason`.

    Returns
    -------
    bool
        True if the relationship is active, False otherwise.
    """
    if on is None:
        return invalid_reason is None
    if start and on < start:
        return False
    if end and on > end:
        return False
    return invalid_reason is None


@dataclass(frozen=True)
class PredicateSummary:
    """
    A summary collection of predicates grouped by their Kind.

    This is primarily used for reporting and visualization in Jupyter environments.
    """

    groups: Mapping[PredicateKind, tuple[Predicate, ...]]

    def __repr__(self) -> str:
        parts = []
        for kind in PredicateKind:
            preds = self.groups.get(kind, ())
            parts.append(f"{kind.name}: {len(preds)}")
        return "PredicateSummary(" + ", ".join(parts) + ")"

    def _repr_html_(self) -> str:
        """
        Rich HTML representation for Jupyter Notebooks.
        """
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