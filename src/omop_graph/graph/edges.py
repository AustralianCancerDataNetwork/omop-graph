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
from dataclasses import dataclass, fields
from datetime import date
from typing import TYPE_CHECKING, Optional

from ..extensions.omop_alchemy import ClassIDEnum

if TYPE_CHECKING:
    from .kg import KnowledgeGraph

logger = logging.getLogger(__name__)


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
    class_id: ClassIDEnum
    subclass_id: str

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
    
    @classmethod
    def from_query(cls, entry) -> "EdgeView":
        data = dict(zip([f.name for f in fields(cls)], entry))
        if "class_id" in data:
            data["class_id"] = ClassIDEnum(data["class_id"])
        return cls(**data)

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
    class_id: ClassIDEnum
    subclass_id: str

    @property
    def defines_ancestry(self) -> bool:
        """
        Check if this predicate is involved in defining ancestry.
        """
        return self.anc_up or self.anc_down

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