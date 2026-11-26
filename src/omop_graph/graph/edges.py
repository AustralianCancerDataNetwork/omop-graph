"""
Definitions for graph edges and predicates.

This module defines the lightweight data structures representing the
relationships (edges) between concepts in the OMOP Knowledge Graph.

It focuses on data definitions and classification logic, not graph traversal algorithms.

Supported Relationships (``PredicateKind``)
--------------------------------------------
* **Hierarchy:** Structural parent/child relationships (e.g., 'is a', 'subsumes').
* **Identity:** Semantic equivalence (e.g., 'maps to', source code to standard concept).
* **Composition:** Part-whole/component relationships.
* **Association:** General, non-hierarchical relationships between concepts.
* **Attribute:** Descriptive properties (e.g., 'has dose form').
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import TYPE_CHECKING

from omop_alchemy.cdm.model import normalised_flag

from ..extensions.omop_alchemy import HierarchyDirection, PredicateKind, PredicateSubkind

if TYPE_CHECKING:
    from .kg import KnowledgeGraph

logger = logging.getLogger(__name__)

IS_A_RELATIONSHIP_ID = "Is a"
"""The relationship id of the subclass edge. Its transitive closure is the strict hierarchy."""

REPLACED_BY_RELATIONSHIP_ID = "Concept replaced by"
"""The relationship id from a deprecated concept to the concept that replaces it."""


@dataclass(frozen=True)
class EdgeFilter:
    """
    Selects edges. Every field is optional and the fields combine with AND.

    Parameters
    ----------
    predicate_ids : frozenset[str], optional
        Only these relationship ids. An empty set selects no edge, and None selects all.
    predicate_kinds : frozenset[PredicateKind], optional
        Only relationships of these kinds.
    predicate_subkinds : frozenset[PredicateSubkind], optional
        Only relationships of these subkinds. The knowledge graph rejects a subkind
        that no relationship is classified under, listing the valid ones.
    active_only : bool
        If True, only edges that are not deprecated (``invalid_reason`` is unset).
    on : date, optional
        Only edges whose validity period covers this date. Independent of
        ``active_only``: set ``active_only=False`` to include edges deprecated since.
    within_domain : bool
        If True, only edges whose two concepts share a domain. Defaults to False because
        most attribute edges cross domains (a Procedure's ``Has method`` to an
        Observation, ``Has dir proc site`` to a Spec Anatomic Site).
    """

    predicate_ids: frozenset[str] | None = None
    predicate_kinds: frozenset[PredicateKind] | None = None
    predicate_subkinds: frozenset[PredicateSubkind] | None = None
    active_only: bool = True
    on: date | None = None
    within_domain: bool = False

    def __post_init__(self) -> None:
        for kind in self.predicate_kinds or ():
            if not isinstance(kind, PredicateKind):
                raise TypeError(f"predicate_kinds must hold PredicateKind members, got {kind!r}")
        for subkind in self.predicate_subkinds or ():
            if not isinstance(subkind, PredicateSubkind):
                raise TypeError(
                    f"predicate_subkinds must hold PredicateSubkind members, got {subkind!r}"
                )


IS_A_FILTER = EdgeFilter(predicate_ids=frozenset({IS_A_RELATIONSHIP_ID}))
"""Selects the active ``Is a`` edges."""

IDENTITY_FILTER = EdgeFilter(predicate_kinds=frozenset({PredicateKind.IDENTITY}))
"""Selects the active identity edges, such as ``Maps to``."""


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
    predicate_kind : PredicateKind
        The high-level category of this predicate (Hierarchy, Identity, Composition, Association, or Attribute).
    predicate_subkind : str
        The more fine-grained subclass of this predicate (e.g. for Identity: 'standard to non-standard', 'source to standard', etc.)
    """

    subject_id: int
    predicate_id: str
    object_id: int
    valid_start_date: date | None
    valid_end_date: date | None
    invalid_reason: str | None
    predicate_kind: PredicateKind
    predicate_subkind: str

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


class MappingStrength(Enum):
    """How closely a mapping relationship equates its two concepts."""

    EXACT = "exact"
    CLOSE = "close"
    RELATED = "related"

    @classmethod
    def of(cls, subkind: PredicateSubkind) -> MappingStrength | None:
        """
        Classify a relationship subkind as a mapping.

        Exact and near equivalence are identity relationships, and a mapping association is
        a related one. Every other subkind, such as version lineage, is not a mapping.
        """
        return _MAPPING_SUBKINDS.get(subkind)


_MAPPING_SUBKINDS: dict[PredicateSubkind, MappingStrength] = {
    PredicateSubkind(PredicateKind.IDENTITY, "Exact equivalence"): MappingStrength.EXACT,
    PredicateSubkind(PredicateKind.IDENTITY, "Near equivalence"): MappingStrength.CLOSE,
    PredicateSubkind(PredicateKind.ASSOCIATION, "Mapping"): MappingStrength.RELATED,
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
    predicate_kind : PredicateKind
        The high-level category of this predicate (Hierarchy, Identity, Composition, Association, or Attribute).
    predicate_subkind : str
        The more fine-grained subclass of this predicate (e.g. for Identity: 'standard to non-standard', 'source to standard', etc.)
    direction : HierarchyDirection, optional
        For a Hierarchy predicate, whether the stored edge points up to the parent or
        group, or down to a child or member. None for every other kind.
    generalises : bool
        Whether a stored ``A R B`` also holds as ``A R C`` for each ``Is a`` ancestor ``C``
        of ``B``. Set for attribute, composition and association relationships such as
        ``Has finding site``, and for the upward taxonomic relationships such as ``Is a``.
    """

    relationship_id: str
    name: str
    reverse_id: str | None
    predicate_kind: PredicateKind
    predicate_subkind: str
    direction: HierarchyDirection | None
    generalises: bool

    @property
    def mapping_strength(self) -> MappingStrength | None:
        """How closely this relationship maps its concepts, or None when it is no mapping."""
        return MappingStrength.of(PredicateSubkind(self.predicate_kind, self.predicate_subkind))

    def __repr__(self) -> str:
        reverse = f" | reverse={self.reverse_id}" if self.reverse_id else ""
        return f"Predicate({self.relationship_id!r}: {self.name!r}{reverse})"


def is_active(
    start: date | None,
    end: date | None,
    invalid_reason: str | None,
    *,
    on: date | None = None,
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
    # normalised_flag folds blank/whitespace-only to None, matching the SQL
    # predicate. A raw ``is None`` test would read '' as a deprecation marker.
    unset = normalised_flag(invalid_reason) is None
    if on is None:
        return unset
    if start and on < start:
        return False
    if end and on > end:
        return False
    return unset
