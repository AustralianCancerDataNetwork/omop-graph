"""A CDM where a stored relationship ends on a non-standard concept below a standard one."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy.orm import Session, sessionmaker

from fixtures.cdm_builders import (
    classified,
    concept,
    edge,
    relationship,
    relationship_class,
    sqlite_cdm,
)
from omop_graph.extensions.relationship_classification import (
    HierarchyDirection,
    PredicateKind,
)

HOLDER, NON_STANDARD_FILLER, STANDARD_PARENT = range(300, 303)


@contextmanager
def filler_cdm() -> Generator[sessionmaker[Session]]:
    """Yield a CDM with HOLDER site NON_STANDARD_FILLER, which is a kind of STANDARD_PARENT."""
    rows = [
        concept(HOLDER),
        concept(NON_STANDARD_FILLER, standard_concept=None),
        concept(STANDARD_PARENT),
        relationship("Is a", "Subsumes"),
        relationship("Subsumes", "Is a"),
        relationship("Has finding site", "Finding site of"),
        relationship_class(PredicateKind.HIERARCHY, "Taxonomic - up"),
        relationship_class(PredicateKind.HIERARCHY, "Taxonomic - down"),
        relationship_class(PredicateKind.ATTRIBUTE, "Site"),
        classified(
            "Subsumes",
            PredicateKind.HIERARCHY,
            "Taxonomic - down",
            direction=HierarchyDirection.DOWN,
        ),
        classified(
            "Is a",
            PredicateKind.HIERARCHY,
            "Taxonomic - up",
            generalises=True,
            direction=HierarchyDirection.UP,
        ),
        classified("Has finding site", PredicateKind.ATTRIBUTE, "Site", generalises=True),
        edge(NON_STANDARD_FILLER, "Is a", STANDARD_PARENT),
        edge(STANDARD_PARENT, "Subsumes", NON_STANDARD_FILLER),
        edge(HOLDER, "Has finding site", NON_STANDARD_FILLER),
    ]
    with sqlite_cdm(rows) as sessions:
        yield sessions
