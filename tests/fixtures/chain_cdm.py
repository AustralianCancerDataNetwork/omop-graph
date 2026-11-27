"""A CDM where a stored relationship has a deprecated concept on the ``Is a`` chain at each end."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

from omop_alchemy.cdm.model.vocabulary.concept import Concept
from omop_alchemy.cdm.model.vocabulary.concept_ancestor import Concept_Ancestor
from omop_alchemy.cdm.model.vocabulary.concept_relationship import Concept_Relationship
from omop_alchemy.cdm.model.vocabulary.concept_synonym import Concept_Synonym
from omop_alchemy.cdm.model.vocabulary.relationship import Relationship
from omop_alchemy.cdm.model.vocabulary.vocabulary import Vocabulary
from sqlalchemy.orm import Session, sessionmaker

from fixtures.cdm_builders import (
    classified,
    concept,
    edge,
    relationship,
    relationship_class,
    sqlite_cdm,
)
from omop_graph.extensions.omop_alchemy import (
    HierarchyDirection,
    PredicateKind,
    RelationshipClass,
    RelationshipMapping,
)

HOLDER, HOLDER_MIDDLE, HOLDER_LEAF, FILLER, FILLER_MIDDLE, FILLER_TOP = range(600, 606)


@contextmanager
def chain_cdm() -> Generator[sessionmaker[Session]]:
    """
    Yield a CDM with HOLDER site FILLER.

    HOLDER_LEAF is a kind of HOLDER_MIDDLE, which is a kind of HOLDER. FILLER is a kind of
    FILLER_MIDDLE, which is a kind of FILLER_TOP. Both middle concepts are deprecated.
    """
    rows = [
        concept(HOLDER),
        concept(HOLDER_MIDDLE, invalid_reason="U", standard_concept=None),
        concept(HOLDER_LEAF),
        concept(FILLER),
        concept(FILLER_MIDDLE, invalid_reason="U", standard_concept=None),
        concept(FILLER_TOP),
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
        edge(HOLDER_MIDDLE, "Is a", HOLDER),
        edge(HOLDER_LEAF, "Is a", HOLDER_MIDDLE),
        edge(FILLER, "Is a", FILLER_MIDDLE),
        edge(FILLER_MIDDLE, "Is a", FILLER_TOP),
        edge(HOLDER, "Has finding site", FILLER),
    ]
    models = (
        Concept,
        Relationship,
        Concept_Ancestor,
        Concept_Relationship,
        Concept_Synonym,
        Vocabulary,
        RelationshipClass,
        RelationshipMapping,
    )
    with sqlite_cdm(models, rows) as sessions:
        yield sessions
