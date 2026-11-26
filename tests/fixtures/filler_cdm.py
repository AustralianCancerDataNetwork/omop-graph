"""A CDM where a stored relationship ends on a non-standard concept below a standard one."""

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
        relationship_class(PredicateKind.HIERARCHY, "Taxonomic - up", HierarchyDirection.UP),
        relationship_class(PredicateKind.HIERARCHY, "Taxonomic - down", HierarchyDirection.DOWN),
        relationship_class(PredicateKind.ATTRIBUTE, "Site"),
        classified("Subsumes", PredicateKind.HIERARCHY, "Taxonomic - down"),
        classified("Is a", PredicateKind.HIERARCHY, "Taxonomic - up", generalises=True),
        classified("Has finding site", PredicateKind.ATTRIBUTE, "Site", generalises=True),
        edge(NON_STANDARD_FILLER, "Is a", STANDARD_PARENT),
        edge(STANDARD_PARENT, "Subsumes", NON_STANDARD_FILLER),
        edge(HOLDER, "Has finding site", NON_STANDARD_FILLER),
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
