"""A small CDM where the strict ``Is a`` hierarchy differs from ``concept_ancestor``."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy.orm import Session, sessionmaker

from fixtures.cdm_builders import (
    ancestor,
    classified,
    concept,
    edge,
    relationship,
    relationship_class,
    sqlite_cdm,
    synonym,
)
from omop_graph.extensions.omop_alchemy import (
    HierarchyDirection,
    PredicateKind,
)
from omop_graph.graph.language import OMOPLanguage

(
    DISEASE,
    CANCER,
    SMOKING_RELATED,
    LUNG_CANCER,
    RETIRED_PARENT,
    COHORT,
    LUNG,
    THORAX_ORGAN,
    LOOP_A,
    LOOP_B,
) = range(200, 210)
OUTDATED, OUTER_ROOT, SOURCE, SOURCE_TWO, ORPHAN, SOURCE_LUNG = range(210, 216)
IN_SCOPE = 11


@contextmanager
def hierarchy_cdm() -> Generator[sessionmaker[Session]]:
    """Yield the session factory of the small hierarchy CDM."""
    rows = [
        *(concept(concept_id) for concept_id in range(DISEASE, LOOP_B + 1)),
        concept(OUTDATED, invalid_reason="U", standard_concept=None),
        concept(OUTER_ROOT),
        *(
            concept(concept_id, standard_concept=None)
            for concept_id in (SOURCE, SOURCE_TWO, ORPHAN, SOURCE_LUNG)
        ),
        relationship("Is a", "Subsumes"),
        relationship("Subsumes", "Is a"),
        relationship("Has finding site", "Finding site of"),
        relationship("Maps to", "Mapped from"),
        relationship("Mapped from", "Maps to"),
        relationship_class(PredicateKind.IDENTITY, "Exact"),
        relationship_class(PredicateKind.HIERARCHY, "Taxonomic - up"),
        relationship_class(PredicateKind.HIERARCHY, "Taxonomic - down"),
        relationship_class(PredicateKind.ATTRIBUTE, "Site"),
        classified("Maps to", PredicateKind.IDENTITY, "Exact"),
        classified("Mapped from", PredicateKind.IDENTITY, "Exact"),
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
        classified("Finding site of", PredicateKind.ATTRIBUTE, "Site"),
        edge(SOURCE, "Maps to", LUNG_CANCER),
        edge(SOURCE_TWO, "Maps to", LUNG_CANCER),
        edge(SOURCE_TWO, "Maps to", LUNG),
        edge(SOURCE_LUNG, "Maps to", LUNG),
        edge(LUNG_CANCER, "Mapped from", SOURCE),
        edge(DISEASE, "Subsumes", CANCER),
        edge(CANCER, "Subsumes", LUNG_CANCER),
        edge(CANCER, "Is a", DISEASE),
        edge(SMOKING_RELATED, "Is a", DISEASE),
        edge(LUNG_CANCER, "Is a", CANCER),
        edge(LUNG_CANCER, "Is a", SMOKING_RELATED),
        edge(LUNG_CANCER, "Is a", RETIRED_PARENT, invalid_reason="D"),
        edge(CANCER, "Is a", OUTDATED),
        edge(OUTDATED, "Is a", OUTER_ROOT),
        edge(LUNG, "Is a", THORAX_ORGAN),
        edge(LUNG_CANCER, "Has finding site", LUNG),
        edge(LOOP_A, "Is a", LOOP_B),
        edge(LOOP_B, "Is a", LOOP_A),
        synonym(LUNG_CANCER, "Lung neoplasm", OMOPLanguage.ENGLISH),
        synonym(LUNG_CANCER, "Cancer de pulmon", OMOPLanguage.SPANISH),
        ancestor(DISEASE, LUNG_CANCER),
        ancestor(COHORT, LUNG_CANCER),
    ]
    with sqlite_cdm(rows) as sessions:
        yield sessions
