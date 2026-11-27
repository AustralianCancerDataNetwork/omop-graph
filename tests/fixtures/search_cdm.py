"""A small CDM with every kind of search match: exact, prefix, substring, alias, code and retired."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

from omop_alchemy.cdm.model.vocabulary.vocabulary import Vocabulary
from sqlalchemy.orm import Session, sessionmaker

from fixtures.cdm_builders import (
    classified,
    concept,
    relationship_class,
    sqlite_cdm,
    synonym,
)
from omop_graph.extensions.omop_alchemy import PredicateKind
from omop_graph.graph.language import OMOPLanguage

CANCER, PREFIXED, CONTAINING, LOWER, SOURCE, RETIRED, PERCENT = range(400, 407)
SPACED_CODE = "C34 9"


@contextmanager
def search_cdm() -> Generator[sessionmaker[Session]]:
    """Yield the session factory of the search CDM."""
    rows = [
        *(
            Vocabulary(vocabulary_id=v, vocabulary_name=v, vocabulary_concept_id=i)
            for i, v in enumerate(("SNOMED", "ICD10CM"))
        ),
        relationship_class(PredicateKind.IDENTITY, "Exact equivalence"),
        classified("Maps to", PredicateKind.IDENTITY, "Exact equivalence"),
        concept(CANCER, "Cancer"),
        concept(PREFIXED, "Cancer of lung"),
        concept(CONTAINING, "Small cell lung cancer"),
        concept(LOWER, "cancer", standard_concept=None),
        concept(
            SOURCE,
            "Malignant neoplasm",
            standard_concept=None,
            vocabulary_id="ICD10CM",
            concept_code=SPACED_CODE,
        ),
        concept(RETIRED, "Cancer old", standard_concept=None, invalid_reason="U"),
        concept(PERCENT, "Dose of 100% oxygen"),
        synonym(CONTAINING, "Oat cell carcinoma", OMOPLanguage.ENGLISH),
        synonym(CANCER, "Cancer", OMOPLanguage.ENGLISH),
    ]
    with sqlite_cdm(rows) as sessions:
        yield sessions
