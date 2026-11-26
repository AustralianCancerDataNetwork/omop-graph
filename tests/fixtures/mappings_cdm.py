"""
A small CDM with every kind of mapping edge: exact, near, associated and version lineage.

The exact edges are of every cardinality: one source to one standard concept, three sources
to the STANDARD concept, and a source that splits into two with one of them shared. The edges
of OTHER_SOURCE start on the date Athena gives to relationships without one.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from datetime import date

from omop_alchemy.cdm.model.vocabulary.vocabulary import Vocabulary
from sqlalchemy.orm import Session, sessionmaker

from fixtures.cdm_builders import (
    classified,
    concept,
    edge_pair,
    relationship,
    relationship_class,
    sqlite_cdm,
)
from omop_graph.extensions.relationship_classification import (
    PredicateKind,
)

STANDARD, SOURCE_ICD, SOURCE_READ, BRAND, INGREDIENT, REPLACED, ICDO, SPACED = range(300, 308)
UNIQUE_SOURCE, UNIQUE_STANDARD, SPLIT_SOURCE, SPLIT_A, SPLIT_B, OTHER_SOURCE = range(308, 314)
PAIRS = (
    ("Maps to", "Mapped from", PredicateKind.IDENTITY, "Exact equivalence"),
    ("Brand name of", "Has brand name", PredicateKind.IDENTITY, "Near equivalence"),
    ("ICDO - SNOMED", "SNOMED - ICDO", PredicateKind.ASSOCIATION, "Mapping"),
    ("Concept replaced by", "Concept replaces", PredicateKind.IDENTITY, "Version lineage"),
)


@contextmanager
def mappings_cdm() -> Generator[sessionmaker[Session]]:
    """Yield the session factory of the mappings CDM."""
    vocabularies = ("SNOMED", "ICD10CM", "Read", "RxNorm", "ICDO3", "Nebraska Lexicon")
    rows = [
        concept(STANDARD),
        concept(SOURCE_ICD, vocabulary_id="ICD10CM", standard_concept=None),
        concept(SOURCE_READ, vocabulary_id="Read", standard_concept=None),
        concept(BRAND, vocabulary_id="RxNorm", standard_concept=None),
        concept(INGREDIENT, vocabulary_id="RxNorm"),
        concept(REPLACED, standard_concept=None, invalid_reason="U"),
        concept(ICDO, vocabulary_id="ICDO3", standard_concept=None),
        concept(SPACED, vocabulary_id="Nebraska Lexicon", standard_concept=None),
        concept(UNIQUE_SOURCE, vocabulary_id="ICD10CM", standard_concept=None),
        concept(UNIQUE_STANDARD),
        concept(SPLIT_SOURCE, vocabulary_id="ICD10CM", standard_concept=None),
        concept(SPLIT_A),
        concept(SPLIT_B),
        concept(OTHER_SOURCE, vocabulary_id="ICD10CM", standard_concept=None),
        *(
            Vocabulary(vocabulary_id=v, vocabulary_name=v, vocabulary_concept_id=i)
            for i, v in enumerate(vocabularies)
        ),
        *(
            relationship(r, reverse)
            for pair in PAIRS
            for r, reverse in ((pair[0], pair[1]), (pair[1], pair[0]))
        ),
        *(relationship_class(kind, subkind) for kind, subkind in {(p[2], p[3]) for p in PAIRS}),
        *(
            classified(r, kind, subkind)
            for forward, backward, kind, subkind in PAIRS
            for r in (forward, backward)
        ),
        *edge_pair(SOURCE_ICD, "Maps to", "Mapped from", STANDARD),
        *edge_pair(SOURCE_READ, "Maps to", "Mapped from", STANDARD),
        *edge_pair(SOURCE_READ, "Maps to", "Mapped from", REPLACED, invalid_reason="D"),
        *edge_pair(STANDARD, "Maps to", "Mapped from", STANDARD),
        *edge_pair(BRAND, "Brand name of", "Has brand name", INGREDIENT),
        *edge_pair(ICDO, "ICDO - SNOMED", "SNOMED - ICDO", STANDARD),
        *edge_pair(REPLACED, "Concept replaced by", "Concept replaces", STANDARD),
        *edge_pair(SPACED, "Maps to", "Mapped from", STANDARD),
        *edge_pair(UNIQUE_SOURCE, "Maps to", "Mapped from", UNIQUE_STANDARD),
        *edge_pair(SPLIT_SOURCE, "Maps to", "Mapped from", SPLIT_A),
        *edge_pair(SPLIT_SOURCE, "Maps to", "Mapped from", SPLIT_B),
        *edge_pair(OTHER_SOURCE, "Maps to", "Mapped from", SPLIT_A, start=date(1970, 1, 1)),
    ]
    with sqlite_cdm(rows) as sessions:
        yield sessions
