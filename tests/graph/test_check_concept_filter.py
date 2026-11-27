"""The valid domains and vocabularies come from their own tables, not from the concepts."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fixtures.cdm_builders import classified, concept, relationship, relationship_class, sqlite_cdm
from omop_alchemy.cdm.model.vocabulary import Domain, Vocabulary
from omop_alchemy.cdm.query import ConceptFilter

from omop_graph.extensions.relationship_classification import (
    PredicateKind,
)
from omop_graph.graph.kg import KnowledgeGraph


@pytest.fixture(scope="module")
def kg() -> Iterator[KnowledgeGraph]:
    rows = [
        Domain(domain_id="Condition", domain_name="Condition", domain_concept_id=19),
        Domain(domain_id="Unused", domain_name="Unused", domain_concept_id=20),
        Vocabulary(vocabulary_id="SNOMED", vocabulary_name="SNOMED", vocabulary_concept_id=1),
        Vocabulary(vocabulary_id="Unused", vocabulary_name="Unused", vocabulary_concept_id=2),
        concept(1),
        relationship("Maps to", "Mapped from"),
        relationship_class(PredicateKind.IDENTITY, "Exact"),
        classified("Maps to", PredicateKind.IDENTITY, "Exact"),
    ]
    with sqlite_cdm(rows) as sessions:
        yield KnowledgeGraph(sessions)


def test_a_domain_or_vocabulary_of_the_tables_is_valid_even_without_concepts(kg) -> None:
    kg.check_concept_filter(ConceptFilter(domains=("Unused",), vocabularies=("Unused",)))


def test_a_domain_or_vocabulary_that_is_in_no_table_is_refused_with_the_valid_ones(kg) -> None:
    with pytest.raises(ValueError, match=r"Invalid domain.*Available domains: \['Condition'"):
        kg.check_concept_filter(ConceptFilter(domains=("Nope",)))
    with pytest.raises(ValueError, match=r"Invalid vocabulary.*Available vocabularies: \['SNOMED'"):
        kg.check_concept_filter(ConceptFilter(vocabularies=("Nope",)))
