from __future__ import annotations

from collections.abc import Iterator

import pytest
from fixtures.search_cdm import (
    CANCER,
    CONTAINING,
    LOWER,
    PREFIXED,
    RETIRED,
    SOURCE,
    search_cdm,
)
from omop_alchemy.cdm.query import ConceptFilter

from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.graph.nodes import LabelMatchKind
from omop_graph.graph.scope import every_concept, narrowed_to_vocabularies

EXACT = LabelMatchKind.EXACT
STARTS = LabelMatchKind.STARTS_WITH
PARTIAL = LabelMatchKind.PARTIAL
RANKED = (EXACT, STARTS, PARTIAL)


@pytest.fixture(scope="module")
def kg() -> Iterator[KnowledgeGraph]:
    with search_cdm() as sessions:
        yield KnowledgeGraph(sessions)


def test_an_exact_search_ignores_case(kg) -> None:
    assert list(kg.search("CANCER", concept_filter=ConceptFilter(require_active=True))) == [
        CANCER,
        LOWER,
    ]


def test_the_kinds_rank_exact_then_prefix_then_substring(kg) -> None:
    found = list(kg.search("cancer", RANKED, concept_filter=ConceptFilter(require_active=True)))

    assert found == [CANCER, LOWER, PREFIXED, CONTAINING]


def test_each_concept_is_found_once(kg) -> None:
    found = list(kg.search("cancer", RANKED, synonyms=True))

    assert len(found) == len(set(found))


def test_the_limit_keeps_the_best_matches(kg) -> None:
    concept_filter = ConceptFilter(require_active=True)
    everything = list(kg.search("cancer", RANKED, concept_filter=concept_filter))

    assert (
        list(kg.search("cancer", RANKED, concept_filter=concept_filter, limit=2)) == everything[:2]
    )
    assert list(kg.search("cancer", RANKED, concept_filter=concept_filter, limit=0)) == []


def test_synonyms_are_searched_on_request(kg) -> None:
    assert list(kg.search("oat cell carcinoma")) == []
    assert list(kg.search("oat cell carcinoma", synonyms=True)) == [CONTAINING]


def test_named_identifiers_come_first_when_in_scope(kg) -> None:
    concept_filter = ConceptFilter(require_active=True)

    assert list(
        kg.search("cancer", RANKED, concept_filter=concept_filter, identifiers=[PREFIXED])
    ) == [
        PREFIXED,
        CANCER,
        LOWER,
        CONTAINING,
    ]
    assert list(kg.search("nothing", concept_filter=concept_filter, identifiers=[RETIRED])) == []


def test_identifiers_count_towards_the_limit(kg) -> None:
    concept_filter = ConceptFilter(require_active=True)

    assert list(
        kg.search("cancer", RANKED, concept_filter=concept_filter, limit=1, identifiers=[PREFIXED])
    ) == [PREFIXED]


def test_a_scope_narrows_the_matches(kg) -> None:
    concept_filter = ConceptFilter(vocabularies=("ICD10CM",))

    assert list(kg.search("malignant neoplasm", concept_filter=concept_filter)) == [SOURCE]
    assert list(kg.search("cancer", concept_filter=concept_filter)) == []


def test_every_concept_drops_standardness_and_validity() -> None:
    concept_filter = ConceptFilter(
        vocabularies=("SNOMED",),
        domains=("Condition",),
        require_standard=True,
        include_classification=True,
        require_active=True,
    )

    assert every_concept(concept_filter) == ConceptFilter(
        vocabularies=("SNOMED",), domains=("Condition",)
    )


def test_narrowing_keeps_the_vocabularies_both_allow() -> None:
    concept_filter = ConceptFilter(vocabularies=("SNOMED", "LOINC"))

    assert narrowed_to_vocabularies(concept_filter, ["LOINC", "ICD10CM", "LOINC"]).vocabularies == (
        "LOINC",
    )
    assert narrowed_to_vocabularies(concept_filter, ["ICD10CM"]).vocabularies == ()


def test_narrowing_a_scope_without_vocabularies_takes_them_all() -> None:
    assert narrowed_to_vocabularies(ConceptFilter(), ["ICD10CM", "SNOMED"]).vocabularies == (
        "ICD10CM",
        "SNOMED",
    )
