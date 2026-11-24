"""Search over labels, aliases and identifiers, on a small CDM with every kind of match."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fixtures.cdm_builders import (
    classified,
    concept,
    relationship_class,
    sqlite_cdm,
    synonym,
)
from oaklib.datamodels.search import SearchConfiguration, SearchProperty, SearchTermSyntax
from omop_alchemy.cdm.model.vocabulary.concept import Concept
from omop_alchemy.cdm.model.vocabulary.concept_synonym import Concept_Synonym
from omop_alchemy.cdm.model.vocabulary.vocabulary import Vocabulary

from omop_graph.extensions.omop_alchemy import PredicateKind, RelationshipClass, RelationshipMapping
from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.oaklib_interface.codec import OMOPLanguage
from omop_graph.oaklib_interface.omop_implementation import OMOPAlchemyImplementation

CANCER, PREFIXED, CONTAINING, LOWER, SOURCE, RETIRED = range(400, 406)
SPACED_CODE = "C34.9"


def curie(concept_id: int) -> str:
    return f"OMOP:{concept_id}"


def config(**fields) -> SearchConfiguration:
    return SearchConfiguration(**fields)


@pytest.fixture(scope="module")
def adapter() -> Iterator[OMOPAlchemyImplementation]:
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
        synonym(CONTAINING, "Oat cell carcinoma", OMOPLanguage.ENGLISH),
    ]
    models = (Concept, Concept_Synonym, Vocabulary, RelationshipClass, RelationshipMapping)
    with sqlite_cdm(models, rows) as sessions:
        yield OMOPAlchemyImplementation(kg=KnowledgeGraph(sessions))


def search(adapter, term: str, **fields) -> list[str]:
    return list(adapter.basic_search(term, config(**fields)))


def test_the_default_search_is_an_exact_case_insensitive_label_match(adapter) -> None:
    assert search(adapter, "CANCER") == [curie(CANCER), curie(LOWER)]
    assert list(adapter.basic_search("cancer")) == [curie(CANCER), curie(LOWER)]


def test_partial_matches_come_exact_then_prefix_then_substring(adapter) -> None:
    assert search(adapter, "cancer", is_partial=True) == [
        curie(CANCER),
        curie(LOWER),
        curie(PREFIXED),
        curie(CONTAINING),
    ]


def test_starts_with_does_not_match_inside_a_label(adapter) -> None:
    syntax = SearchTermSyntax.STARTS_WITH

    assert search(adapter, "cancer", syntax=syntax) == [
        curie(CANCER),
        curie(LOWER),
        curie(PREFIXED),
    ]


def test_the_limit_keeps_the_best_matches(adapter) -> None:
    everything = search(adapter, "cancer", is_partial=True)

    assert search(adapter, "cancer", is_partial=True, limit=2) == everything[:2]
    assert search(adapter, "cancer", is_partial=True, limit=1) == everything[:1]


def test_aliases_are_only_searched_on_request(adapter) -> None:
    assert search(adapter, "oat cell carcinoma") == []
    assert search(adapter, "oat cell carcinoma", properties=[SearchProperty.ALIAS]) == [
        curie(CONTAINING)
    ]


def test_a_label_and_an_alias_match_is_reported_once(adapter) -> None:
    assert search(adapter, "cancer", properties=[SearchProperty.ALIAS]).count(curie(CANCER)) == 1


def test_deprecated_concepts_are_left_out_unless_requested(adapter) -> None:
    assert curie(RETIRED) not in search(adapter, "cancer old")
    assert search(adapter, "cancer old", include_obsoletes_in_results=True) == [curie(RETIRED)]


def test_an_identifier_is_found_first(adapter) -> None:
    identifier = [SearchProperty.IDENTIFIER]

    assert search(adapter, curie(CANCER), properties=identifier) == [curie(CANCER)]
    assert search(adapter, f"ICD10CM:{SPACED_CODE}", properties=identifier) == [curie(SOURCE)]
    assert search(adapter, "ICD10CM:nothing", properties=identifier) == []
    assert search(adapter, curie(RETIRED), properties=identifier) == []
    assert search(
        adapter, curie(RETIRED), properties=identifier, include_obsoletes_in_results=True
    ) == [curie(RETIRED)]


def test_anything_searches_labels_aliases_and_identifiers(adapter) -> None:
    anything = [SearchProperty.ANYTHING]

    assert search(adapter, curie(CANCER), properties=anything) == [curie(CANCER)]
    assert search(adapter, "oat cell carcinoma", properties=anything) == [curie(CONTAINING)]


@pytest.mark.parametrize(
    "syntax",
    [SearchTermSyntax.REGULAR_EXPRESSION, SearchTermSyntax.SQL, SearchTermSyntax.LUCENE],
)
def test_unsupported_syntaxes_raise(adapter, syntax) -> None:
    with pytest.raises(ValueError, match="not supported"):
        search(adapter, "cancer", syntax=syntax)


def test_fuzzy_search_without_a_full_text_index_adds_nothing(adapter) -> None:
    assert search(adapter, "cancer", is_fuzzy=True) == search(adapter, "cancer")


def test_multiple_terms(adapter) -> None:
    assert list(adapter.multiterm_search(["cancer", "cancer of lung"])) == [
        curie(CANCER),
        curie(LOWER),
        curie(PREFIXED),
    ]
