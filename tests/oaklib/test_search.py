"""Search over labels, aliases and identifiers, on a small CDM with every kind of match."""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest
from fixtures.helpers import curie
from fixtures.search_cdm import (
    CANCER,
    CONTAINING,
    LOWER,
    PERCENT,
    PREFIXED,
    RETIRED,
    SOURCE,
    SPACED_CODE,
    search_cdm,
)
from oaklib.datamodels.search import SearchConfiguration, SearchProperty, SearchTermSyntax
from omop_alchemy.cdm.query import ConceptFilter

from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.oaklib_interface import search as search_module
from omop_graph.oaklib_interface.omop_implementation import OMOPAlchemyImplementation


def config(**fields) -> SearchConfiguration:
    return SearchConfiguration(**fields)


@pytest.fixture(scope="module")
def adapter() -> Iterator[OMOPAlchemyImplementation]:
    with search_cdm() as sessions:
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


@pytest.mark.parametrize("term", ["_", "\\", "cancer%", "c_ncer"])
def test_wildcards_in_a_partial_search_match_themselves(adapter, term) -> None:
    assert search(adapter, term, is_partial=True) == []


def test_a_wildcard_character_in_a_name_is_found(adapter) -> None:
    assert search(adapter, "100%", is_partial=True) == [curie(PERCENT)]


def test_a_limit_of_zero_yields_nothing_even_for_an_identifier(adapter) -> None:
    assert search(adapter, curie(CANCER), properties=[SearchProperty.IDENTIFIER], limit=0) == []


def test_search_stays_within_the_adapters_vocabularies(adapter) -> None:
    icd = OMOPAlchemyImplementation(
        kg=adapter.kg, concept_filter=ConceptFilter(vocabularies=("ICD10CM",))
    )
    props = {"properties": [SearchProperty.ANYTHING], "is_partial": True}

    assert search(icd, "neoplasm", **props) == [curie(SOURCE)]
    assert search(icd, "cancer", **props) == []
    assert search(icd, curie(CANCER), **props) == []
    assert search(icd, f"ICD10CM:{SPACED_CODE}", **props) == [curie(SOURCE)]


def test_search_ignores_the_standardness_of_the_adapters_filter(adapter) -> None:
    standard = OMOPAlchemyImplementation(
        kg=adapter.kg, concept_filter=ConceptFilter(require_standard=True)
    )

    assert search(standard, "malignant neoplasm") == [curie(SOURCE)]


def test_configuration_without_an_effect_is_logged(adapter, caplog) -> None:
    with caplog.at_level(logging.INFO, logger=search_module.logger.name):
        search(adapter, "cancer", categories=["biolink:Disease"], is_complete=True)

    assert "categories, is_complete" in caplog.text


def test_fuzzy_search_without_a_full_text_index_adds_nothing(adapter) -> None:
    assert search(adapter, "cancer", is_fuzzy=True) == search(adapter, "cancer")


def test_multiple_terms(adapter) -> None:
    assert list(adapter.multiterm_search(["cancer", "cancer of lung"])) == [
        curie(CANCER),
        curie(LOWER),
        curie(PREFIXED),
    ]
