from __future__ import annotations

from collections.abc import Iterator

import pytest
from fixtures.hierarchy_cdm import CANCER, LUNG_CANCER, hierarchy_cdm

from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.graph.language import OMOPLanguage


@pytest.fixture(scope="module")
def kg() -> Iterator[KnowledgeGraph]:
    with hierarchy_cdm() as sessions:
        yield KnowledgeGraph(sessions)


def test_a_language_is_found_by_its_concept_or_its_code() -> None:
    assert OMOPLanguage.from_concept_id(OMOPLanguage.SPANISH.concept_id) is OMOPLanguage.SPANISH
    assert OMOPLanguage.from_code("fr") is OMOPLanguage.FRENCH
    assert OMOPLanguage.from_concept_id(1) is None
    assert OMOPLanguage.from_code("xx") is None


def test_the_languages_synonyms_use_come_in_a_fixed_order(kg) -> None:
    assert kg.synonym_languages() == (OMOPLanguage.ENGLISH, OMOPLanguage.SPANISH)


def test_a_translation_is_the_synonym_in_the_language(kg) -> None:
    assert kg.translations([LUNG_CANCER], OMOPLanguage.SPANISH) == {LUNG_CANCER: "Cancer de pulmon"}
    assert kg.translations([LUNG_CANCER], OMOPLanguage.ENGLISH) == {LUNG_CANCER: "Lung neoplasm"}


def test_a_concept_without_a_synonym_in_the_language_is_left_out(kg) -> None:
    assert kg.translations([CANCER, LUNG_CANCER], OMOPLanguage.FRENCH) == {}
    assert kg.translations([CANCER, LUNG_CANCER], OMOPLanguage.SPANISH) == {
        LUNG_CANCER: "Cancer de pulmon"
    }


def test_no_concepts_have_no_translations(kg) -> None:
    assert kg.translations([], OMOPLanguage.SPANISH) == {}
