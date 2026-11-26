"""How the adapter resolves the CURIEs it is given, on the small hierarchy CDM."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest
from fixtures.hierarchy_cdm import CANCER, LUNG_CANCER, hierarchy_cdm
from oaklib.datamodels.vocabulary import IS_A, OWL_THING

from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.oaklib_interface.codec import OMOPCodec
from omop_graph.oaklib_interface.omop_implementation import OMOPAlchemyImplementation

MISSING = "OMOP:999999"
UNCODED = ("OMOP:99999999999999999999", "OMOP:03", "OMOP:" + "9" * 5000, "OMOP:-1", "OMOP:1.5")


def curie(concept_id: int) -> str:
    return f"OMOP:{concept_id}"


@pytest.fixture(scope="module")
def adapter() -> Iterator[OMOPAlchemyImplementation]:
    with hierarchy_cdm() as sessions:
        yield OMOPAlchemyImplementation(kg=KnowledgeGraph(sessions))


def test_a_concept_is_as_similar_to_itself_as_it_can_be(adapter) -> None:
    x = curie(LUNG_CANCER)

    similarity = adapter.pairwise_similarity(x, x)

    assert similarity is not None
    assert similarity.jaccard_similarity == 1.0
    assert similarity.ancestor_id == x
    assert list(adapter.most_recent_common_ancestors(x, x)) == [x]
    assert x in set(adapter.common_ancestors(x, x)) - {OWL_THING}
    assert set(adapter.common_descendants(x, x)) == {x}


def test_a_term_set_matches_each_term_with_itself_first(adapter) -> None:
    x = curie(LUNG_CANCER)

    result = adapter.termset_pairwise_similarity([x], [x])

    assert result.subject_best_matches[x].match_target == x
    assert result.best_score == adapter.pairwise_similarity(x, x).ancestor_information_content


def test_two_spellings_of_one_concept_both_get_their_synonyms(adapter) -> None:
    spellings = [curie(LUNG_CANCER), f"SNOMED:C{LUNG_CANCER}"]

    synonyms = {
        spelling: {
            value.val
            for subject, value in adapter.synonym_property_values(spellings)
            if subject == spelling
        }
        for spelling in spellings
    }
    labelled = {
        subject for subject, name, lang in adapter.multilingual_labels(spellings) if lang == "es"
    }

    assert synonyms[spellings[0]] == synonyms[spellings[1]] != set()
    assert labelled == set(spellings)


def test_a_start_that_is_an_ancestor_of_another_is_dropped_when_not_reflexive(adapter) -> None:
    starts = [curie(CANCER), curie(LUNG_CANCER)]

    assert curie(CANCER) not in set(adapter.ancestors(starts, reflexive=False))
    assert curie(CANCER) in set(adapter.ancestors(starts))


@pytest.mark.parametrize("text", UNCODED)
def test_a_curie_that_is_not_a_concept_id_names_no_concept(adapter, text) -> None:
    assert adapter.codec.resolve_concept(text) is None
    assert list(adapter.relationships(subjects=[text])) == []
    assert list(adapter.ancestors(text)) == []


def test_the_largest_concept_id_is_a_concept_id() -> None:
    codec = OMOPCodec.__new__(OMOPCodec)

    assert codec.concept_candidates(f"OMOP:{OMOPCodec.MAX_CONCEPT_ID}") == (
        OMOPCodec.MAX_CONCEPT_ID,
    )
    assert codec.concept_candidates(f"OMOP:{OMOPCodec.MAX_CONCEPT_ID + 1}") == ()


@pytest.mark.parametrize("value", [True, "1", 1.5])
def test_a_concept_curie_is_made_from_an_integer_only(adapter, value) -> None:
    with pytest.raises(TypeError):
        adapter.codec.concept_curie(value)


CALLS: dict[str, tuple[Callable[[OMOPAlchemyImplementation], object], object]] = {
    "label": (lambda a: a.label(MISSING), None),
    "labels": (lambda a: list(a.labels([MISSING])), [(MISSING, None)]),
    "ancestors": (lambda a: list(a.ancestors(MISSING)), []),
    "descendants": (lambda a: list(a.descendants(MISSING)), []),
    "node": (lambda a: a.node(MISSING), None),
    "hierarchical_parents": (lambda a: a.hierarchical_parents(MISSING), []),
    "synonym_property_values": (lambda a: list(a.synonym_property_values(MISSING)), []),
    "multilingual_labels": (
        lambda a: list(a.multilingual_labels([MISSING])),
        [(MISSING, None, None)],
    ),
    "relationships": (lambda a: list(a.relationships(subjects=[MISSING])), []),
    "information_content_scores": (lambda a: list(a.information_content_scores([MISSING])), []),
    "pairwise_similarity": (lambda a: a.pairwise_similarity(MISSING, curie(CANCER)), None),
    "common_ancestors": (
        lambda a: list(a.common_ancestors(MISSING, curie(CANCER))),
        [OWL_THING],
    ),
    "common_descendants": (lambda a: list(a.common_descendants(MISSING, curie(CANCER))), []),
    "most_recent_common_ancestors": (
        lambda a: list(a.most_recent_common_ancestors(MISSING, curie(CANCER))),
        [OWL_THING],
    ),
    "sssom_mappings": (lambda a: list(a.sssom_mappings([MISSING])), []),
    "get_transitive_mappings_by_curie": (
        lambda a: list(a.get_transitive_mappings_by_curie(MISSING)),
        [],
    ),
    "relationships_metadata": (
        lambda a: list(a.relationships_metadata([(MISSING, IS_A, curie(CANCER))])),
        [((MISSING, IS_A, curie(CANCER)), [])],
    ),
}


@pytest.mark.parametrize("name", CALLS)
def test_a_curie_of_a_concept_that_does_not_exist_is_answered_as_oaklib_does(adapter, name) -> None:
    call, expected = CALLS[name]

    assert call(adapter) == expected
