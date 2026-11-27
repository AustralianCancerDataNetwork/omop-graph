"""Semantic similarity served from the strict Is a hierarchy, on the small hierarchy CDM."""

from __future__ import annotations

import math
from collections.abc import Iterator

import pytest
from fixtures.helpers import curie
from fixtures.hierarchy_cdm import (
    CANCER,
    DISEASE,
    IN_SCOPE,
    LUNG,
    LUNG_CANCER,
    OUTER_ROOT,
    SMOKING_RELATED,
    SOURCE,
    hierarchy_cdm,
)
from oaklib.datamodels.vocabulary import IS_A, OWL_THING
from omop_alchemy.cdm.query import ConceptFilter

from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.oaklib_interface.omop_implementation import OMOPAlchemyImplementation


@pytest.fixture(scope="module")
def adapter() -> Iterator[OMOPAlchemyImplementation]:
    with hierarchy_cdm() as sessions:
        yield OMOPAlchemyImplementation(kg=KnowledgeGraph(sessions))


def test_the_most_recent_common_ancestors_have_none_of_the_common_ancestors_below(adapter) -> None:
    found = list(adapter.most_recent_common_ancestors(curie(LUNG_CANCER), curie(SMOKING_RELATED)))

    assert found == [curie(SMOKING_RELATED)]


def test_concepts_without_a_common_ancestor_share_owl_thing_when_asked(adapter) -> None:
    unrelated = (curie(LUNG_CANCER), curie(LUNG))

    assert list(adapter.most_recent_common_ancestors(*unrelated)) == [OWL_THING]
    assert list(adapter.most_recent_common_ancestors(*unrelated, include_owl_thing=False)) == []


def test_the_common_ancestors_are_those_in_scope(adapter) -> None:
    found = set(
        adapter.common_ancestors(curie(LUNG_CANCER), curie(CANCER), include_owl_thing=False)
    )

    assert found == {curie(CANCER), curie(DISEASE), curie(OUTER_ROOT)}
    assert OWL_THING in set(adapter.common_ancestors(curie(LUNG_CANCER), curie(CANCER)))


def test_the_common_descendants_are_below_both(adapter) -> None:
    assert list(adapter.common_descendants(curie(CANCER), curie(SMOKING_RELATED))) == [
        curie(LUNG_CANCER)
    ]


def test_an_unknown_curie_has_no_common_ancestors(adapter) -> None:
    assert list(adapter.common_ancestors(curie(CANCER), "OMOP:abc", include_owl_thing=False)) == []


def test_information_content_is_the_log_of_the_share_of_concepts_below(adapter) -> None:
    scores = dict(adapter.information_content_scores([curie(DISEASE), curie(LUNG_CANCER), "X:1"]))

    assert scores == {
        curie(DISEASE): pytest.approx(-math.log2(4 / IN_SCOPE)),
        curie(LUNG_CANCER): pytest.approx(math.log2(IN_SCOPE)),
    }
    assert adapter.get_information_content(curie(DISEASE)) == pytest.approx(scores[curie(DISEASE)])


def test_owl_thing_has_no_information_and_is_its_own_only_ancestor(adapter) -> None:
    assert adapter.get_information_content(OWL_THING) == 0.0
    assert dict(adapter.information_content_scores([OWL_THING, curie(DISEASE)])).keys() == {
        OWL_THING,
        curie(DISEASE),
    }
    assert list(adapter.ancestors(OWL_THING)) == [OWL_THING]
    assert list(adapter.ancestors(OWL_THING, reflexive=False)) == []


def test_the_information_content_method_names_the_hierarchy(adapter) -> None:
    method = adapter.information_content_method()

    assert method.source == "omop-graph"
    assert method.closure_predicates == [IS_A]


def test_associations_are_not_a_source_of_information_content(adapter) -> None:
    with pytest.raises(ValueError, match="associations"):
        list(adapter.information_content_scores([curie(DISEASE)], use_associations=True))


def test_only_is_a_can_be_followed(adapter) -> None:
    with pytest.raises(ValueError, match="rdfs:subClassOf"):
        adapter.pairwise_similarity(curie(CANCER), curie(LUNG_CANCER), predicates=["BFO:0000050"])
    assert adapter.pairwise_similarity(curie(CANCER), curie(LUNG_CANCER), predicates=[IS_A])


def test_a_bad_argument_raises_at_the_call_and_not_when_the_result_is_read(adapter) -> None:
    with pytest.raises(ValueError, match="rdfs:subClassOf"):
        adapter.most_recent_common_ancestors(
            curie(CANCER), curie(LUNG_CANCER), predicates=["BFO:0000050"]
        )
    with pytest.raises(ValueError, match="associations"):
        adapter.information_content_scores([curie(DISEASE)], use_associations=True)


def test_the_common_descendants_of_a_stand_in_are_those_of_its_standard_concept(adapter) -> None:
    standardising = OMOPAlchemyImplementation(kg=adapter.kg, standardise=True)
    pair = (curie(SOURCE), curie(CANCER))

    assert list(standardising.common_descendants(*pair)) == [curie(LUNG_CANCER)]
    assert curie(LUNG_CANCER) not in set(adapter.common_descendants(*pair))


def test_pairwise_similarity_carries_the_scores_of_the_graph(adapter) -> None:
    similarity = adapter.pairwise_similarity(curie(LUNG_CANCER), curie(CANCER))
    ancestor_ic = -math.log2(2 / IN_SCOPE)

    assert similarity is not None
    assert similarity.ancestor_id == curie(CANCER)
    assert similarity.jaccard_similarity == pytest.approx(0.6)
    assert similarity.ancestor_information_content == pytest.approx(ancestor_ic)
    assert similarity.subject_information_content == pytest.approx(math.log2(IN_SCOPE))
    assert similarity.object_information_content == pytest.approx(ancestor_ic)
    assert similarity.phenodigm_score == pytest.approx(math.sqrt(0.6 * ancestor_ic))


def test_unrelated_concepts_have_no_ancestor(adapter) -> None:
    similarity = adapter.pairwise_similarity(curie(LUNG_CANCER), curie(LUNG))

    assert similarity is not None
    assert similarity.ancestor_id is None
    assert similarity.jaccard_similarity == 0.0


def test_a_pair_below_a_minimum_is_not_scored(adapter) -> None:
    pair = (curie(LUNG_CANCER), curie(CANCER))

    assert adapter.pairwise_similarity(*pair, min_jaccard_similarity=0.9) is None
    assert adapter.pairwise_similarity(*pair, min_ancestor_information_content=99.0) is None
    assert adapter.pairwise_similarity(*pair, min_jaccard_similarity=0.5) is not None


def test_an_unknown_concept_has_no_similarity(adapter) -> None:
    assert adapter.pairwise_similarity(curie(CANCER), "OMOP:abc") is None


def test_the_set_wise_similarity_is_built_from_the_pairs(adapter) -> None:
    similarity = adapter.termset_pairwise_similarity(
        [curie(LUNG_CANCER), curie(LUNG)], [curie(CANCER), curie(SMOKING_RELATED)]
    )

    assert similarity.subject_best_matches[curie(LUNG_CANCER)].match_target in {
        curie(CANCER),
        curie(SMOKING_RELATED),
    }
    assert similarity.average_score > 0


def test_the_information_content_counts_the_concepts_the_adapter_lists() -> None:
    with hierarchy_cdm() as sessions:
        adapter = OMOPAlchemyImplementation(
            kg=KnowledgeGraph(sessions), concept_filter=ConceptFilter(vocabularies=("SNOMED",))
        )
        listed = len(list(adapter.entities()))

        [(_, score)] = adapter.information_content_scores([curie(LUNG)])

    assert score == pytest.approx(math.log2(listed))


def test_a_non_standard_concept_stands_in_for_its_standard_concept_when_asked(adapter) -> None:
    standardising = OMOPAlchemyImplementation(kg=adapter.kg, standardise=True)
    pair = (curie(SOURCE), curie(CANCER))

    with pytest.raises(ValueError, match="outside the concept_filter"):
        adapter.pairwise_similarity(*pair)
    stood_in = standardising.pairwise_similarity(*pair)

    assert stood_in is not None
    assert stood_in.jaccard_similarity
    assert curie(CANCER) in set(standardising.common_ancestors(*pair))
    assert curie(CANCER) not in set(adapter.common_ancestors(*pair))
    assert curie(CANCER) in list(standardising.most_recent_common_ancestors(*pair))
