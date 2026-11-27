from __future__ import annotations

import math
from collections.abc import Iterator

import pytest
from fixtures.hierarchy_cdm import (
    CANCER,
    DISEASE,
    IN_SCOPE,
    LUNG,
    LUNG_CANCER,
    ORPHAN,
    OUTER_ROOT,
    SMOKING_RELATED,
    SOURCE,
    SOURCE_TWO,
    hierarchy_cdm,
)
from omop_alchemy.cdm.query import ConceptFilter

from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.graph.scope import ALL_CONCEPTS
from omop_graph.graph.structural_similarity import ICSource


@pytest.fixture(scope="module")
def kg() -> Iterator[KnowledgeGraph]:
    with hierarchy_cdm() as sessions:
        yield KnowledgeGraph(sessions)


def test_a_concept_is_its_own_common_ancestor_with_a_descendant(kg) -> None:
    assert kg.common_ancestors(LUNG_CANCER, SMOKING_RELATED) == {SMOKING_RELATED, DISEASE}
    assert kg.most_recent_common_ancestors(LUNG_CANCER, SMOKING_RELATED) == {SMOKING_RELATED}


def test_siblings_share_their_parent_as_the_most_recent_common_ancestor(kg) -> None:
    assert kg.common_ancestors(CANCER, SMOKING_RELATED) == {DISEASE}
    assert kg.most_recent_common_ancestors(CANCER, SMOKING_RELATED) == {DISEASE}


def test_concepts_in_separate_hierarchies_share_nothing(kg) -> None:
    assert kg.common_ancestors(LUNG_CANCER, LUNG) == frozenset()
    assert kg.most_recent_common_ancestors(LUNG_CANCER, LUNG) == frozenset()
    assert kg.common_descendants(LUNG_CANCER, LUNG) == frozenset()


def test_common_descendants_are_below_both(kg) -> None:
    assert kg.common_descendants(CANCER, SMOKING_RELATED) == {LUNG_CANCER}
    assert kg.common_descendants(DISEASE, LUNG_CANCER) == {LUNG_CANCER}


def test_the_scope_hides_concepts_but_the_walk_passes_through_them(kg) -> None:
    assert OUTER_ROOT in kg.common_ancestors(LUNG_CANCER, CANCER)
    assert kg.common_ancestors(
        LUNG_CANCER, CANCER, concept_filter=ALL_CONCEPTS
    ) > kg.common_ancestors(LUNG_CANCER, CANCER)


def test_jaccard_is_common_over_all_ancestors(kg) -> None:
    assert kg.jaccard_similarity(CANCER, SMOKING_RELATED) == pytest.approx(1 / 4)
    assert kg.jaccard_similarity(LUNG_CANCER, LUNG) == 0.0
    assert kg.jaccard_similarity(CANCER, CANCER) == 1.0


def test_information_content_falls_with_the_descendants(kg) -> None:
    scores = kg.information_content([DISEASE, CANCER, LUNG_CANCER])

    assert scores[DISEASE] == pytest.approx(-math.log2(4 / IN_SCOPE))
    assert scores[CANCER] == pytest.approx(-math.log2(2 / IN_SCOPE))
    assert scores[LUNG_CANCER] == pytest.approx(math.log2(IN_SCOPE))
    assert scores[DISEASE] < scores[CANCER] < scores[LUNG_CANCER]


def test_a_concept_outside_the_scope_has_no_information_content(kg) -> None:
    with pytest.raises(ValueError, match=rf"\[{SOURCE}\] are outside the concept_filter"):
        kg.information_content([SOURCE])


def test_a_pair_with_a_concept_outside_the_scope_is_refused_unless_it_is_standardised(kg) -> None:
    with pytest.raises(ValueError, match="outside the concept_filter"):
        kg.pairwise_similarity(SOURCE, CANCER)


def test_a_standardised_pair_scores_the_one_standard_concept_of_an_outside_concept(kg) -> None:
    similarity = kg.pairwise_similarity(SOURCE, CANCER, standardise=True)
    standard = kg.information_content([LUNG_CANCER])[LUNG_CANCER]

    assert similarity.information_content_a == standard
    assert similarity.information_content_b == kg.information_content([CANCER])[CANCER]


def test_a_standardised_concept_with_several_standard_concepts_cannot_be_scored(kg) -> None:
    with pytest.raises(ValueError, match="several standard concepts"):
        kg.pairwise_similarity(SOURCE_TWO, CANCER, standardise=True)
    with pytest.raises(ValueError, match="several standard concepts"):
        kg.jaccard_similarity(SOURCE_TWO, CANCER, standardise=True)


def test_a_standardised_concept_without_a_standard_concept_cannot_be_scored(kg) -> None:
    with pytest.raises(ValueError, match="outside the concept_filter"):
        kg.pairwise_similarity(ORPHAN, CANCER, standardise=True)


def test_a_jaccard_of_a_concept_outside_the_scope_is_refused_unless_it_is_standardised(kg) -> None:
    with pytest.raises(ValueError, match="outside the concept_filter"):
        kg.jaccard_similarity(SOURCE, CANCER)
    assert kg.jaccard_similarity(SOURCE, CANCER, standardise=True) > 0


def test_a_concept_the_scope_includes_never_scores_below_zero(kg) -> None:
    everything = ConceptFilter(concept_ids=(DISEASE,))
    score = kg.information_content([DISEASE], concept_filter=everything)[DISEASE]

    assert score == 0.0
    assert math.copysign(1.0, score) == 1.0


def test_the_information_content_is_the_same_when_read_again(kg) -> None:
    assert kg.information_content([DISEASE]) == kg.information_content([DISEASE])


def test_the_counts_are_kept_for_each_scope_and_source(kg) -> None:
    narrow = ConceptFilter(concept_ids=(DISEASE, CANCER))
    broad = kg.information_content([DISEASE])[DISEASE]

    assert kg.information_content([DISEASE], concept_filter=narrow)[DISEASE] == 0.0
    assert kg.information_content([DISEASE])[DISEASE] == broad
    assert kg.information_content([DISEASE], ICSource.CONCEPT_ANCESTOR)[DISEASE] != broad


def test_the_concept_ancestor_table_can_count_the_descendants(kg) -> None:
    scores = kg.information_content([DISEASE], ICSource.CONCEPT_ANCESTOR)

    assert scores[DISEASE] == pytest.approx(-math.log2(2 / IN_SCOPE))


def test_no_concepts_have_no_information_content(kg) -> None:
    assert kg.information_content([]) == {}


def test_pairwise_similarity_scores_through_the_most_recent_ancestor(kg) -> None:
    similarity = kg.pairwise_similarity(LUNG_CANCER, CANCER)
    ancestor_ic = -math.log2(2 / IN_SCOPE)

    assert similarity.common_ancestors == {CANCER, DISEASE, OUTER_ROOT}
    assert similarity.most_recent_common_ancestors == {CANCER}
    assert similarity.ancestor_id == CANCER
    assert similarity.jaccard == pytest.approx(0.6)
    assert similarity.ancestor_information_content == pytest.approx(ancestor_ic)
    assert similarity.information_content_a == pytest.approx(math.log2(IN_SCOPE))
    assert similarity.information_content_b == pytest.approx(ancestor_ic)
    assert similarity.phenodigm == pytest.approx(math.sqrt(0.6 * ancestor_ic))


def test_pairwise_similarity_is_symmetric_in_its_scores(kg) -> None:
    forward = kg.pairwise_similarity(CANCER, SMOKING_RELATED)
    backward = kg.pairwise_similarity(SMOKING_RELATED, CANCER)

    assert forward.jaccard == backward.jaccard
    assert forward.ancestor_id == backward.ancestor_id == DISEASE
    assert forward.phenodigm == pytest.approx(backward.phenodigm)


def test_unrelated_concepts_score_zero_without_an_ancestor(kg) -> None:
    similarity = kg.pairwise_similarity(LUNG_CANCER, LUNG)

    assert similarity.ancestor_id is None
    assert similarity.ancestor_information_content == 0.0
    assert similarity.jaccard == 0.0
    assert similarity.phenodigm == 0.0


def test_a_concept_against_itself_is_as_similar_as_it_can_be(kg) -> None:
    similarity = kg.pairwise_similarity(CANCER, CANCER)

    assert similarity.jaccard == 1.0
    assert similarity.most_recent_common_ancestors == {CANCER}
    assert similarity.ancestor_information_content == similarity.information_content_a
