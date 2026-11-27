from __future__ import annotations

from fixtures.mock_cdm import CONCEPT_META_ID, PARENT_CANCER_ID

from omop_graph.graph.kg import KnowledgeGraph


def test_get_potential_ancestors_batch_returns_only_real_ancestors(
    mock_cdm_kg: KnowledgeGraph,
) -> None:
    # CONCEPT_META_ID is a real concept but has no concept_ancestor row for 196653,
    # so it should be silently absent from the result, unlike PARENT_CANCER_ID.
    matches = mock_cdm_kg.get_potential_ancestors_batch(
        child_ids=(196653,), parent_ids=(PARENT_CANCER_ID, CONCEPT_META_ID)
    )

    assert set(matches[196653]) == {PARENT_CANCER_ID}
    match = matches[196653][PARENT_CANCER_ID]
    assert match.ancestor_concept_id == PARENT_CANCER_ID
    assert match.descendant_concept_id == 196653
    assert match.min_levels_of_separation == 2


def test_get_potential_ancestors_batch_no_matches_returns_empty_dict(
    mock_cdm_kg: KnowledgeGraph,
) -> None:
    matches = mock_cdm_kg.get_potential_ancestors_batch(
        child_ids=(196653,), parent_ids=(CONCEPT_META_ID,)
    )

    assert matches == {}


def test_get_potential_ancestors_batch_empty_parent_ids_returns_empty_dict(
    mock_cdm_kg: KnowledgeGraph,
) -> None:
    matches = mock_cdm_kg.get_potential_ancestors_batch(child_ids=(196653,), parent_ids=())

    assert matches == {}
