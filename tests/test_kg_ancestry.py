from __future__ import annotations

from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.graph.nodes import LabelMatchKind
from omop_graph.graph.paths import find_standard_paths
from omop_graph.reasoning.resolvers.resolvers import CandidateHit
from fixtures.mock_cdm import PARENT_CANCER_ID, CONCEPT_META_ID  # type: ignore


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


def test_find_standard_paths_checks_all_targets_in_one_pass(
    mock_cdm_kg: KnowledgeGraph,
) -> None:
    # Concept 196653 is itself a standard concept, so the very first item popped
    # from the BFS queue (the candidate itself) triggers the ancestry check —
    # exercising the multi-target batching directly without needing graph edges.
    candidate = CandidateHit(
        concept_id=196653,
        match_kind=LabelMatchKind.EXACT,
        matched_concept_label="Malignant tumor of kidney",
        synonym=False,
    )

    found = find_standard_paths(
        kg=mock_cdm_kg,
        targets=(PARENT_CANCER_ID, CONCEPT_META_ID),
        candidate=candidate,
        max_depth=6,
    )

    # Only the real ancestor relationship (PARENT_CANCER_ID) should produce a
    # result; CONCEPT_META_ID has no concept_ancestor row for 196653.
    assert len(found) == 1
    assert found[0].concept_id == 196653
    assert found[0].separation == 2


def test_find_standard_paths_max_concepts_caps_per_target(
    mock_cdm_kg: KnowledgeGraph,
) -> None:
    candidate = CandidateHit(
        concept_id=196653,
        match_kind=LabelMatchKind.EXACT,
        matched_concept_label="Malignant tumor of kidney",
        synonym=False,
    )

    found = find_standard_paths(
        kg=mock_cdm_kg,
        targets=(PARENT_CANCER_ID,),
        candidate=candidate,
        max_depth=6,
        max_concepts=1,
    )

    assert len(found) == 1
