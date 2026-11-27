from __future__ import annotations

from omop_alchemy.cdm.query import ConceptFilter

from omop_graph.graph.kg import KnowledgeGraph

KIDNEY_CARCINOMA = 196653


def test_entities_follow_the_given_concept_filter(mock_cdm_kg: KnowledgeGraph) -> None:
    assert list(mock_cdm_kg.entities(ConceptFilter(concept_ids=(KIDNEY_CARCINOMA,)))) == [
        KIDNEY_CARCINOMA
    ]
    assert list(mock_cdm_kg.entities(limit=2)) == sorted(mock_cdm_kg.entities())[:2]
    assert list(mock_cdm_kg.entities(ConceptFilter(domains=("No such domain",)))) == []


def test_roots_leaves_and_singletons_follow_the_given_concept_filter(
    mock_cdm_kg: KnowledgeGraph,
) -> None:
    nothing = ConceptFilter(domains=("No such domain",))

    assert list(mock_cdm_kg.roots(nothing)) == []
    assert list(mock_cdm_kg.leaves(nothing)) == []
    assert list(mock_cdm_kg.singletons(nothing)) == []


def test_singletons_are_the_concepts_that_are_both_roots_and_leaves(
    mock_cdm_kg: KnowledgeGraph,
) -> None:
    roots, leaves = set(mock_cdm_kg.roots()), set(mock_cdm_kg.leaves())

    assert roots
    assert set(mock_cdm_kg.singletons()) == roots & leaves
