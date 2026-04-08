from __future__ import annotations

import pytest

from omop_graph.extensions.omop_alchemy import ClassIDEnum
from omop_graph.graph.constraints import SearchConstraintConcept
from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.reasoning.grounding import GroundingConstraints, ground_term
from omop_graph.reasoning.resolvers.resolver_pipeline import ResolverPipeline
from omop_graph.reasoning.resolvers.resolvers import (
    ExactLabelResolver,
    ExactSynonymResolver,
    PartialLabelResolver,
    PartialSynonymResolver,
)
from fixtures.mock_cdm import PARENT_CANCER_ID  # type: ignore


def _constraints() -> GroundingConstraints:
    return GroundingConstraints(
        parent_ids=(PARENT_CANCER_ID,),
        search_constraint=SearchConstraintConcept(
            domains=("Condition",),
            vocabs=("SNOMED",),
            require_standard=False,
        ),
        max_depth=6,
        predicate_kinds=frozenset({ClassIDEnum.IDENTITY}),
    )


@pytest.mark.parametrize(
    "input_text,expected_concept_id",
    [
        pytest.param("Hodgkin's disease (clinical)", 4038835, id="exact-hodgkin"),
        pytest.param("Malignant neoplasm of ovary", 4181351, id="exact-ovary"),
        pytest.param("Acute myeloid leukaemia", 140352, id="synonym-aml"),
        pytest.param("Myelodysplasia", 138994, id="synonym-mds"),
    ],
)
def test_grounding_resolves_expected_standard_concepts(
    mock_cdm_kg: KnowledgeGraph,
    input_text: str,
    expected_concept_id: int,
) -> None:
    pipeline = ResolverPipeline(
        resolvers=(
            ExactLabelResolver(),
            ExactSynonymResolver(),
            PartialLabelResolver(),
            PartialSynonymResolver(),
        )
    )

    ranked = ground_term(
        resolver_pipeline=pipeline,
        kg=mock_cdm_kg,
        text=input_text,
        text_embedding=None,
        text_embedding_model=None,
        embedding_client=None,
        constraints=_constraints(),
        max_candidates=1,
        metric_type=None,
        index_type=None,
    )

    assert ranked, f"Expected at least one grounding for: {input_text}"
    assert ranked[0].concept_id == expected_concept_id


def test_grounding_maps_non_standard_candidate_via_relationships(
    mock_cdm_kg: KnowledgeGraph,
) -> None:
    pipeline = ResolverPipeline(resolvers=(ExactLabelResolver(),))

    ranked = ground_term(
        resolver_pipeline=pipeline,
        kg=mock_cdm_kg,
        text="Kidney carcinoma term",
        text_embedding=None,
        text_embedding_model=None,
        embedding_client=None,
        constraints=_constraints(),
        max_candidates=1,
        metric_type=None,
        index_type=None,
    )

    assert ranked, "Expected non-standard concept to map to a valid standard concept"
    assert ranked[0].concept_id == 196653


def test_grounding_rejects_concepts_outside_anchored_hierarchy(
    mock_cdm_kg: KnowledgeGraph,
) -> None:
    pipeline = ResolverPipeline(resolvers=(ExactLabelResolver(),))

    ranked = ground_term(
        resolver_pipeline=pipeline,
        kg=mock_cdm_kg,
        text="Meta concept",
        text_embedding=None,
        text_embedding_model=None,
        embedding_client=None,
        constraints=_constraints(),
        max_candidates=1,
        metric_type=None,
        index_type=None,
    )

    assert ranked == []
