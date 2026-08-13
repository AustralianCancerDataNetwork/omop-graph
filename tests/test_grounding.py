from __future__ import annotations

import pytest

from omop_graph.extensions.omop_alchemy import PredicateKind
from omop_alchemy.cdm.query import ConceptFilter
from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.reasoning.grounding import (
    GroundingConstraints,
    _query_text_with_context,
    ground_term,
)
from omop_graph.reasoning.resolvers.resolver_pipeline import ResolverPipeline
from omop_graph.reasoning.resolvers.resolvers import (
    ExactLabelResolver,
    ExactSynonymResolver,
    PartialLabelResolver,
    PartialSynonymResolver,
)
from fixtures.mock_cdm import PARENT_CANCER_ID  # type: ignore


class TestQueryTextWithContext:
    def test_no_context_returns_query_unchanged(self) -> None:
        assert _query_text_with_context("acute MI", None) == "acute MI"

    def test_empty_context_returns_query_unchanged(self) -> None:
        assert _query_text_with_context("acute MI", "") == "acute MI"

    def test_context_appended_with_blank_line_separator(self) -> None:
        result = _query_text_with_context("acute MI", "patient has a history of smoking")
        assert result == "acute MI\n\npatient has a history of smoking"


def _constraints() -> GroundingConstraints:
    return GroundingConstraints(
        parent_ids=(PARENT_CANCER_ID,),
        search_constraint=ConceptFilter(
            domains=("Condition",),
            vocabularies=("SNOMED",),
            require_standard=False,
        ),
        max_depth=6,
        predicate_kinds=frozenset({PredicateKind.IDENTITY}),
    )


def _unconstrained_constraints() -> GroundingConstraints:
    # max_depth is passed explicitly (like _constraints() above) so these tests don't
    # depend on OmopGraphConfig's package config being set up in the environment.
    return GroundingConstraints(
        parent_ids=None,
        search_constraint=ConceptFilter(
            domains=("Condition",),
            vocabularies=("SNOMED",),
            require_standard=False,
        ),
        max_depth=6,
        predicate_kinds=frozenset({PredicateKind.IDENTITY}),
    )


@pytest.mark.parametrize(
    "query,expected_concept_id",
    [
        pytest.param("Hodgkin's disease (clinical)", 4038835, id="exact-hodgkin"),
        pytest.param("Malignant neoplasm of ovary", 4181351, id="exact-ovary"),
        pytest.param("Acute myeloid leukaemia", 140352, id="synonym-aml"),
        pytest.param("Myelodysplasia", 138994, id="synonym-mds"),
    ],
)
def test_grounding_resolves_expected_standard_concepts(
    mock_cdm_kg: KnowledgeGraph,
    query: str,
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
        query=query,
        query_embedding=None,
        constraints=_constraints(),
        max_candidates=1,
    )

    assert ranked, f"Expected at least one grounding for: {query}"
    assert ranked[0].concept_id == expected_concept_id


def test_grounding_maps_non_standard_candidate_via_relationships(
    mock_cdm_kg: KnowledgeGraph,
) -> None:
    pipeline = ResolverPipeline(resolvers=(ExactLabelResolver(),))

    ranked = ground_term(
        resolver_pipeline=pipeline,
        kg=mock_cdm_kg,
        query="Kidney carcinoma term",
        query_embedding=None,
        constraints=_constraints(),
        max_candidates=1,
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
        query="Meta concept",
        query_embedding=None,
        constraints=_constraints(),
        max_candidates=1,
    )

    assert ranked == []


def test_grounding_maps_non_standard_candidate_without_parent_ids(
    mock_cdm_kg: KnowledgeGraph,
) -> None:
    pipeline = ResolverPipeline(resolvers=(ExactLabelResolver(),))

    ranked = ground_term(
        resolver_pipeline=pipeline,
        kg=mock_cdm_kg,
        query="Kidney carcinoma term",
        query_embedding=None,
        constraints=_unconstrained_constraints(),
        max_candidates=1,
    )

    assert ranked, "Expected non-standard candidate to standardize without a parent anchor"
    assert ranked[0].concept_id == 196653
    assert ranked[0].identity_hops == 1


def test_grounding_standard_candidate_without_parent_ids_is_zero_hop(
    mock_cdm_kg: KnowledgeGraph,
) -> None:
    pipeline = ResolverPipeline(resolvers=(ExactLabelResolver(),))

    ranked = ground_term(
        resolver_pipeline=pipeline,
        kg=mock_cdm_kg,
        query="Malignant tumor of kidney",
        query_embedding=None,
        constraints=_unconstrained_constraints(),
        max_candidates=1,
    )

    assert ranked, "Expected already-standard candidate to ground without a parent anchor"
    assert ranked[0].concept_id == 196653
    assert ranked[0].identity_hops == 0
    assert ranked[0].separation == 0
