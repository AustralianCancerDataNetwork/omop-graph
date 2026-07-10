from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from omop_graph.extensions.omop_alchemy import PredicateKind
from omop_graph.graph.constraints import SearchConstraintConcept
from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.reasoning.grounding import (
    GroundingConstraints,
    _query_text_with_context,
    ground_term,
)
from omop_graph.reasoning.resolvers.resolver_pipeline import ResolverPipeline
from omop_graph.reasoning.resolvers.resolvers import (
    EmbeddingResolver,
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
        search_constraint=SearchConstraintConcept(
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


class TestQueryTextWithContext:
    """_query_text_with_context: plain concatenation, no context is a no-op."""

    def test_no_context_returns_query_unchanged(self):
        assert _query_text_with_context("EGFR", None) == "EGFR"

    def test_empty_context_returns_query_unchanged(self):
        assert _query_text_with_context("EGFR", "") == "EGFR"

    def test_context_is_appended(self):
        result = _query_text_with_context("EGFR", "genomic mutation panel")
        assert result == "EGFR\n\ngenomic mutation panel"


def test_ground_term_folds_context_into_on_demand_embedding_text(
    mock_cdm_kg: KnowledgeGraph,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """context is concatenated into the text passed to embed_texts when the
    query embedding is computed on demand (query_embedding=None)."""
    fake_writer = Mock()
    fake_writer.embed_texts.return_value = np.zeros((1, 8), dtype=np.float32)
    monkeypatch.setattr(
        "omop_graph.reasoning.grounding.get_embedding_writer_interface",
        lambda kg: fake_writer,
    )
    # Avoid needing a real embedding-backed nearest-neighbour lookup — the
    # resolution outcome is irrelevant to this test, only the embed_texts call.
    monkeypatch.setattr(
        "omop_graph.reasoning.resolvers.resolvers.get_neareast_concepts",
        lambda **kwargs: None,
    )

    pipeline = ResolverPipeline(resolvers=(EmbeddingResolver(),))

    ground_term(
        resolver_pipeline=pipeline,
        kg=mock_cdm_kg,
        query="EGFR",
        query_embedding=None,
        constraints=_unconstrained_constraints(),
        context="genomic mutation panel",
    )

    fake_writer.embed_texts.assert_called_once()
    assert fake_writer.embed_texts.call_args.kwargs["texts"] == (
        "EGFR\n\ngenomic mutation panel",
    )


def test_ground_term_omits_context_separator_when_not_given(
    mock_cdm_kg: KnowledgeGraph,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_writer = Mock()
    fake_writer.embed_texts.return_value = np.zeros((1, 8), dtype=np.float32)
    monkeypatch.setattr(
        "omop_graph.reasoning.grounding.get_embedding_writer_interface",
        lambda kg: fake_writer,
    )
    monkeypatch.setattr(
        "omop_graph.reasoning.resolvers.resolvers.get_neareast_concepts",
        lambda **kwargs: None,
    )

    pipeline = ResolverPipeline(resolvers=(EmbeddingResolver(),))

    ground_term(
        resolver_pipeline=pipeline,
        kg=mock_cdm_kg,
        query="EGFR",
        query_embedding=None,
        constraints=_unconstrained_constraints(),
    )

    assert fake_writer.embed_texts.call_args.kwargs["texts"] == ("EGFR",)


def test_ground_term_context_has_no_effect_when_query_embedding_supplied(
    mock_cdm_kg: KnowledgeGraph,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """context only feeds the on-demand embedding path; a caller-supplied
    query_embedding means no embedding is computed, so embed_texts is never
    called at all."""
    fake_writer = Mock()
    monkeypatch.setattr(
        "omop_graph.reasoning.grounding.get_embedding_writer_interface",
        lambda kg: fake_writer,
    )
    monkeypatch.setattr(
        "omop_graph.reasoning.resolvers.resolvers.get_neareast_concepts",
        lambda **kwargs: None,
    )

    pipeline = ResolverPipeline(resolvers=(EmbeddingResolver(),))

    ground_term(
        resolver_pipeline=pipeline,
        kg=mock_cdm_kg,
        query="EGFR",
        query_embedding=np.zeros((1, 8), dtype=np.float32),
        constraints=_unconstrained_constraints(),
        context="genomic mutation panel",
    )

    fake_writer.embed_texts.assert_not_called()
