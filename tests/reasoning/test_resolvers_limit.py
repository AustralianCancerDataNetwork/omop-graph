"""The resolvers read a concept filter as a scope and take the number of candidates as a limit."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fixtures.search_cdm import CANCER, CONTAINING, search_cdm
from omop_alchemy.cdm.query import ConceptFilter

from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.reasoning.resolvers.resolver_pipeline import ResolverPipeline
from omop_graph.reasoning.resolvers.resolvers import (
    ExactLabelResolver,
    ExactSynonymResolver,
    FullTextResolver,
    FullTextSynonymResolver,
    PartialLabelResolver,
    PartialSynonymResolver,
)

LEXICAL = [
    ExactLabelResolver,
    ExactSynonymResolver,
    PartialLabelResolver,
    PartialSynonymResolver,
    FullTextResolver,
    FullTextSynonymResolver,
]


@pytest.fixture(scope="module")
def kg() -> Iterator[KnowledgeGraph]:
    with search_cdm() as sessions:
        yield KnowledgeGraph(sessions)


@pytest.mark.parametrize("resolver", LEXICAL)
def test_a_resolver_refuses_a_filter_that_sets_a_limit(kg, resolver) -> None:
    if resolver in (FullTextResolver, FullTextSynonymResolver):
        pytest.skip("Full-text search needs PostgreSQL and returns nothing here.")
    with pytest.raises(ValueError, match="Pass limit to the method"):
        resolver().resolve(kg, "cancer", concept_filter=ConceptFilter(limit=1))


@pytest.mark.parametrize("resolver", [ExactLabelResolver, PartialLabelResolver])
def test_a_resolver_returns_at_most_its_limit(kg, resolver) -> None:
    unlimited = list(resolver().resolve(kg, "cancer"))

    assert len(unlimited) > 1
    assert len(list(resolver().resolve(kg, "cancer", limit=1))) == 1


@pytest.mark.parametrize("resolver", [ExactSynonymResolver, PartialSynonymResolver])
def test_a_synonym_resolver_returns_at_most_its_limit(kg, resolver) -> None:
    hits = list(resolver().resolve(kg, "cancer", limit=1))

    assert [hit.concept_id for hit in hits] == [CANCER]


def test_a_resolver_applies_the_scope_and_the_limit_together(kg) -> None:
    scope = ConceptFilter(concept_ids=(CONTAINING,))

    hits = list(PartialLabelResolver().resolve(kg, "cancer", concept_filter=scope, limit=5))

    assert [hit.concept_id for hit in hits] == [CONTAINING]


def test_the_pipeline_gives_each_resolver_the_limit(kg) -> None:
    pipeline = ResolverPipeline(resolvers=(ExactLabelResolver(), PartialLabelResolver()))

    assert [hit.concept_id for hit in pipeline.resolve(kg, "cancer", limit=1)] == [CANCER]
    assert len(list(pipeline.resolve(kg, "cancer"))) > 1
