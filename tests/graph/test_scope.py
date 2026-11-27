from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator

import pytest
import sqlalchemy as sa
from fixtures.filler_cdm import HOLDER, NON_STANDARD_FILLER, STANDARD_PARENT, filler_cdm
from fixtures.hierarchy_cdm import CANCER, DISEASE, LUNG_CANCER, SMOKING_RELATED, hierarchy_cdm
from omop_alchemy.cdm.model.vocabulary import Concept
from omop_alchemy.cdm.query import ConceptFilter

from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.graph.nodes import LabelMatchKind
from omop_graph.graph.queries import DEFAULT_CONCEPT_FILTER
from omop_graph.graph.scope import (
    _require_scope,
    limited,
    narrowed_to_domains,
    narrowed_to_vocabularies,
    restricted_to_ids,
)

SCOPED_CALLS: dict[str, Callable[[KnowledgeGraph, ConceptFilter], object]] = {
    "all_entailed_relationships": lambda kg, scope: kg.all_entailed_relationships(
        concept_filter=scope
    ),
    "closure": lambda kg, scope: kg.closure([LUNG_CANCER], concept_filter=scope),
    "common_ancestors": lambda kg, scope: kg.common_ancestors(
        LUNG_CANCER, CANCER, concept_filter=scope
    ),
    "common_descendants": lambda kg, scope: kg.common_descendants(
        LUNG_CANCER, CANCER, concept_filter=scope
    ),
    "concept_counts": lambda kg, scope: kg.concept_counts(scope),
    "concept_lookup": lambda kg, scope: kg.concept_lookup(
        "cancer", LabelMatchKind.EXACT, concept_filter=scope
    ),
    "entailed_relationships": lambda kg, scope: kg.entailed_relationships(
        (LUNG_CANCER,), None, concept_filter=scope
    ),
    "entities": lambda kg, scope: kg.entities(scope),
    "information_content": lambda kg, scope: kg.information_content(
        [LUNG_CANCER], concept_filter=scope
    ),
    "jaccard_similarity": lambda kg, scope: kg.jaccard_similarity(
        LUNG_CANCER, CANCER, concept_filter=scope
    ),
    "leaves": lambda kg, scope: kg.leaves(scope),
    "most_recent_common_ancestors": lambda kg, scope: kg.most_recent_common_ancestors(
        LUNG_CANCER, CANCER, concept_filter=scope
    ),
    "obsolete_concepts": lambda kg, scope: kg.obsolete_concepts(concept_filter=scope),
    "pairwise_similarity": lambda kg, scope: kg.pairwise_similarity(
        LUNG_CANCER, CANCER, concept_filter=scope
    ),
    "relationship_counts": lambda kg, scope: kg.relationship_counts(scope),
    "roots": lambda kg, scope: kg.roots(scope),
    "scope_statistics": lambda kg, scope: kg.scope_statistics(scope),
    "search": lambda kg, scope: kg.search("cancer", concept_filter=scope),
    "singletons": lambda kg, scope: kg.singletons(scope),
    "synonym_counts": lambda kg, scope: kg.synonym_counts(scope),
}
"""A call of every graph method that takes a scope, which the scope is passed to."""

STORE_ONLY = {"nearest_concepts"}
"""The scoped methods that need an embedding store, covered in the embedding tests."""


@pytest.fixture(scope="module")
def kg() -> Iterator[KnowledgeGraph]:
    with hierarchy_cdm() as sessions:
        yield KnowledgeGraph(sessions)


@pytest.fixture(scope="module")
def filler_kg() -> Iterator[KnowledgeGraph]:
    with filler_cdm() as sessions:
        yield KnowledgeGraph(sessions)


def test_a_limit_cannot_decide_membership() -> None:
    with pytest.raises(ValueError, match="limit"):
        _require_scope(ConceptFilter(limit=1))


def test_restricting_to_ids_keeps_the_ids_the_scope_already_limits() -> None:
    scope = ConceptFilter(concept_ids=(1, 2, 3))

    assert restricted_to_ids(scope, (3, 4, 1)).concept_ids == (3, 1)
    assert restricted_to_ids(ConceptFilter(), (3, 4, 3)).concept_ids == (3, 4)


def test_narrowing_keeps_only_what_the_scope_allows() -> None:
    scope = ConceptFilter(vocabularies=("SNOMED", "LOINC"), domains=("Condition",))

    assert narrowed_to_vocabularies(scope, ("LOINC", "ICD10")).vocabularies == ("LOINC",)
    assert narrowed_to_domains(scope, ("Drug", "Condition")).domains == ("Condition",)
    assert narrowed_to_domains(ConceptFilter(), ("Drug",)).domains == ("Drug",)


def test_every_method_taking_a_scope_is_covered() -> None:
    scoped = {
        name
        for name, method in inspect.getmembers(KnowledgeGraph, inspect.isfunction)
        if not name.startswith("_") and "concept_filter" in inspect.signature(method).parameters
    }

    assert scoped == set(SCOPED_CALLS) | STORE_ONLY


@pytest.mark.parametrize("name", sorted(SCOPED_CALLS))
def test_a_scope_with_a_limit_is_refused_by_every_method(kg, name) -> None:
    with pytest.raises(ValueError, match="Pass limit to the method"):
        result = SCOPED_CALLS[name](kg, ConceptFilter(limit=1))
        if isinstance(result, Iterator):
            next(result, None)


def test_a_limit_keeps_the_lowest_concept_ids(kg) -> None:
    everything = list(kg.entities())

    assert list(kg.entities(limit=2)) == sorted(everything)[:2]
    assert list(kg.roots(limit=1)) == sorted(kg.roots())[:1]


def test_limited_orders_by_concept_id_and_leaves_an_unlimited_statement_alone() -> None:
    stmt = sa.select(Concept.concept_id)

    assert limited(stmt, None) is stmt
    assert "ORDER BY" in str(limited(stmt, 3)) and "LIMIT" in str(limited(stmt, 3))


def test_the_ids_of_a_scope_limit_what_a_walk_reports(kg) -> None:
    reached = kg.closure([LUNG_CANCER], concept_filter=ConceptFilter(concept_ids=(CANCER,)))

    assert reached[LUNG_CANCER] == {CANCER}


def test_a_scope_with_ids_does_not_widen_the_identifiers_of_a_search(kg) -> None:
    named = (SMOKING_RELATED, DISEASE)
    scope = ConceptFilter(concept_ids=(DISEASE,))

    assert list(kg.search("none", identifiers=named, concept_filter=scope)) == [DISEASE]


def test_entailment_reports_the_same_triple_from_either_end(filler_kg) -> None:
    from_holder = set(
        filler_kg.entailed_relationships((HOLDER,), None, concept_filter=DEFAULT_CONCEPT_FILTER)
    )
    into_parent = set(
        filler_kg.entailed_relationships(
            None, (STANDARD_PARENT,), concept_filter=DEFAULT_CONCEPT_FILTER
        )
    )

    wanted = (HOLDER, "Has finding site", STANDARD_PARENT)
    assert wanted in from_holder
    assert wanted in into_parent
    assert (HOLDER, "Has finding site", NON_STANDARD_FILLER) in from_holder
