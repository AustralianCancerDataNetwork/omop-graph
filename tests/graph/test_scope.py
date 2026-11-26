from __future__ import annotations

import inspect
from collections.abc import Iterator

import pytest
from fixtures.filler_cdm import HOLDER, NON_STANDARD_FILLER, STANDARD_PARENT, filler_cdm
from fixtures.hierarchy_cdm import CANCER, DISEASE, LUNG_CANCER, SMOKING_RELATED, hierarchy_cdm
from omop_alchemy.cdm.query import ConceptFilter

from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.graph.queries import DEFAULT_CONCEPT_FILTER
from omop_graph.graph.scope import (
    SCOPE_ONLY_MARKER,
    narrowed_to_domains,
    narrowed_to_vocabularies,
    require_scope,
    restricted_to_ids,
)

QUERIES = {"entities", "roots", "leaves", "singletons", "obsolete_concepts"}
"""The methods that take a concept filter as a query, where ``limit`` truncates the answer."""


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
        require_scope(ConceptFilter(limit=1))


def test_restricting_to_ids_keeps_the_ids_the_scope_already_limits() -> None:
    scope = ConceptFilter(concept_ids=(1, 2, 3))

    assert restricted_to_ids(scope, (3, 4, 1)).concept_ids == (3, 1)
    assert restricted_to_ids(ConceptFilter(), (3, 4, 3)).concept_ids == (3, 4)


def test_narrowing_keeps_only_what_the_scope_allows() -> None:
    scope = ConceptFilter(vocabularies=("SNOMED", "LOINC"), domains=("Condition",))

    assert narrowed_to_vocabularies(scope, ("LOINC", "ICD10")).vocabularies == ("LOINC",)
    assert narrowed_to_domains(scope, ("Drug", "Condition")).domains == ("Condition",)
    assert narrowed_to_domains(ConceptFilter(), ("Drug",)).domains == ("Drug",)


def test_every_method_taking_a_scope_checks_it() -> None:
    unchecked = [
        name
        for name, method in inspect.getmembers(KnowledgeGraph, inspect.isfunction)
        if not name.startswith("_")
        and "concept_filter" in inspect.signature(method).parameters
        and name not in QUERIES
        and not getattr(method, SCOPE_ONLY_MARKER, False)
    ]

    assert unchecked == []


def test_a_scope_with_a_limit_is_refused_at_the_call(kg) -> None:
    with pytest.raises(ValueError, match="limit"):
        kg.closure([LUNG_CANCER], concept_filter=ConceptFilter(limit=1))
    with pytest.raises(ValueError, match="limit"):
        kg.information_content([LUNG_CANCER], concept_filter=ConceptFilter(limit=1))
    with pytest.raises(ValueError, match="limit"):
        kg.entailed_relationships((LUNG_CANCER,), None, concept_filter=ConceptFilter(limit=1))


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
