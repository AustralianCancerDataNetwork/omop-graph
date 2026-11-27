from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import patch

import pytest
from fixtures.cdm_builders import (
    classified,
    concept,
    edge,
    relationship,
    relationship_class,
    sqlite_cdm,
    synonym,
)
from fixtures.hierarchy_cdm import (
    CANCER,
    DISEASE,
    LUNG_CANCER,
    SMOKING_RELATED,
    hierarchy_cdm,
)
from omop_alchemy.cdm.query import ConceptFilter

from omop_graph.extensions.relationship_classification import (
    HierarchyDirection,
    PredicateKind,
)
from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.graph.model.counts import Grouping, SynonymCounts
from omop_graph.graph.model.edges import EdgeFilter
from omop_graph.graph.model.language import OMOPLanguage
from omop_graph.graph.model.scope import ALL_CONCEPTS


@pytest.fixture(scope="module")
def kg() -> Iterator[KnowledgeGraph]:
    with hierarchy_cdm() as sessions:
        yield KnowledgeGraph(sessions)


def test_concepts_are_counted_by_vocabulary_domain_and_validity(kg) -> None:
    counts = kg.concept_counts()

    assert counts.total == 16
    assert counts.deprecated == 1
    assert counts.by_vocabulary() == {"SNOMED": 16}
    assert counts.by_domain() == {"Condition": 16}


def test_a_scope_narrows_the_concept_counts(kg) -> None:
    assert kg.concept_counts(ConceptFilter(require_standard=True)).total == 11
    assert kg.concept_counts(ConceptFilter(require_active=True)).deprecated == 0
    assert kg.concept_counts(ConceptFilter(vocabularies=("LOINC",))).total == 0


def test_active_relationships_are_counted_by_id(kg) -> None:
    assert kg.relationship_counts() == {
        "Is a": 9,
        "Maps to": 4,
        "Mapped from": 1,
        "Subsumes": 2,
        "Has finding site": 1,
    }


def test_the_relationships_of_a_scope_are_those_of_its_subjects(kg) -> None:
    standard = ConceptFilter(require_standard=True)

    assert kg.relationship_counts(standard) == {
        "Is a": 8,
        "Mapped from": 1,
        "Subsumes": 2,
        "Has finding site": 1,
    }
    assert kg.relationship_counts(ALL_CONCEPTS) == kg.relationship_counts()


def test_synonyms_are_counted_with_their_distinct_names(kg) -> None:
    counts = kg.synonym_counts()

    assert (counts.statements, counts.distinct) == (2, 2)
    assert kg.synonym_counts(ConceptFilter(vocabularies=("LOINC",))).statements == 0


def test_gap_filling_drops_the_edge_another_concept_lies_between(kg) -> None:
    seeds = (LUNG_CANCER, CANCER, SMOKING_RELATED, DISEASE)

    assert kg.gap_filled_relationships(seeds, EdgeFilter(predicate_ids=frozenset({"Is a"}))) == (
        (CANCER, "Is a", DISEASE),
        (SMOKING_RELATED, "Is a", DISEASE),
        (LUNG_CANCER, "Is a", CANCER),
        (LUNG_CANCER, "Is a", SMOKING_RELATED),
    )


def test_gap_filling_keeps_every_relationship_by_default(kg) -> None:
    triples = kg.gap_filled_relationships((LUNG_CANCER, CANCER, DISEASE))

    assert (DISEASE, "Subsumes", CANCER) in triples
    assert (LUNG_CANCER, "Is a", DISEASE) not in triples


def test_gap_filling_a_single_concept_finds_nothing(kg) -> None:
    assert kg.gap_filled_relationships((LUNG_CANCER,)) == ()
    assert kg.gap_filled_relationships(()) == ()


def test_the_relationships_of_every_group_are_counted_in_one_pass(kg) -> None:
    by_vocabulary = kg.relationship_counts_by(Grouping.VOCABULARY)

    assert by_vocabulary == {"SNOMED": kg.relationship_counts()}
    assert kg.relationship_counts_by(Grouping.DOMAIN) == {"Condition": kg.relationship_counts()}


def test_the_synonyms_of_every_group_are_counted_in_one_pass(kg) -> None:
    assert kg.synonym_counts_by(Grouping.VOCABULARY) == {"SNOMED": kg.synonym_counts()}


@pytest.mark.parametrize("grouping", list(Grouping))
def test_the_statistics_of_each_group_are_those_counted_for_the_group_alone(kg, grouping) -> None:
    grouped = kg.group_statistics(grouping)

    assert grouped
    for group, statistics in grouped.items():
        alone = (
            ConceptFilter(vocabularies=(group,))
            if grouping is Grouping.VOCABULARY
            else ConceptFilter(domains=(group,))
        )
        assert statistics.relationships == kg.relationship_counts(alone)
        assert statistics.synonyms == kg.synonym_counts(alone)


def test_the_statistics_of_every_group_are_counted_once(kg) -> None:
    assert kg.group_statistics(Grouping.VOCABULARY) is kg.group_statistics(Grouping.VOCABULARY)


def test_a_scope_is_counted_whatever_the_standardness_of_its_concepts(kg) -> None:
    scope = ConceptFilter(vocabularies=("SNOMED",), require_standard=True, require_active=True)
    counted = kg.scope_statistics(scope)

    assert counted.relationships == kg.relationship_counts(ConceptFilter(vocabularies=("SNOMED",)))
    assert counted.synonyms == kg.synonym_counts(ConceptFilter(vocabularies=("SNOMED",)))


def test_a_single_group_is_read_from_the_counts_of_every_group_when_asked(kg) -> None:
    scope = ConceptFilter(domains=("Condition",))
    alone = kg.scope_statistics(scope)

    with patch.object(kg, "relationship_counts") as counted:
        grouped = kg.scope_statistics(scope, by_group=True)

    counted.assert_not_called()
    assert grouped == alone


def test_a_scope_that_is_not_a_single_group_is_counted_on_its_own_even_when_grouped(kg) -> None:
    scope = ConceptFilter(vocabularies=("SNOMED",), domains=("Condition",))

    assert kg.scope_statistics(scope, by_group=True) == kg.scope_statistics(scope)
    assert kg.scope_statistics(ConceptFilter(vocabularies=("Nope",)), by_group=True) == (
        kg.scope_statistics(ConceptFilter(vocabularies=("Nope",)))
    )


@pytest.fixture(scope="module")
def grouped_kg() -> Iterator[KnowledgeGraph]:
    """
    Concepts 1 to 4 in two vocabularies and two domains, each pair of groups of a different size.

    SNOMED holds 1 and 2 and LOINC holds 3 and 4. Condition holds 1 and 3 and Drug holds 2
    and 4. The relationships start from the concept that counts them, so concept 1 has one,
    2 has one, 3 has two and 4 has one. Concepts 1 and 2 share the synonym "first", and
    concept 4 repeats "shared" of concept 1.
    """
    drug = concept(2)
    drug.domain_id = "Drug"
    loinc_drug = concept(4, vocabulary_id="LOINC")
    loinc_drug.domain_id = "Drug"
    rows = [
        concept(1),
        drug,
        concept(3, vocabulary_id="LOINC"),
        loinc_drug,
        relationship("Is a", "Subsumes"),
        relationship_class(PredicateKind.HIERARCHY, "Taxonomic - up"),
        classified(
            "Is a",
            PredicateKind.HIERARCHY,
            "Taxonomic - up",
            generalises=True,
            direction=HierarchyDirection.UP,
        ),
        edge(1, "Is a", 2),
        edge(2, "Is a", 1),
        edge(3, "Is a", 1),
        edge(3, "Is a", 2),
        edge(4, "Is a", 1),
        synonym(1, "shared", OMOPLanguage.ENGLISH),
        synonym(1, "first", OMOPLanguage.ENGLISH),
        synonym(2, "first", OMOPLanguage.ENGLISH),
        synonym(2, "second", OMOPLanguage.ENGLISH),
        synonym(4, "shared", OMOPLanguage.ENGLISH),
    ]
    with sqlite_cdm(rows) as sessions:
        yield KnowledgeGraph(sessions)


def test_each_vocabulary_and_each_domain_has_its_own_counts(grouped_kg) -> None:
    by_vocabulary = grouped_kg.group_statistics(Grouping.VOCABULARY)
    by_domain = grouped_kg.group_statistics(Grouping.DOMAIN)

    assert {name: g.relationships for name, g in by_vocabulary.items()} == {
        "SNOMED": {"Is a": 2},
        "LOINC": {"Is a": 3},
    }
    assert {name: g.relationships for name, g in by_domain.items()} == {
        "Condition": {"Is a": 3},
        "Drug": {"Is a": 2},
    }
    assert by_vocabulary["SNOMED"].synonyms == SynonymCounts(4, 3)
    assert by_vocabulary["LOINC"].synonyms == SynonymCounts(1, 1)
    assert by_domain["Condition"].synonyms == SynonymCounts(2, 2)
    assert by_domain["Drug"].synonyms == SynonymCounts(3, 3)
