"""Summary statistics and gap filling, on the small CDMs."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fixtures.helpers import MAPS_TO, curie
from fixtures.hierarchy_cdm import CANCER, DISEASE, LUNG_CANCER, SMOKING_RELATED, hierarchy_cdm
from fixtures.mappings_cdm import mappings_cdm
from oaklib.datamodels.vocabulary import IS_A, OWL_OBJECT_PROPERTY
from omop_alchemy.cdm.query import ConceptFilter

from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.oaklib_interface.codec import OMOPMetadataKey
from omop_graph.oaklib_interface.omop_implementation import OMOPAlchemyImplementation


@pytest.fixture(scope="module")
def adapter() -> Iterator[OMOPAlchemyImplementation]:
    with mappings_cdm() as sessions:
        yield OMOPAlchemyImplementation(kg=KnowledgeGraph(sessions))


@pytest.fixture(scope="module")
def hierarchy() -> Iterator[OMOPAlchemyImplementation]:
    with hierarchy_cdm() as sessions:
        yield OMOPAlchemyImplementation(kg=KnowledgeGraph(sessions))


def test_the_global_statistics_count_the_concepts_edges_and_synonyms(adapter) -> None:
    report = adapter.global_summary_statistics()
    statistics = report.partitions["AllBranches"]

    assert statistics.deprecated_class_count == 1
    assert statistics.non_deprecated_class_count == statistics.class_count - 1
    assert statistics.edge_count_by_predicate[MAPS_TO].filtered_count == 4
    assert statistics.edge_count == 14
    assert statistics.class_count_with_text_definitions == 0
    assert statistics.synonym_statement_count == 0


def test_a_group_by_vocabulary_has_a_partition_for_each_vocabulary(adapter) -> None:
    report = adapter.global_summary_statistics(group_by=OMOPMetadataKey.VOCABULARY)
    whole = adapter.global_summary_statistics().partitions["AllBranches"]

    assert set(report.partitions) == {"SNOMED", "RxNorm"}
    assert sum(p.class_count for p in report.partitions.values()) == whole.class_count


def test_a_group_by_domain_counts_the_concepts_of_the_domain(adapter) -> None:
    report = adapter.global_summary_statistics(group_by=OMOPMetadataKey.DOMAIN)

    assert set(report.partitions) == {"Condition"}


def test_a_vocabulary_has_only_the_edges_of_its_concepts(adapter) -> None:
    statistics = adapter.branch_summary_statistics(
        property_values={OMOPMetadataKey.VOCABULARY: "ICD10CM"}
    )

    assert statistics.edge_count_by_predicate[MAPS_TO].filtered_count == 1


def test_a_group_has_the_counts_of_its_scope(adapter) -> None:
    grouped = adapter.branch_summary_statistics(
        property_values={OMOPMetadataKey.VOCABULARY: "SNOMED"}
    )
    narrowed = adapter.kg.relationship_counts(ConceptFilter(vocabularies=("SNOMED",)))

    assert grouped.edge_count == sum(
        count
        for relationship_id, count in narrowed.items()
        if relationship_id in adapter.kg.default_relationship_ids
    )


def test_the_prefixes_narrow_the_vocabularies(adapter) -> None:
    narrowed = adapter.branch_summary_statistics(prefixes=["ICD10CM"])

    assert narrowed.edge_count_by_predicate[MAPS_TO].filtered_count == 1


def test_what_cannot_be_counted_is_refused(adapter) -> None:
    with pytest.raises(ValueError, match="roots"):
        adapter.branch_summary_statistics(branch_roots=[curie(1)])
    with pytest.raises(ValueError, match="entailed"):
        adapter.branch_summary_statistics(include_entailed=True)
    with pytest.raises(ValueError, match="group"):
        adapter.branch_summary_statistics(property_values={"omop:concept_code": "C1"})


def test_the_properties_that_are_not_groups_have_no_values(adapter) -> None:
    assert adapter.metadata_property_summary_statistics("omop:concept_code") == {}
    assert adapter.metadata_property_summary_statistics(OMOPMetadataKey.DOMAIN) == {
        "Condition": sum(
            adapter.metadata_property_summary_statistics(OMOPMetadataKey.VOCABULARY).values()
        )
    }


def test_gap_filling_connects_the_seeds_without_redundant_edges(hierarchy) -> None:
    found = set(
        hierarchy.gap_fill_relationships(
            [curie(LUNG_CANCER), curie(CANCER), curie(SMOKING_RELATED), curie(DISEASE)],
            predicates=[IS_A],
        )
    )

    assert found == {
        (curie(CANCER), IS_A, curie(DISEASE)),
        (curie(SMOKING_RELATED), IS_A, curie(DISEASE)),
        (curie(LUNG_CANCER), IS_A, curie(CANCER)),
        (curie(LUNG_CANCER), IS_A, curie(SMOKING_RELATED)),
    }


def test_a_gap_filled_graph_holds_the_seeds_and_their_edges(hierarchy) -> None:
    graph = hierarchy.extract_gap_filled_graph([curie(LUNG_CANCER), curie(CANCER), curie(DISEASE)])

    assert {node.id for node in graph.nodes} >= {
        curie(LUNG_CANCER),
        curie(CANCER),
        curie(DISEASE),
    }
    assert {(edge.sub, edge.obj) for edge in graph.edges if edge.pred == IS_A} == {
        (curie(LUNG_CANCER), curie(CANCER)),
        (curie(CANCER), curie(DISEASE)),
    }


def test_unknown_seeds_are_left_out(hierarchy) -> None:
    assert list(hierarchy.gap_fill_relationships(["OMOP:abc", curie(CANCER)])) == []


def test_a_domain_does_not_widen_the_domains_of_the_adapter(adapter) -> None:
    scoped = OMOPAlchemyImplementation(
        kg=adapter.kg, concept_filter=ConceptFilter(domains=("Measurement",), require_active=True)
    )

    statistics = scoped.branch_summary_statistics(
        property_values={OMOPMetadataKey.DOMAIN: "Condition"}
    )

    assert statistics.class_count == statistics.deprecated_class_count


@pytest.mark.parametrize("fixture", ["adapter", "hierarchy"])
def test_the_statistics_describe_what_the_adapter_enumerates(request, fixture) -> None:
    adapter = request.getfixturevalue(fixture)
    statistics = adapter.branch_summary_statistics()
    concept_ids = tuple(adapter.kg.entities(ConceptFilter()))

    assert statistics.class_count == len(list(adapter.entities(filter_obsoletes=False)))
    assert statistics.deprecated_class_count == len(list(adapter.obsoletes()))
    assert statistics.object_property_count == len(
        list(adapter.entities(owl_type=OWL_OBJECT_PROPERTY))
    )
    assert statistics.edge_count == len(list(adapter.relationships()))
    assert statistics.synonym_statement_count == len(adapter.kg.synonyms(concept_ids))
