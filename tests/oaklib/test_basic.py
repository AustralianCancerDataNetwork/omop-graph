"""Behaviour of the basic ontology interface on a small CDM with every OMOP case it maps."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fixtures.cdm_builders import (
    ancestor,
    classified,
    concept,
    edge,
    relationship,
    relationship_class,
    sqlite_cdm,
    synonym,
)
from oaklib.datamodels.vocabulary import (
    DEPRECATED_PREDICATE,
    INVERSE_OF,
    IS_A,
    LABEL_PREDICATE,
    OWL_CLASS,
    OWL_OBJECT_PROPERTY,
    TERM_REPLACED_BY,
)
from omop_alchemy.cdm.model.vocabulary.concept import Concept
from omop_alchemy.cdm.model.vocabulary.concept_ancestor import Concept_Ancestor
from omop_alchemy.cdm.model.vocabulary.concept_relationship import Concept_Relationship
from omop_alchemy.cdm.model.vocabulary.concept_synonym import Concept_Synonym
from omop_alchemy.cdm.model.vocabulary.domain import Domain
from omop_alchemy.cdm.model.vocabulary.relationship import Relationship
from omop_alchemy.cdm.model.vocabulary.vocabulary import Vocabulary
from omop_alchemy.cdm.query import ConceptFilter

from omop_graph.extensions.omop_alchemy import (
    HierarchyDirection,
    PredicateKind,
    RelationshipClass,
    RelationshipMapping,
)
from omop_graph.graph.edges import EdgeFilter
from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.oaklib_interface.codec import OMOPLanguage
from omop_graph.oaklib_interface.omop_implementation import OMOPAlchemyImplementation

DISEASE, CANCER, DELETED, OUTDATED, ICD_CODE, LONELY = 100, 101, 102, 103, 104, 105
SNOMED_ONTOLOGY, ICD_ONTOLOGY, CONDITION_DOMAIN = 44819097, 44819098, 3
SUBSUMES = "OMOP_REL:Subsumes"


@pytest.fixture(scope="module")
def adapter() -> Iterator[OMOPAlchemyImplementation]:
    rows = [
        Domain(domain_id="Condition", domain_name="Condition", domain_concept_id=3),
        Vocabulary(
            vocabulary_id="SNOMED",
            vocabulary_name="SNOMED CT",
            vocabulary_reference="http://snomed.example",
            vocabulary_version="2024",
            vocabulary_concept_id=SNOMED_ONTOLOGY,
        ),
        Vocabulary(
            vocabulary_id="ICD10",
            vocabulary_name="ICD-10",
            vocabulary_concept_id=ICD_ONTOLOGY,
        ),
        concept(DISEASE, "Disease"),
        concept(CANCER, "Cancer"),
        concept(DELETED, "Removed disease", invalid_reason="D"),
        concept(OUTDATED, "Outdated disease", invalid_reason="U"),
        concept(ICD_CODE, "Code", vocabulary_id="ICD10", standard_concept=None),
        concept(LONELY, "Lonely disease"),
        relationship("Is a", "Subsumes"),
        relationship("Subsumes", "Is a"),
        relationship("Concept replaced by", "Concept replaces"),
        relationship("Concept replaces", "Concept replaced by"),
        relationship("Unclassified", "Unclassified"),
        relationship_class(PredicateKind.HIERARCHY, "Taxonomic - up", HierarchyDirection.UP),
        relationship_class(PredicateKind.HIERARCHY, "Taxonomic - down", HierarchyDirection.DOWN),
        relationship_class(PredicateKind.IDENTITY, "Version lineage"),
        classified("Is a", PredicateKind.HIERARCHY, "Taxonomic - up"),
        classified("Subsumes", PredicateKind.HIERARCHY, "Taxonomic - down"),
        classified("Concept replaced by", PredicateKind.IDENTITY, "Version lineage"),
        classified("Concept replaces", PredicateKind.IDENTITY, "Version lineage"),
        edge(CANCER, "Is a", DISEASE),
        edge(DISEASE, "Subsumes", CANCER),
        edge(OUTDATED, "Concept replaced by", CANCER),
        edge(CANCER, "Concept replaces", OUTDATED),
        ancestor(DISEASE, DISEASE, 0),
        ancestor(CANCER, CANCER, 0),
        ancestor(LONELY, LONELY, 0),
        ancestor(DISEASE, CANCER, 1),
        synonym(CANCER, "Neoplasm", OMOPLanguage.ENGLISH),
        synonym(CANCER, "Cáncer", OMOPLanguage.SPANISH),
        synonym(CANCER, "Cancer fr", OMOPLanguage.FRENCH),
    ]
    models = (
        Concept,
        Domain,
        Vocabulary,
        Relationship,
        Concept_Ancestor,
        Concept_Relationship,
        Concept_Synonym,
        RelationshipClass,
        RelationshipMapping,
    )
    with sqlite_cdm(models, rows) as sessions:
        yield OMOPAlchemyImplementation(kg=KnowledgeGraph(sessions))


def test_each_vocabulary_is_an_ontology_with_its_metadata(adapter) -> None:
    snomed = f"OMOP:{SNOMED_ONTOLOGY}"

    assert set(adapter.ontologies()) == {snomed, f"OMOP:{ICD_ONTOLOGY}"}
    assert adapter.ontology_metadata_map(snomed)[LABEL_PREDICATE] == ["SNOMED CT"]
    assert list(adapter.ontology_versions(snomed)) == ["2024"]
    assert list(adapter.ontology_versions(f"OMOP:{ICD_ONTOLOGY}")) == []
    assert adapter.ontology_metadata_map("OMOP:1") == {}


def test_an_entity_is_defined_by_the_ontology_of_its_vocabulary(adapter) -> None:
    defined = dict(adapter.defined_bys(["OMOP:100", "OMOP:104", "OMOP_REL:Subsumes", "OMOP:999"]))

    assert defined == {
        "OMOP:100": f"OMOP:{SNOMED_ONTOLOGY}",
        "OMOP:104": f"OMOP:{ICD_ONTOLOGY}",
        "OMOP_REL:Subsumes": None,
        "OMOP:999": None,
    }


@pytest.mark.parametrize(
    ("include_merged", "expected"),
    [(True, {"OMOP:102", "OMOP:103"}), (False, {"OMOP:102"})],
)
def test_obsoletes_include_updated_concepts_only_when_merged_are_included(
    adapter, include_merged, expected
) -> None:
    assert set(adapter.obsoletes(include_merged=include_merged)) == expected


def test_every_obsolete_is_an_entity_when_obsoletes_are_not_filtered(adapter) -> None:
    everything = set(adapter.entities(filter_obsoletes=False))

    assert set(adapter.obsoletes()) <= everything
    assert {f"OMOP:{ICD_CODE}", f"OMOP:{DELETED}"} <= everything
    assert {f"OMOP:{ICD_CODE}", f"OMOP:{DELETED}"}.isdisjoint(adapter.entities())


def test_obsoletes_stay_within_the_adapters_vocabularies(adapter) -> None:
    restricted = OMOPAlchemyImplementation(
        kg=adapter.kg, concept_filter=ConceptFilter(vocabularies=("ICD10",))
    )

    assert set(restricted.obsoletes()) == set()
    assert set(restricted.entities(filter_obsoletes=False)) == {f"OMOP:{ICD_CODE}"}


def test_an_updated_concept_is_deprecated_and_names_its_replacement(adapter) -> None:
    outdated = adapter.entity_metadata_map("OMOP:103")

    assert outdated[DEPRECATED_PREDICATE] == ["true"]
    assert outdated[TERM_REPLACED_BY] == ["OMOP:101"]
    assert list(adapter.obsoletes_migration_relationships(["OMOP:103"])) == [
        ("OMOP:103", TERM_REPLACED_BY, "OMOP:101")
    ]
    assert DEPRECATED_PREDICATE not in adapter.entity_metadata_map("OMOP:101")


def test_a_relationship_has_its_name_and_inverse_as_metadata(adapter) -> None:
    assert adapter.entity_metadata_map(IS_A) == {
        "id": [IS_A],
        LABEL_PREDICATE: ["Is a"],
        INVERSE_OF: [SUBSUMES],
    }


def test_entity_metadata_is_empty_for_an_unknown_entity(adapter) -> None:
    assert adapter.entity_metadata_map("OMOP:999") == {}
    assert adapter.entity_alias_map("OMOP:999") == {}


def test_owl_types_tell_concepts_from_relationships_and_skip_unknown_entities(adapter) -> None:
    types = dict(adapter.owl_types(["OMOP:100", IS_A, "OMOP:999"]))

    assert types == {"OMOP:100": OWL_CLASS, IS_A: OWL_OBJECT_PROPERTY}
    assert list(adapter.dangling(["OMOP:100", "OMOP:999", "OMOP_REL:Nope"])) == [
        "OMOP:999",
        "OMOP_REL:Nope",
    ]
    assert list(adapter.dangling()) == []


def test_the_category_of_a_concept_is_its_domain_concept(adapter) -> None:
    assert list(adapter.terms_categories(["OMOP:100", "OMOP:999"])) == [
        ("OMOP:100", f"OMOP:{CONDITION_DOMAIN}")
    ]


def test_labels_are_looked_up_in_one_batch_for_concepts_and_relationships(adapter) -> None:
    curies = ["OMOP:100", IS_A, "OMOP:999"]

    assert dict(adapter.labels(curies)) == {"OMOP:100": "Disease", IS_A: "Is a", "OMOP:999": None}
    assert dict(adapter.labels(curies, allow_none=False)) == {"OMOP:100": "Disease", IS_A: "Is a"}
    assert adapter.label("OMOP:999") is None


def test_a_label_in_another_language_is_the_first_synonym_in_that_language(adapter) -> None:
    assert adapter.label("OMOP:101", lang="es") == "Cáncer"
    assert adapter.label("OMOP:101", lang="en") == "Cancer"
    assert adapter.label("OMOP:100", lang="es") is None
    assert adapter.label("OMOP:101", lang="xx") is None


def test_the_languages_are_those_synonyms_are_written_in(adapter) -> None:
    assert set(adapter.languages()) == {"en", "es", "fr"}
    assert adapter.multilingual is True


def test_multilingual_labels_tag_translations_and_leave_the_default_untagged(adapter) -> None:
    assert set(adapter.multilingual_labels(["OMOP:101"])) == {
        ("OMOP:101", "Cancer", None),
        ("OMOP:101", "Cáncer", "es"),
        ("OMOP:101", "Cancer fr", "fr"),
    }
    assert set(adapter.multilingual_labels(["OMOP:101"], langs=["es"])) == {
        ("OMOP:101", "Cáncer", "es")
    }
    assert list(adapter.multilingual_labels(["OMOP:999"])) == [("OMOP:999", None, None)]


def test_omop_has_no_definitions_comments_or_subsets(adapter) -> None:
    assert adapter.definition("OMOP:100") is None
    assert list(adapter.definitions(["OMOP:100"])) == []
    assert list(adapter.comments(["OMOP:100"])) == [("OMOP:100", None)]
    assert list(adapter.comments(["OMOP:100"], allow_none=False)) == []
    assert list(adapter.subsets()) == []
    assert list(adapter.subset_members("OMOP:100")) == []


def test_hierarchical_parents_come_from_is_a_edges_or_the_closure(adapter) -> None:
    assert adapter.hierarchical_parents("OMOP:101", isa_only=True) == ["OMOP:100"]
    assert adapter.hierarchical_parents("OMOP:101") == ["OMOP:100"]
    assert adapter.hierarchical_parents("OMOP:100", isa_only=True) == []


def test_roots_leafs_and_singletons_use_the_closure_without_predicates(adapter) -> None:
    assert set(adapter.roots()) == {"OMOP:100", "OMOP:105"}
    assert set(adapter.leafs()) == {"OMOP:101", "OMOP:105"}
    assert set(adapter.singletons()) == {"OMOP:105"}


def test_roots_leafs_and_singletons_follow_the_named_predicates(adapter) -> None:
    assert set(adapter.roots(predicates=[IS_A])) == {"OMOP:100", "OMOP:105"}
    assert set(adapter.leafs(predicates=[IS_A])) == {"OMOP:101", "OMOP:105"}
    assert set(adapter.singletons(predicates=[IS_A])) == {"OMOP:105"}


def test_roots_can_require_edges_within_a_domain(adapter) -> None:
    within_domain = adapter.kg.roots(
        edge_filter=EdgeFilter(predicate_ids=frozenset({"Is a"}), within_domain=True)
    )

    assert set(within_domain) == {100, 105}


def test_a_predicate_that_selects_no_edge_leaves_every_concept_a_root(adapter) -> None:
    assert set(adapter.roots(predicates=["OMOP_REL:Unclassified"])) == set(adapter.entities())


def test_enumerators_honour_filter_obsoletes_and_id_prefixes(adapter) -> None:
    assert "OMOP:103" not in set(adapter.roots())
    assert "OMOP:103" in set(adapter.roots(filter_obsoletes=False))
    assert set(adapter.roots(id_prefixes=["ICD10"], filter_obsoletes=False)) == set()
    assert set(adapter.roots(id_prefixes=["SNOMED"])) == {"OMOP:100", "OMOP:105"}
    assert set(adapter.roots(id_prefixes=["OMOP"])) == {"OMOP:100", "OMOP:105"}


def test_an_empty_subject_list_selects_no_relationship(adapter) -> None:
    assert list(adapter.relationships(subjects=[])) == []
    assert list(adapter.relationships(objects=[])) == []


def test_relationship_metadata_is_the_validity_of_the_edge(adapter) -> None:
    triple = ("OMOP:101", IS_A, "OMOP:100")

    assert list(adapter.relationships_metadata([triple])) == [
        (
            triple,
            [("omop:valid_start_date", "2020-01-01"), ("omop:valid_end_date", "2099-12-31")],
        )
    ]
    assert list(adapter.relationships_metadata([("OMOP:100", IS_A, "OMOP:101")])) == [
        (("OMOP:100", IS_A, "OMOP:101"), [])
    ]


def test_all_relationships_stream_the_exposed_edges_only(adapter) -> None:
    assert set(adapter._all_relationships()) == {
        ("OMOP:101", IS_A, "OMOP:100"),
        ("OMOP:103", "OMOP_REL:Concept%20replaced%20by", "OMOP:101"),
        ("OMOP:101", "OMOP_REL:Concept%20replaces", "OMOP:103"),
    }
