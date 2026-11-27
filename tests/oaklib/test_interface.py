from __future__ import annotations

from typing import cast

import pytest
from linkml_runtime.linkml_model.annotations import Annotation
from oaklib.datamodels.text_annotator import TextAnnotationConfiguration
from oaklib.datamodels.vocabulary import (
    OWL_ANNOTATION_PROPERTY,
    OWL_CLASS,
    OWL_DATATYPE_PROPERTY,
    OWL_NAMED_INDIVIDUAL,
    OWL_OBJECT_PROPERTY,
)
from omop_alchemy.cdm.query import ConceptFilter

from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.oaklib_interface.omop_implementation import OMOPAlchemyImplementation
from omop_graph.oaklib_interface.text_annotator import AnnotationKey, OMOPTextAnnotatorInterface
from omop_graph.oaklib_interface.unsupported import OMOPUnsupportedError

WHOLE_TEXT = TextAnnotationConfiguration(matches_whole_text=True)


def test_annotate_text_standardizes_non_standard_candidate_without_parent_annotation(
    mock_cdm_kg: KnowledgeGraph,
) -> None:
    interface = OMOPTextAnnotatorInterface(kg=mock_cdm_kg)

    annotations = list(interface.annotate_text("Kidney carcinoma term", WHOLE_TEXT))

    assert len(annotations) == 1
    assert annotations[0].object_id == "OMOP:196653"


def test_annotate_text_yields_nothing_when_nothing_grounds(mock_cdm_kg: KnowledgeGraph) -> None:
    interface = OMOPTextAnnotatorInterface(kg=mock_cdm_kg)

    assert list(interface.annotate_text("zzzz qqqq nonsense", WHOLE_TEXT)) == []


@pytest.mark.parametrize(
    "configuration", [None, TextAnnotationConfiguration(), TextAnnotationConfiguration(limit=1)]
)
def test_annotate_text_without_whole_text_matching_is_unsupported(
    mock_cdm_kg: KnowledgeGraph, configuration: TextAnnotationConfiguration | None
) -> None:
    interface = OMOPTextAnnotatorInterface(kg=mock_cdm_kg)

    with pytest.raises(OMOPUnsupportedError, match="matches_whole_text"):
        interface.annotate_text("Kidney carcinoma term", configuration)


def test_annotate_text_rejects_a_vocabulary_that_does_not_exist(
    mock_cdm_kg: KnowledgeGraph,
) -> None:
    interface = OMOPTextAnnotatorInterface(kg=mock_cdm_kg)
    annotations = {AnnotationKey.VOCABS.value: Annotation(tag="vocabs", value="NoSuchVocabulary")}

    with pytest.raises(ValueError, match="NoSuchVocabulary"):
        list(interface.annotate_text("Kidney carcinoma term", WHOLE_TEXT, annotations=annotations))


def test_annotate_text_rejects_a_parent_id_that_names_no_concept(
    mock_cdm_kg: KnowledgeGraph,
) -> None:
    interface = OMOPTextAnnotatorInterface(kg=mock_cdm_kg)
    annotations = {
        AnnotationKey.PARENT_IDS.value: Annotation(tag="parent_ids", value="NoSuch:code")
    }

    with pytest.raises(ValueError, match="names no concept"):
        list(interface.annotate_text("Kidney carcinoma term", WHOLE_TEXT, annotations=annotations))


def test_annotate_text_stops_at_the_limit(mock_cdm_kg: KnowledgeGraph) -> None:
    interface = OMOPTextAnnotatorInterface(kg=mock_cdm_kg)
    text = "Kidney carcinoma term"
    unlimited = list(interface.annotate_text(text, WHOLE_TEXT))

    limited = TextAnnotationConfiguration(matches_whole_text=True, limit=0)

    assert unlimited
    assert list(interface.annotate_text(text, limited)) == []


def test_annotate_text_drops_excluded_tokens(mock_cdm_kg: KnowledgeGraph) -> None:
    interface = OMOPTextAnnotatorInterface(kg=mock_cdm_kg)
    configuration = TextAnnotationConfiguration(
        matches_whole_text=True, token_exclusion_list=["patient", "has"]
    )

    annotations = list(interface.annotate_text("patient has Kidney carcinoma term", configuration))

    assert [annotation.object_id for annotation in annotations] == ["OMOP:196653"]


def test_annotate_text_names_the_configuration_fields_it_ignores(
    mock_cdm_kg: KnowledgeGraph, caplog: pytest.LogCaptureFixture
) -> None:
    interface = OMOPTextAnnotatorInterface(kg=mock_cdm_kg)
    configuration = TextAnnotationConfiguration(
        matches_whole_text=True, sources=["SNOMED"], include_aliases=True
    )

    with caplog.at_level("INFO", logger="omop_graph.oaklib_interface.text_annotator"):
        list(interface.annotate_text("Kidney carcinoma term", configuration))

    assert "sources, include_aliases" in caplog.text


def test_the_full_adapter_annotates_text_and_names_the_ignored_fields(
    mock_cdm_kg: KnowledgeGraph, caplog: pytest.LogCaptureFixture
) -> None:
    interface = OMOPAlchemyImplementation(kg=mock_cdm_kg)
    configuration = TextAnnotationConfiguration(
        matches_whole_text=True, sources=["SNOMED"], include_aliases=True
    )

    with caplog.at_level("INFO", logger="omop_graph.oaklib_interface.text_annotator"):
        list(interface.annotate_text("Kidney carcinoma term", configuration))

    assert "sources, include_aliases" in caplog.text


def test_entities_lists_concepts_or_relationships_by_owl_type(mock_cdm_kg: KnowledgeGraph) -> None:
    adapter = OMOPAlchemyImplementation(kg=mock_cdm_kg)

    concepts = set(adapter.entities())
    predicates = set(adapter.entities(owl_type=OWL_OBJECT_PROPERTY))

    assert concepts == set(adapter.entities(owl_type=OWL_CLASS))
    assert concepts
    assert predicates == {"OMOP_REL:maps%20to", "OMOP_REL:mapped%20from"}
    assert all(
        list(adapter.entities(owl_type=owl_type)) == []
        for owl_type in (OWL_NAMED_INDIVIDUAL, OWL_DATATYPE_PROPERTY, OWL_ANNOTATION_PROPERTY)
    )


@pytest.mark.parametrize("owl_type", ["owl:class", "Class", ""])
def test_entities_rejects_a_string_that_is_not_an_owl_type(
    mock_cdm_kg: KnowledgeGraph, owl_type: str
) -> None:
    adapter = OMOPAlchemyImplementation(kg=mock_cdm_kg)

    with pytest.raises(ValueError, match="not an OWL type"):
        list(adapter.entities(owl_type=owl_type))


class _RecordingKnowledgeGraph:
    """Stands in for the KnowledgeGraph and records the filter ``entities`` is called with."""

    def __init__(self) -> None:
        self.scopes: list[ConceptFilter] = []
        self.obsolete_scopes: list[ConceptFilter] = []

    def entities(self, concept_filter: ConceptFilter, limit: int | None = None):
        self.scopes.append(concept_filter)
        return iter(())

    def obsolete_concepts(self, include_updated: bool, concept_filter: ConceptFilter):
        self.obsolete_scopes.append(concept_filter)
        return iter(())


@pytest.mark.parametrize(
    ("filter_obsoletes", "expected_obsolete_scopes"),
    [(True, []), (False, [ConceptFilter(domains=("Condition",))])],
)
def test_filter_obsoletes_applies_to_the_standing_scope(
    filter_obsoletes: bool, expected_obsolete_scopes: list[ConceptFilter]
) -> None:
    kg = _RecordingKnowledgeGraph()
    standing = ConceptFilter(domains=("Condition",), require_standard=True)
    adapter = OMOPAlchemyImplementation(kg=cast(KnowledgeGraph, kg), concept_filter=standing)

    list(adapter.entities(filter_obsoletes=filter_obsoletes))

    assert kg.scopes == [
        ConceptFilter(domains=("Condition",), require_standard=True, require_active=True)
    ]
    assert kg.obsolete_scopes == expected_obsolete_scopes


def test_an_explicit_concept_filter_replaces_the_standing_scope_as_given() -> None:
    kg = _RecordingKnowledgeGraph()
    adapter = OMOPAlchemyImplementation(kg=cast(KnowledgeGraph, kg))
    explicit = ConceptFilter(domains=("Drug",))

    list(adapter.entities(concept_filter=explicit))

    assert kg.scopes == [explicit]


def test_a_limit_keeps_the_concepts_with_the_lowest_ids_and_counts_the_obsoletes(
    mock_cdm_kg: KnowledgeGraph,
) -> None:
    adapter = OMOPAlchemyImplementation(kg=mock_cdm_kg)
    by_id = sorted(adapter.entities(), key=lambda curie: int(curie.split(":")[1]))

    assert list(adapter.entities(limit=2)) == by_id[:2]
    assert len(list(adapter.entities(filter_obsoletes=False, limit=len(by_id) + 1))) == (
        len(list(adapter.entities(filter_obsoletes=False)))
    )
    assert len(list(adapter.entities(filter_obsoletes=False, limit=1))) == 1


def test_an_explicit_concept_filter_that_sets_a_limit_is_refused(
    mock_cdm_kg: KnowledgeGraph,
) -> None:
    adapter = OMOPAlchemyImplementation(kg=mock_cdm_kg)

    with pytest.raises(ValueError, match="Pass limit to the method"):
        list(adapter.entities(concept_filter=ConceptFilter(limit=1)))


@pytest.mark.parametrize(
    "concept_filter", [ConceptFilter(limit=5), ConceptFilter(concept_ids=(1,))]
)
def test_a_standing_scope_must_not_truncate_enumeration(
    mock_cdm_kg: KnowledgeGraph, concept_filter: ConceptFilter
) -> None:
    with pytest.raises(ValueError, match="limit or concept_ids"):
        OMOPAlchemyImplementation(kg=mock_cdm_kg, concept_filter=concept_filter)


def test_relationships_follow_the_oaklib_flags(mock_cdm_kg: KnowledgeGraph) -> None:
    adapter = OMOPAlchemyImplementation(kg=mock_cdm_kg)

    assert list(adapter.relationships(subjects=["OMOP:900001"])) == [
        ("OMOP:900001", "OMOP_REL:maps%20to", "OMOP:196653")
    ]
    assert list(adapter.relationships(include_tbox=False)) == []
    assert list(adapter.relationships(include_abox=False)) == list(adapter.relationships())
    assert list(adapter.relationships(subjects=["OMOP:900001"], include_entailed=True)) == list(
        adapter.relationships(subjects=["OMOP:900001"])
    )
