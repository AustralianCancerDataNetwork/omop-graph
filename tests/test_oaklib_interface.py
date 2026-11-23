from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest
from oaklib.datamodels.vocabulary import (
    OWL_ANNOTATION_PROPERTY,
    OWL_CLASS,
    OWL_DATATYPE_PROPERTY,
    OWL_NAMED_INDIVIDUAL,
    OWL_OBJECT_PROPERTY,
)
from omop_alchemy.cdm.query import ConceptFilter

from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.oaklib_interface.omop_implementation import (
    OMOPAlchemyImplementation,
    OMOPTextAnnotatorInterface,
)


def test_annotate_text_standardizes_non_standard_candidate_without_parent_annotation(
    mock_cdm_kg: KnowledgeGraph,
) -> None:
    interface = OMOPTextAnnotatorInterface(kg=mock_cdm_kg)

    annotations = list(interface.annotate_text(text="Kidney carcinoma term", annotations=None))

    assert len(annotations) == 1
    assert annotations[0].object_id == "OMOP:196653"


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

    def entities(self, concept_filter: ConceptFilter):
        self.scopes.append(concept_filter)
        return iter(())


@pytest.mark.parametrize("filter_obsoletes", [True, False])
def test_filter_obsoletes_applies_to_the_standing_scope(filter_obsoletes: bool) -> None:
    kg = _RecordingKnowledgeGraph()
    standing = ConceptFilter(domains=("Condition",), require_standard=True)
    adapter = OMOPAlchemyImplementation(kg=cast(KnowledgeGraph, kg), concept_filter=standing)

    list(adapter.entities(filter_obsoletes=filter_obsoletes))

    assert kg.scopes == [replace(standing, require_active=filter_obsoletes)]


def test_an_explicit_concept_filter_replaces_the_standing_scope_as_given() -> None:
    kg = _RecordingKnowledgeGraph()
    adapter = OMOPAlchemyImplementation(kg=cast(KnowledgeGraph, kg))
    explicit = ConceptFilter(domains=("Drug",))

    list(adapter.entities(concept_filter=explicit))

    assert kg.scopes == [explicit]


@pytest.mark.parametrize("scope", [ConceptFilter(limit=5), ConceptFilter(concept_ids=(1,))])
def test_a_standing_scope_must_not_truncate_enumeration(
    mock_cdm_kg: KnowledgeGraph, scope: ConceptFilter
) -> None:
    with pytest.raises(ValueError, match="limit or concept_ids"):
        OMOPAlchemyImplementation(kg=mock_cdm_kg, concept_filter=scope)


def test_relationships_follow_the_oaklib_flags(mock_cdm_kg: KnowledgeGraph) -> None:
    adapter = OMOPAlchemyImplementation(kg=mock_cdm_kg)

    assert list(adapter.relationships(subjects=["OMOP:900001"])) == [
        ("OMOP:900001", "OMOP_REL:maps%20to", "OMOP:196653")
    ]
    assert list(adapter.relationships(include_tbox=False)) == []
    assert list(adapter.relationships(include_abox=False)) == list(adapter.relationships())
    with pytest.raises(NotImplementedError, match="include_entailed"):
        list(adapter.relationships(include_entailed=True))
