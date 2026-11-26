from __future__ import annotations

from collections.abc import Iterator

import pytest
from fixtures.cdm_builders import (
    classified,
    concept,
    edge,
    relationship,
    relationship_class,
    sqlite_cdm,
)
from omop_alchemy.cdm.model.vocabulary.concept import Concept
from omop_alchemy.cdm.model.vocabulary.concept_relationship import Concept_Relationship
from omop_alchemy.cdm.model.vocabulary.concept_synonym import Concept_Synonym
from omop_alchemy.cdm.model.vocabulary.relationship import Relationship
from omop_alchemy.cdm.model.vocabulary.vocabulary import Vocabulary

from omop_graph.extensions.omop_alchemy import (
    PredicateKind,
    RelationshipClass,
    RelationshipMapping,
)
from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.graph.nodes import Standardness
from omop_graph.oaklib_interface.codec import OMOPMetadataKey
from omop_graph.oaklib_interface.omop_implementation import OMOPAlchemyImplementation

STANDARD, CLASSIFICATION, SOURCE, DEPRECATED = range(500, 504)
REPLACED_BY = "IAO:0100001"


@pytest.fixture(scope="module")
def kg() -> Iterator[KnowledgeGraph]:
    rows = [
        concept(STANDARD),
        concept(CLASSIFICATION, standard_concept="C"),
        concept(SOURCE, standard_concept=None),
        concept(DEPRECATED, standard_concept=None, invalid_reason="D"),
        relationship("Maps to", "Mapped from"),
        relationship("Mapped from", "Maps to"),
        relationship_class(PredicateKind.IDENTITY, "Exact"),
        classified("Maps to", PredicateKind.IDENTITY, "Exact"),
        classified("Mapped from", PredicateKind.IDENTITY, "Exact"),
        edge(SOURCE, "Maps to", STANDARD),
        edge(STANDARD, "Mapped from", SOURCE),
    ]
    models = (
        Concept,
        Relationship,
        Concept_Relationship,
        Concept_Synonym,
        Vocabulary,
        RelationshipClass,
        RelationshipMapping,
    )
    with sqlite_cdm(models, rows) as sessions:
        yield KnowledgeGraph(sessions)


def test_a_concept_has_the_standardness_of_its_flag(kg) -> None:
    flags = {view.concept_id: view.standardness for view in kg.concept_views(range(500, 504))}

    assert flags == {
        STANDARD: Standardness.STANDARD,
        CLASSIFICATION: Standardness.CLASSIFICATION,
        SOURCE: Standardness.NONE,
        DEPRECATED: Standardness.NONE,
    }


def test_only_a_standard_concept_is_a_mapping_target(kg) -> None:
    assert kg.standardise([STANDARD, CLASSIFICATION, SOURCE]) == {
        STANDARD: (STANDARD,),
        CLASSIFICATION: (),
        SOURCE: (STANDARD,),
    }


def test_the_metadata_of_a_concept_carries_its_raw_standardness_flag(kg) -> None:
    adapter = OMOPAlchemyImplementation(kg=kg)
    metadata = adapter.entity_metadata_map(f"OMOP:{CLASSIFICATION}")

    assert metadata[OMOPMetadataKey.STANDARD_CONCEPT] == ["C"]
    assert OMOPMetadataKey.STANDARD_CONCEPT not in adapter.entity_metadata_map(f"OMOP:{SOURCE}")


def test_a_deprecated_concept_without_a_replacement_has_no_replacement_key(kg) -> None:
    metadata = OMOPAlchemyImplementation(kg=kg).entity_metadata_map(f"OMOP:{DEPRECATED}")

    assert REPLACED_BY not in metadata
