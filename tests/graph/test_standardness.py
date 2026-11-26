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

from omop_graph.extensions.omop_alchemy import (
    PredicateKind,
)
from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.graph.model.nodes import Standardness
from omop_graph.oaklib_interface.codec import OMOPMetadataKey
from omop_graph.oaklib_interface.omop_implementation import OMOPAlchemyImplementation

STANDARD, CLASSIFICATION, SOURCE, DEPRECATED, UNMAPPED_CLASSIFICATION, TO_CLASSIFICATION = range(
    500, 506
)


@pytest.fixture(scope="module")
def kg() -> Iterator[KnowledgeGraph]:
    rows = [
        concept(STANDARD),
        concept(CLASSIFICATION, standard_concept="C"),
        concept(SOURCE, standard_concept=None),
        concept(DEPRECATED, standard_concept=None, invalid_reason="D"),
        concept(UNMAPPED_CLASSIFICATION, standard_concept="C"),
        concept(TO_CLASSIFICATION, standard_concept=None),
        relationship("Maps to", "Mapped from"),
        relationship("Mapped from", "Maps to"),
        relationship_class(PredicateKind.IDENTITY, "Exact"),
        classified("Maps to", PredicateKind.IDENTITY, "Exact"),
        classified("Mapped from", PredicateKind.IDENTITY, "Exact"),
        edge(SOURCE, "Maps to", STANDARD),
        edge(STANDARD, "Mapped from", SOURCE),
        edge(CLASSIFICATION, "Maps to", STANDARD),
        edge(TO_CLASSIFICATION, "Maps to", UNMAPPED_CLASSIFICATION),
    ]
    with sqlite_cdm(rows) as sessions:
        yield KnowledgeGraph(sessions)


def test_a_concept_has_the_standardness_of_its_flag(kg) -> None:
    flags = {view.concept_id: view.standardness for view in kg.concept_views(range(500, 504))}

    assert flags == {
        STANDARD: Standardness.STANDARD,
        CLASSIFICATION: Standardness.CLASSIFICATION,
        SOURCE: Standardness.NONE,
        DEPRECATED: Standardness.NONE,
    }


def test_the_views_come_best_match_first_unless_unsorted(kg) -> None:
    ids = range(STANDARD, TO_CLASSIFICATION + 1)

    assert [view.concept_id for view in kg.concept_views(ids)] == [
        STANDARD,
        CLASSIFICATION,
        UNMAPPED_CLASSIFICATION,
        SOURCE,
        TO_CLASSIFICATION,
        DEPRECATED,
    ]
    assert [view.concept_id for view in kg.concept_views(ids, sort=False)] == list(ids)


def test_a_classification_concept_stands_for_the_standard_concept_it_maps_to(kg) -> None:
    assert kg.standardise([STANDARD, CLASSIFICATION, SOURCE]) == {
        STANDARD: (STANDARD,),
        CLASSIFICATION: (STANDARD,),
        SOURCE: (STANDARD,),
    }


def test_only_a_standard_concept_is_a_mapping_target(kg) -> None:
    assert kg.standardise([UNMAPPED_CLASSIFICATION, TO_CLASSIFICATION]) == {
        UNMAPPED_CLASSIFICATION: (),
        TO_CLASSIFICATION: (),
    }


def test_the_metadata_of_a_concept_carries_its_raw_standardness_flag(kg) -> None:
    adapter = OMOPAlchemyImplementation(kg=kg)
    metadata = adapter.entity_metadata_map(f"OMOP:{CLASSIFICATION}")

    assert metadata[OMOPMetadataKey.STANDARD_CONCEPT] == ["C"]
    assert OMOPMetadataKey.STANDARD_CONCEPT not in adapter.entity_metadata_map(f"OMOP:{SOURCE}")


def test_a_deprecated_concept_without_a_replacement_has_no_replacement_key(kg) -> None:
    metadata = OMOPAlchemyImplementation(kg=kg).entity_metadata_map(f"OMOP:{DEPRECATED}")

    assert "IAO:0100001" not in metadata
