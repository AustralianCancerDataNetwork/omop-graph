"""A relationship that generalises holds for the subtypes of its subject, on a small CDM."""

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
from fixtures.helpers import MAPS_TO, SITE, curie
from oaklib.datamodels.vocabulary import IS_A
from omop_alchemy.cdm.model.vocabulary.concept import Concept
from omop_alchemy.cdm.model.vocabulary.concept_ancestor import Concept_Ancestor
from omop_alchemy.cdm.model.vocabulary.concept_relationship import Concept_Relationship
from omop_alchemy.cdm.model.vocabulary.concept_synonym import Concept_Synonym
from omop_alchemy.cdm.model.vocabulary.relationship import Relationship
from omop_alchemy.cdm.model.vocabulary.vocabulary import Vocabulary

from omop_graph.extensions.omop_alchemy import (
    HierarchyDirection,
    PredicateKind,
    RelationshipClass,
    RelationshipMapping,
)
from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.oaklib_interface.omop_implementation import OMOPAlchemyImplementation

DISEASE, LUNG_CANCER, SMALL_CELL, LUNG, THORAX, RIGHT_LUNG, SOURCE, SOURCE_CHILD = range(300, 308)
SITE_OF = "OMOP_REL:Finding%20site%20of"


@pytest.fixture(scope="module")
def adapter() -> Iterator[OMOPAlchemyImplementation]:
    rows = [
        *(concept(concept_id) for concept_id in (DISEASE, LUNG_CANCER, SMALL_CELL)),
        *(concept(concept_id) for concept_id in (LUNG, THORAX, RIGHT_LUNG)),
        concept(SOURCE, standard_concept=None),
        concept(SOURCE_CHILD, standard_concept=None),
        relationship("Is a", "Subsumes"),
        relationship("Has finding site", "Finding site of"),
        relationship("Finding site of", "Has finding site"),
        relationship("Maps to", "Mapped from"),
        relationship_class(PredicateKind.IDENTITY, "Exact"),
        relationship_class(PredicateKind.HIERARCHY, "Taxonomic - up"),
        relationship_class(PredicateKind.ATTRIBUTE, "Site"),
        classified("Maps to", PredicateKind.IDENTITY, "Exact"),
        classified(
            "Is a",
            PredicateKind.HIERARCHY,
            "Taxonomic - up",
            generalises=True,
            direction=HierarchyDirection.UP,
        ),
        classified("Has finding site", PredicateKind.ATTRIBUTE, "Site", generalises=True),
        classified("Finding site of", PredicateKind.ATTRIBUTE, "Site"),
        edge(LUNG_CANCER, "Is a", DISEASE),
        edge(SMALL_CELL, "Is a", LUNG_CANCER),
        edge(LUNG, "Is a", THORAX),
        edge(RIGHT_LUNG, "Is a", LUNG),
        edge(SOURCE_CHILD, "Is a", SOURCE),
        edge(LUNG_CANCER, "Has finding site", LUNG),
        edge(LUNG, "Finding site of", LUNG_CANCER),
        edge(SOURCE, "Maps to", LUNG_CANCER),
    ]
    models = (
        Concept,
        Relationship,
        Concept_Relationship,
        RelationshipClass,
        RelationshipMapping,
        Concept_Ancestor,
        Concept_Synonym,
        Vocabulary,
    )
    with sqlite_cdm(models, rows) as sessions:
        yield OMOPAlchemyImplementation(kg=KnowledgeGraph(sessions))


def test_a_subtype_has_the_relationships_of_its_supertypes(adapter) -> None:
    assert set(adapter.entailed_outgoing_relationships(curie(SMALL_CELL), [SITE])) == {
        (SITE, curie(LUNG)),
        (SITE, curie(THORAX)),
    }


def test_a_subtype_is_a_subject_of_the_relationships_into_its_supertypes_fillers(adapter) -> None:
    assert set(adapter.entailed_incoming_relationships(curie(THORAX), [SITE])) == {
        (SITE, curie(LUNG_CANCER)),
        (SITE, curie(SMALL_CELL)),
    }
    assert set(adapter.entailed_incoming_relationships(curie(LUNG), [SITE])) == {
        (SITE, curie(LUNG_CANCER)),
        (SITE, curie(SMALL_CELL)),
    }


def test_the_inheritance_matches_in_both_directions(adapter) -> None:
    outgoing = {
        (curie(SMALL_CELL), p, o)
        for p, o in adapter.entailed_outgoing_relationships(curie(SMALL_CELL), [SITE])
    }
    incoming = {
        (s, p, o)
        for o in (curie(LUNG), curie(THORAX))
        for p, s in adapter.entailed_incoming_relationships(o, [SITE])
        if s == curie(SMALL_CELL)
    }

    assert incoming == outgoing


def test_is_a_gives_the_ancestors_once(adapter) -> None:
    triples = list(
        adapter.relationships(
            subjects=[curie(SMALL_CELL)], predicates=[IS_A], include_entailed=True
        )
    )

    assert len(triples) == len(set(triples))
    assert {o for _, _, o in triples} == {curie(LUNG_CANCER), curie(DISEASE)}


def test_the_other_side_of_a_relationship_is_not_inherited(adapter) -> None:
    assert set(adapter.entailed_outgoing_relationships(curie(RIGHT_LUNG), [SITE_OF])) == set()
    assert set(adapter.entailed_outgoing_relationships(curie(LUNG), [SITE_OF])) == {
        (SITE_OF, curie(LUNG_CANCER))
    }


def test_a_mapping_is_not_inherited(adapter) -> None:
    assert set(adapter.entailed_outgoing_relationships(curie(SOURCE_CHILD), [MAPS_TO])) == set()
    assert set(adapter.entailed_outgoing_relationships(curie(SOURCE), [MAPS_TO])) == {
        (MAPS_TO, curie(LUNG_CANCER))
    }


def test_the_stream_holds_the_inherited_relationships(adapter) -> None:
    streamed = set(adapter._all_entailed_relationships())

    assert (curie(SMALL_CELL), SITE, curie(THORAX)) in streamed
    assert (curie(RIGHT_LUNG), SITE_OF, curie(LUNG_CANCER)) not in streamed
