from __future__ import annotations

from collections.abc import Iterator

import pytest
from fixtures.hierarchy_cdm import (
    CANCER,
    DISEASE,
    LUNG,
    LUNG_CANCER,
    ORPHAN,
    OUTER_ROOT,
    SMOKING_RELATED,
    SOURCE,
    SOURCE_TWO,
    THORAX_ORGAN,
    hierarchy_cdm,
)

from omop_graph.graph.edges import EdgeFilter
from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.graph.queries import DEFAULT_CONCEPT_FILTER

ENTAILING = EdgeFilter(predicate_ids=frozenset({"Is a", "Has finding site"}))


@pytest.fixture(scope="module")
def kg() -> Iterator[KnowledgeGraph]:
    with hierarchy_cdm() as sessions:
        yield KnowledgeGraph(sessions)


def test_the_relationships_that_generalise_are_the_flagged_ones(kg) -> None:
    assert kg.entailing_relationship_ids == {"Is a", "Has finding site"}


def test_the_default_relationships_leave_out_the_down_direction_of_a_hierarchy(kg) -> None:
    assert kg.default_relationship_ids == {
        "Is a",
        "Has finding site",
        "Finding site of",
        "Maps to",
        "Mapped from",
    }


def test_entailed_relationships_need_subjects_or_objects(kg) -> None:
    with pytest.raises(ValueError, match="subjects or objects"):
        list(kg.entailed_relationships(None, None))


def test_a_relationship_that_generalises_is_entailed_and_the_others_are_stored(kg) -> None:
    triples = list(kg.entailed_relationships((LUNG_CANCER,), None))

    assert (LUNG_CANCER, "Has finding site", THORAX_ORGAN) in triples
    assert (LUNG_CANCER, "Is a", DISEASE) in triples
    assert (LUNG_CANCER, "Mapped from", SOURCE) in triples
    assert len(triples) == len(set(triples))
    assert not {t for t in triples if t[1] == "Mapped from" and t[2] != SOURCE}


def test_the_stored_relationships_come_before_the_entailed_ones(kg) -> None:
    triples = list(kg.entailed_relationships((LUNG_CANCER,), None))
    stored = [
        i for i, (_, relationship_id, _) in enumerate(triples) if relationship_id == "Mapped from"
    ]

    assert stored == [0]


def test_a_filter_selecting_no_relationship_gives_nothing(kg) -> None:
    nothing = EdgeFilter(predicate_ids=frozenset())

    assert list(kg.entailed_relationships((LUNG_CANCER,), None, nothing)) == []


def test_entailed_relationships_into_objects_reach_the_subtypes(kg) -> None:
    triples = set(kg.entailed_relationships(None, (THORAX_ORGAN,), ENTAILING))

    assert (LUNG_CANCER, "Has finding site", THORAX_ORGAN) in triples
    assert (LUNG, "Is a", THORAX_ORGAN) in triples


def test_a_non_standard_concept_has_no_entailed_relationships_of_its_own(kg) -> None:
    assert list(kg.entailed_relationships((SOURCE,), None, ENTAILING)) == []


def test_standardise_lends_the_entailed_relationships_of_the_standard_concepts(kg) -> None:
    triples = set(
        kg.entailed_relationships(
            (SOURCE,), None, ENTAILING, DEFAULT_CONCEPT_FILTER, standardise=True
        )
    )

    assert triples == {
        (SOURCE, "Is a", CANCER),
        (SOURCE, "Is a", DISEASE),
        (SOURCE, "Is a", SMOKING_RELATED),
        (SOURCE, "Is a", OUTER_ROOT),
        (SOURCE, "Has finding site", LUNG),
        (SOURCE, "Has finding site", THORAX_ORGAN),
    }


def test_standardise_reports_incoming_relationships_for_the_concept_standing_in(kg) -> None:
    triples = set(kg.entailed_relationships(None, (SOURCE_TWO,), ENTAILING, standardise=True))

    assert (LUNG_CANCER, "Has finding site", SOURCE_TWO) in triples
    assert {t[2] for t in triples} == {SOURCE_TWO}


def test_standardise_does_not_lend_stored_relationships(kg) -> None:
    triples = set(kg.entailed_relationships((SOURCE,), None, standardise=True))

    assert {t for t in triples if t[1] in {"Mapped from", "Maps to"}} == {
        (SOURCE, "Maps to", LUNG_CANCER)
    }


def test_a_concept_with_no_standard_concept_has_nothing_to_lend(kg) -> None:
    assert list(kg.entailed_relationships((ORPHAN,), None, ENTAILING, standardise=True)) == []


def test_streaming_every_concept_gives_what_asking_for_all_of_them_gives(kg) -> None:
    concept_ids = tuple(kg.entities(DEFAULT_CONCEPT_FILTER))

    assert set(kg.all_entailed_relationships(ENTAILING)) == set(
        kg.entailed_relationships(concept_ids, None, ENTAILING, DEFAULT_CONCEPT_FILTER)
    )


def test_streaming_with_standardise_reports_for_each_concept_in_scope(kg) -> None:
    streamed = set(kg.all_entailed_relationships(ENTAILING, standardise=True))

    assert (SOURCE, "Has finding site", LUNG) not in streamed
