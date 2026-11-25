from __future__ import annotations

from typing import cast

import pytest
from oaklib.datamodels.vocabulary import IS_A

from omop_graph.extensions.omop_alchemy import HierarchyDirection, PredicateKind
from omop_graph.graph.edges import EdgeFilter, Predicate
from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.oaklib_interface.omop_implementation import OMOPAlchemyImplementation


def _predicate(
    relationship_id: str, kind: PredicateKind, direction: HierarchyDirection | None = None
) -> Predicate:
    return Predicate(
        relationship_id=relationship_id,
        name=relationship_id,
        reverse_id=None,
        predicate_kind=kind,
        predicate_subkind="x",
        direction=direction,
    )


class _FakeKnowledgeGraph:
    """Serves a fixed predicate table and records the filter ``relationships`` receives."""

    def __init__(self) -> None:
        self.filters: list[EdgeFilter] = []

    def predicates(self) -> tuple[Predicate, ...]:
        return (
            _predicate("Is a", PredicateKind.HIERARCHY, HierarchyDirection.UP),
            _predicate("Subsumes", PredicateKind.HIERARCHY, HierarchyDirection.DOWN),
            _predicate("Has dir proc site", PredicateKind.ATTRIBUTE),
            _predicate("Maps to", PredicateKind.IDENTITY),
        )

    def relationships(self, subjects, objects, edge_filter: EdgeFilter, invert: bool = False):
        self.filters.append(edge_filter)
        return iter(())


@pytest.fixture
def kg() -> _FakeKnowledgeGraph:
    return _FakeKnowledgeGraph()


@pytest.fixture
def adapter(kg: _FakeKnowledgeGraph) -> OMOPAlchemyImplementation:
    return OMOPAlchemyImplementation(kg=cast(KnowledgeGraph, kg))


@pytest.mark.parametrize(
    ("relationship_id", "curie"),
    [
        ("Is a", IS_A),
        ("Maps to", "OMOP_REL:Maps%20to"),
        ("Has dir proc site", "OMOP_REL:Has%20dir%20proc%20site"),
        ("a/b:c%d", "OMOP_REL:a%2Fb%3Ac%25d"),
    ],
)
def test_predicate_curies_round_trip(
    adapter: OMOPAlchemyImplementation, relationship_id: str, curie: str
) -> None:
    assert adapter.codec.predicate_curie(relationship_id) == curie
    assert adapter.codec.parse_predicate(curie) == relationship_id


def test_is_a_is_also_accepted_in_its_omop_rel_form(adapter: OMOPAlchemyImplementation) -> None:
    assert adapter.codec.parse_predicate("OMOP_REL:Is%20a") == "Is a"


def test_a_predicate_outside_omop_names_no_relationship(
    adapter: OMOPAlchemyImplementation,
) -> None:
    assert adapter.codec.parse_predicate("skos:exactMatch") is None


def test_only_the_up_direction_of_hierarchy_relationships_is_exposed_by_default(
    adapter: OMOPAlchemyImplementation, kg: _FakeKnowledgeGraph
) -> None:
    list(adapter.relationships())

    assert kg.filters[0].predicate_ids == {"Is a", "Has dir proc site", "Maps to"}


def test_the_down_direction_is_available_by_naming_it(
    adapter: OMOPAlchemyImplementation, kg: _FakeKnowledgeGraph
) -> None:
    list(adapter.relationships(predicates=["OMOP_REL:Subsumes", IS_A]))

    assert kg.filters[0].predicate_ids == {"Subsumes", "Is a"}
