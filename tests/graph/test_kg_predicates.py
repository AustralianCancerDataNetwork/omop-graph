"""Predicates are read once and looked up from memory."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fixtures.cdm_builders import classified, relationship, relationship_class, sqlite_cdm
from sqlalchemy import event

from omop_graph.extensions.omop_alchemy import (
    PredicateKind,
)
from omop_graph.graph.kg import KnowledgeGraph


@pytest.fixture(scope="module")
def kg() -> Iterator[KnowledgeGraph]:
    rows = [
        relationship("Maps to", "Mapped from"),
        relationship("Unclassified", "Unclassified of"),
        relationship_class(PredicateKind.IDENTITY, "Exact"),
        classified("Maps to", PredicateKind.IDENTITY, "Exact"),
    ]
    with sqlite_cdm(rows) as sessions:
        yield KnowledgeGraph(sessions)


def test_a_classified_relationship_has_its_predicate_and_name(kg) -> None:
    assert kg.predicate("Maps to").reverse_id == "Mapped from"
    assert kg.find_predicate("Maps to") == kg.predicate("Maps to")
    assert kg.predicate_name("Maps to") == "Maps to"


def test_an_unclassified_relationship_has_no_predicate(kg) -> None:
    assert kg.find_predicate("Unclassified") is None
    with pytest.raises(AttributeError, match="Unclassified"):
        kg.predicate("Unclassified")
    with pytest.raises(AttributeError, match="Unclassified"):
        kg.predicate_name("Unclassified")


def test_the_predicates_are_read_once(kg) -> None:
    kg.predicates()
    statements: list[str] = []
    with kg.session_factory() as session:
        engine = session.get_bind()

    def record(conn, cursor, statement, *args):
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", record)
    try:
        for _ in range(3):
            kg.predicate("Maps to")
            kg.predicate_name("Maps to")
            kg.find_predicate("Maps to")
            kg.predicates()
    finally:
        event.remove(engine, "before_cursor_execute", record)

    assert statements == []
