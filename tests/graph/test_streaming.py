"""A consumer that stops reading a streamed result gives its connection back."""

from __future__ import annotations

from fixtures.hierarchy_cdm import hierarchy_cdm
from sqlalchemy import event

from omop_graph.graph.kg import KnowledgeGraph


def test_stopping_an_enumeration_early_returns_the_connection_to_the_pool() -> None:
    with hierarchy_cdm() as sessions:
        kg = KnowledgeGraph(sessions)
        in_use = 0

        def checked_out(*_) -> None:
            nonlocal in_use
            in_use += 1

        def checked_in(*_) -> None:
            nonlocal in_use
            in_use -= 1

        engine = sessions.kw["bind"]
        event.listen(engine, "checkout", checked_out)
        event.listen(engine, "checkin", checked_in)

        entities = kg.entities()
        next(entities)
        assert in_use == 1

        entities.close()
        assert in_use == 0
