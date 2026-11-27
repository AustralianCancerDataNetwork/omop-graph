"""The capability table of the documentation is generated from the mixins of the graph."""

from __future__ import annotations

import inspect

from fixtures.helpers import docs_macros

from omop_graph.graph.kg import KnowledgeGraph


def test_the_table_has_a_row_for_each_capability_and_lists_every_public_member() -> None:
    table = docs_macros()["graph_capabilities"]()
    rows = {line.split("|")[1].strip(): line for line in table.splitlines()[2:]}
    public = {
        name
        for name, _ in inspect.getmembers(KnowledgeGraph)
        if not name.startswith("_") and name != "LOOKUP_BATCH_SIZE"
    }

    assert "`ConceptsMixin`" in rows["`HierarchyMixin`"]
    assert "`KnowledgeGraph`" not in rows
    assert {name for name in public if f"`{name}`" not in table} == set()
