"""Edge classification on a colocated database.

``KnowledgeGraph`` classifies every edge from the relationship mapping read at
construction, on any topology. These pin the colocated case, the one most
deployments use, including the ``predicate_kinds`` filter applied in SQL as
relationship IDs.
"""

from __future__ import annotations

import pytest

from omop_graph.extensions.omop_alchemy import PredicateKind
from omop_graph.graph.kg import KnowledgeGraph

pytestmark = [pytest.mark.postgresql, pytest.mark.db_dialect]


def test_edges_are_classified_from_the_relationship_mapping(
    mock_cdm_kg: KnowledgeGraph,
) -> None:
    edges = mock_cdm_kg.edges(
        concept_ids=900001,
        direction="out",
        active_only=False,
        within_domain=False,
    )

    assert len(edges) == 1
    edge = edges[0]
    assert edge.subject_id == 900001
    assert edge.object_id == 196653
    assert edge.predicate_id == "maps to"
    assert edge.predicate_kind == PredicateKind.IDENTITY
    assert edge.predicate_subkind == "mapping"


@pytest.mark.parametrize(("kinds", "expected"), [({PredicateKind.IDENTITY}, 1), ({PredicateKind.HIERARCHY}, 0)])
def test_predicate_kinds_filter_edges(mock_cdm_kg: KnowledgeGraph, kinds, expected) -> None:
    edges = mock_cdm_kg.edges(
        concept_ids=900001,
        direction="out",
        predicate_kinds=frozenset(kinds),
        active_only=False,
        within_domain=False,
    )

    assert len(edges) == expected
