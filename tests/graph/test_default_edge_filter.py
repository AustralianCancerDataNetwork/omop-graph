"""The default edge filter leaves out metadata relationships, and an empty set selects nothing."""

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
    HierarchyDirection,
    PredicateKind,
    PredicateSubkind,
)
from omop_graph.graph.edges import DEFAULT_EDGE_FILTER, EdgeFilter
from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.graph.paths import DEFAULT_PATH_FILTER, find_shortest_paths
from omop_graph.graph.traverse import traverse

LEFT, RIGHT, PARENT, MODULE = 500, 501, 502, 503
METADATA_ONLY = EdgeFilter(predicate_kinds=frozenset({PredicateKind.METADATA}), within_domain=False)


@pytest.fixture(scope="module")
def kg() -> Iterator[KnowledgeGraph]:
    """Two concepts that share a module, and a parent that only the first has."""
    rows = [
        *(concept(concept_id) for concept_id in (LEFT, RIGHT, PARENT, MODULE)),
        relationship("Is a", "Subsumes"),
        relationship("Has Module", "Module of"),
        relationship("Module of", "Has Module"),
        relationship_class(PredicateKind.HIERARCHY, "Taxonomic"),
        relationship_class(PredicateKind.METADATA, "Module membership"),
        classified(
            "Is a",
            PredicateKind.HIERARCHY,
            "Taxonomic",
            generalises=True,
            direction=HierarchyDirection.UP,
        ),
        classified("Has Module", PredicateKind.METADATA, "Module membership"),
        classified("Module of", PredicateKind.METADATA, "Module membership"),
        edge(LEFT, "Is a", PARENT),
        edge(LEFT, "Has Module", MODULE),
        edge(RIGHT, "Has Module", MODULE),
        edge(MODULE, "Module of", LEFT),
        edge(MODULE, "Module of", RIGHT),
    ]
    with sqlite_cdm(rows) as sessions:
        yield KnowledgeGraph(sessions)


def _predicates(edges) -> set[str]:
    return {edge.predicate_id for edge in edges}


def test_the_default_leaves_out_the_metadata_relationships(kg) -> None:
    assert _predicates(kg.edges(LEFT, "out")) == {"Is a"}
    assert {p for _, p, _ in kg.relationships((LEFT,))} == {"Is a"}
    assert kg.default_relationship_ids == {"Is a"}


def test_none_selects_every_classified_relationship_and_an_explicit_kind_selects_its_own(
    kg,
) -> None:
    assert _predicates(kg.edges(LEFT, "out", edge_filter=None)) == {"Is a", "Has Module"}
    assert _predicates(kg.edges(LEFT, "out", edge_filter=METADATA_ONLY)) == {"Has Module"}


def test_the_default_filter_names_every_kind_but_metadata() -> None:
    assert DEFAULT_EDGE_FILTER.predicate_kinds == frozenset(PredicateKind) - {
        PredicateKind.METADATA
    }
    assert DEFAULT_PATH_FILTER.predicate_kinds == DEFAULT_EDGE_FILTER.predicate_kinds
    assert DEFAULT_PATH_FILTER.within_domain


def test_two_concepts_that_only_share_a_module_have_no_path_by_default(kg) -> None:
    default, _ = find_shortest_paths(kg, LEFT, RIGHT, edge_filter=DEFAULT_PATH_FILTER)
    through_module, _ = find_shortest_paths(kg, LEFT, RIGHT, edge_filter=METADATA_ONLY)

    assert default == []
    assert through_module


def test_a_traversal_does_not_expand_into_the_module_by_default(kg) -> None:
    default, _ = traverse(kg, (LEFT,), max_depth=2, max_nodes=None, trace=False)
    through_module, _ = traverse(
        kg, (LEFT,), edge_filter=METADATA_ONLY, max_depth=2, max_nodes=None, trace=False
    )

    assert MODULE not in default.nodes
    assert {MODULE, RIGHT} <= through_module.nodes


def test_an_empty_set_of_kinds_or_subkinds_selects_no_edge(kg) -> None:
    no_kind = EdgeFilter(predicate_kinds=frozenset())
    no_subkind = EdgeFilter(predicate_subkinds=frozenset())
    metadata = frozenset({PredicateKind.METADATA})
    one_subkind = frozenset({PredicateSubkind(PredicateKind.METADATA, "Module membership")})

    for nothing in (
        no_kind,
        no_subkind,
        EdgeFilter(predicate_kinds=metadata, predicate_subkinds=frozenset()),
    ):
        assert kg.edges(LEFT, "out", edge_filter=nothing) == ()
        assert tuple(kg.relationships((LEFT,), edge_filter=nothing)) == ()
    assert _predicates(
        kg.edges(LEFT, "out", edge_filter=EdgeFilter(predicate_subkinds=one_subkind))
    ) == {"Has Module"}
