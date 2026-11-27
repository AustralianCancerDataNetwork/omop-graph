from __future__ import annotations

from contextlib import nullcontext
from datetime import date
from typing import cast

from sqlalchemy.dialects import postgresql

from omop_graph.extensions.relationship_classification import PredicateKind
from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.graph.model.edges import EdgeFilter, EdgeView
from omop_graph.graph.queries import q_edges
from omop_graph.graph.traverse import traverse


class FakeKnowledgeGraph:
    def __init__(self, adjacency: dict[int, tuple[int, ...]]) -> None:
        self.adjacency = adjacency
        self.materialized_edges = 0
        self.closed_edge_iterators = 0

    @staticmethod
    def session_factory():
        return nullcontext(None)

    def iter_edges(self, *, concept_ids: int, ordered: bool = False, **_kwargs):
        try:
            object_ids = self.adjacency.get(concept_ids, ())
            if ordered:
                object_ids = tuple(sorted(object_ids))
            for object_id in object_ids:
                self.materialized_edges += 1
                yield EdgeView(
                    subject_id=concept_ids,
                    predicate_id="is a",
                    object_id=object_id,
                    valid_start_date=date(2000, 1, 1),
                    valid_end_date=date(2099, 12, 31),
                    invalid_reason=None,
                    predicate_kind=PredicateKind.HIERARCHY,
                    predicate_subkind="test",
                )
        finally:
            self.closed_edge_iterators += 1


def _edge_keys(subgraph) -> set[tuple[int, str, int]]:
    return {(edge.subject_id, edge.predicate_id, edge.object_id) for edge in subgraph.edges}


def _distances(adjacency: dict[int, tuple[int, ...]], seed: int) -> dict[int, int]:
    distances = {seed: 0}
    queue = [seed]
    for node in queue:
        for target in adjacency.get(node, ()):
            if target not in distances:
                distances[target] = distances[node] + 1
                queue.append(target)
    return distances


def test_uncapped_traversal_keeps_existing_nodes_and_edges() -> None:
    kg = FakeKnowledgeGraph({1: (2, 3), 2: (4,), 3: (5,), 4: (6,), 5: ()})

    subgraph, trace = traverse(
        cast(KnowledgeGraph, kg),
        seeds=(1,),
        edge_filter=EdgeFilter(predicate_kinds=frozenset({PredicateKind.HIERARCHY})),
        max_depth=2,
        max_nodes=20,
        trace=True,
    )

    assert subgraph.nodes == frozenset({1, 2, 3, 4, 5})
    assert _edge_keys(subgraph) == {
        (1, "is a", 2),
        (1, "is a", 3),
        (2, "is a", 4),
        (3, "is a", 5),
    }
    assert trace is not None
    assert trace.terminated_reason is None


def test_capped_traversal_respects_node_depth_and_edge_closure() -> None:
    adjacency = {1: (2, 3), 2: (4, 5), 3: (6,), 4: (7,), 5: (), 6: ()}
    kg = FakeKnowledgeGraph(adjacency)

    subgraph, trace = traverse(
        cast(KnowledgeGraph, kg),
        seeds=(1,),
        edge_filter=EdgeFilter(predicate_kinds=frozenset({PredicateKind.HIERARCHY})),
        max_depth=2,
        max_nodes=4,
        trace=True,
    )

    assert subgraph.nodes == frozenset({1, 2, 3, 4})
    assert 1 in subgraph.nodes
    distances = _distances(adjacency, 1)
    assert all(distances[node] <= 2 for node in subgraph.nodes)
    assert all(
        edge.subject_id in subgraph.nodes and edge.object_id in subgraph.nodes
        for edge in subgraph.edges
    )
    assert trace is not None
    assert trace.terminated_reason == "max_nodes"


def test_fanout_traversal_stops_and_closes_edge_iterator_at_node_budget() -> None:
    fanout = 5_000
    kg = FakeKnowledgeGraph({1: tuple(range(2, fanout + 2))})

    subgraph, _ = traverse(
        cast(KnowledgeGraph, kg),
        seeds=(1,),
        edge_filter=EdgeFilter(predicate_kinds=frozenset({PredicateKind.HIERARCHY})),
        max_depth=2,
        max_nodes=32,
        trace=False,
    )

    assert len(subgraph.nodes) == 32
    assert kg.materialized_edges <= 31
    assert kg.closed_edge_iterators == 1


def test_capped_traversal_omits_edges_from_unexpanded_nodes() -> None:
    kg = FakeKnowledgeGraph({1: (2, 3), 2: (3,)})

    subgraph, _ = traverse(
        cast(KnowledgeGraph, kg),
        seeds=(1,),
        edge_filter=EdgeFilter(predicate_kinds=frozenset({PredicateKind.HIERARCHY})),
        max_depth=2,
        max_nodes=3,
        trace=False,
    )

    assert subgraph.nodes == frozenset({1, 2, 3})
    assert _edge_keys(subgraph) == {(1, "is a", 2), (1, "is a", 3)}


def test_capped_traversal_orders_edges_before_filling_node_budget() -> None:
    kg = FakeKnowledgeGraph({1: (9, 2, 5)})

    subgraph, _ = traverse(
        cast(KnowledgeGraph, kg),
        seeds=(1,),
        edge_filter=EdgeFilter(predicate_kinds=frozenset({PredicateKind.HIERARCHY})),
        max_depth=2,
        max_nodes=2,
        trace=False,
    )

    assert subgraph.nodes == frozenset({1, 2})


def test_seeds_at_node_budget_are_not_expanded() -> None:
    kg = FakeKnowledgeGraph({1: (2,), 2: (1,)})

    subgraph, _ = traverse(
        cast(KnowledgeGraph, kg),
        seeds=(1, 2),
        edge_filter=EdgeFilter(predicate_kinds=frozenset({PredicateKind.HIERARCHY})),
        max_depth=2,
        max_nodes=2,
        trace=False,
    )

    assert subgraph.nodes == frozenset({1, 2})
    assert not subgraph.edges
    assert kg.materialized_edges == 0


def test_q_edges_ordering_is_opt_in_and_total() -> None:
    ordered_sql = str(
        q_edges((1,), None, EdgeFilter(), detailed=True, ordered=True).compile(
            dialect=postgresql.dialect()
        )
    )
    unordered_sql = str(
        q_edges((1,), None, EdgeFilter(), detailed=True, ordered=False).compile(
            dialect=postgresql.dialect()
        )
    )

    ordered_columns = [
        column.rsplit(".", 1)[-1] for column in ordered_sql.partition("ORDER BY ")[2].split(", ")
    ]
    assert ordered_columns == ["concept_id_1", "concept_id_2", "relationship_id"]
    assert "ORDER BY" not in unordered_sql
