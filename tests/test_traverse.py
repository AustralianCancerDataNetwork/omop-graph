from __future__ import annotations

from contextlib import nullcontext
from datetime import date

from omop_graph.extensions.omop_alchemy import PredicateKind
from omop_graph.graph.edges import EdgeView
from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.graph.traverse import traverse


class FakeKnowledgeGraph:
    def __init__(self, adjacency: dict[int, tuple[int, ...]]) -> None:
        self.adjacency = adjacency
        self.materialized_edges = 0
        self.closed_edge_iterators = 0
        self.stream_flags: list[bool] = []

    @staticmethod
    def session_factory():
        return nullcontext(None)

    def iter_edges(self, *, concept_ids: int, stream: bool = False, **_kwargs):
        self.stream_flags.append(stream)
        try:
            for object_id in self.adjacency.get(concept_ids, ()):
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
    return {
        (edge.subject_id, edge.predicate_id, edge.object_id)
        for edge in subgraph.edges
    }


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
        kg,
        seeds=(1,),
        predicate_kinds={PredicateKind.HIERARCHY},
        max_depth=2,
        on=None,
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
        kg,
        seeds=(1,),
        predicate_kinds={PredicateKind.HIERARCHY},
        max_depth=2,
        on=None,
        max_nodes=4,
        trace=True,
    )

    assert len(subgraph.nodes) == 4
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
        kg,
        seeds=(1,),
        predicate_kinds={PredicateKind.HIERARCHY},
        max_depth=2,
        on=None,
        max_nodes=32,
        trace=False,
    )

    assert len(subgraph.nodes) == 32
    assert kg.materialized_edges <= 31
    assert kg.closed_edge_iterators == 1
    assert kg.stream_flags == [True]


def test_iter_edges_streaming_is_opt_in() -> None:
    class RecordingSession:
        def __init__(self) -> None:
            self.execution_options: list[object] = []

        def execute(self, _statement, **kwargs):
            self.execution_options.append(kwargs.get("execution_options", {}))
            return iter(())

    kg = KnowledgeGraph.__new__(KnowledgeGraph)
    kg._relationship_mapping = {}
    session = RecordingSession()

    list(kg.iter_edges(session=session, concept_ids=1, direction="out"))
    list(kg.iter_edges(session=session, concept_ids=1, direction="out", stream=True))

    assert session.execution_options == [{}, {"yield_per": 1000}]
