"""A graph given by its edges, for the tests of the algorithms that read a graph."""

from __future__ import annotations

from collections.abc import Iterable
from types import SimpleNamespace
from typing import cast

from omop_graph.graph.kg import KnowledgeGraph

DEFAULT_PREDICATE = "Is a"


class EdgeList:
    """
    A graph of the given edges that answers the calls a path search makes, and counts them.

    Parameters
    ----------
    edges : Iterable[tuple[int, int] | tuple[int, int, str]]
        The edges as subject, object and optionally the relationship, ``Is a`` when left out.
    """

    def __init__(self, edges: Iterable[tuple[int, int] | tuple[int, int, str]]) -> None:
        self.edge_list = tuple(edges)
        self.concepts = sorted({concept for edge in self.edge_list for concept in edge[:2]})
        self.read_from: list[int] = []

    def edges(self, concept_ids, direction, edge_filter=None):
        ids = {concept_ids} if isinstance(concept_ids, int) else set(concept_ids)
        self.read_from.extend(sorted(ids))
        return tuple(
            SimpleNamespace(
                subject_id=subject,
                object_id=object_,
                predicate_id=(predicate or (DEFAULT_PREDICATE,))[0],
            )
            for subject, object_, *predicate in self.edge_list
            if (subject if direction == "out" else object_) in ids
        )

    def concept_views(self, concept_ids, sort=True):
        return tuple(
            SimpleNamespace(concept_id=concept_id, standard_concept=concept_id % 2 == 1)
            for concept_id in concept_ids
        )

    @property
    def graph(self) -> KnowledgeGraph:
        """This list as the graph a search takes, which only needs the calls it answers."""
        return cast(KnowledgeGraph, self)

    def shortest_routes(self, source: int, target: int) -> list[tuple[int, ...]]:
        """Every shortest route, by enumerating the routes breadth first."""
        found: list[tuple[int, ...]] = []
        routes = [(source,)]
        while routes and not found:
            routes = [
                route + (edge[1],)
                for route in routes
                for edge in self.edge_list
                if edge[0] == route[-1] and edge[1] not in route
            ]
            found = [route for route in routes if route[-1] == target]
        return sorted(set(found))
