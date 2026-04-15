from __future__ import annotations

import os
from typing import Any, Optional

import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from omop_graph.extensions.omop_alchemy import ClassIDEnum
from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.graph.nodes import LabelMatchKind
from omop_graph.graph.paths import find_shortest_paths
from omop_graph.mcp.servers.base_server import MCPServer


def _session_factory_from_env() -> sessionmaker:
    db_url = os.getenv("OMOP_DATABASE_URL")
    if not db_url:
        raise RuntimeError("OMOP_DATABASE_URL environment variable not set.")

    engine = sa.create_engine(db_url, future=True, echo=False)
    return sessionmaker(bind=engine, future=True)


def build_kg() -> KnowledgeGraph:
    return KnowledgeGraph(session_factory=_session_factory_from_env())


def _parse_predicate_classes(values: Optional[list[str]]) -> Optional[frozenset[ClassIDEnum]]:
    if not values:
        return None

    out: set[ClassIDEnum] = set()
    for value in values:
        try:
            out.add(ClassIDEnum(value))
        except ValueError as exc:
            allowed = ", ".join(member.value for member in ClassIDEnum)
            raise ValueError(f"Unknown predicate class '{value}'. Allowed: {allowed}") from exc
    return frozenset(out)


class KGServer(MCPServer):
    """MCP server for omop-graph knowledge graph traversal."""

    def __init__(self, kg: Optional[KnowledgeGraph] = None) -> None:
        """Initialize the KG server.

        Args:
            kg: KnowledgeGraph instance. If None, builds from OMOP_DATABASE_URL.
        """
        self.graph = kg or build_kg()

    @property
    def name(self) -> str:
        return "omop-graph-mcp-kg"

    def register_tools(self, server: Any) -> None:
        """Register KG exploration tools on the FastMCP server."""
        graph = self.graph

        @server.tool()
        def concept_view(concept_id: int) -> dict:
            concept = graph.concept_view(concept_id)
            return {
                "concept_id": concept.concept_id,
                "concept_name": concept.concept_name,
                "vocabulary_id": concept.vocabulary_id,
                "domain_id": concept.domain_id,
                "standard_concept": bool(concept.standard_concept),
                "invalid_reason": concept.invalid_reason,
            }

        @server.tool()
        def concept_lookup(
            label: str,
            match_kind: str = "exact",
            max_candidates: int = 25,
        ) -> list[dict]:
            mk = LabelMatchKind(match_kind)
            group = graph.concept_lookup(label=label, match_kind=mk)
            matches = list(group)[: max(1, min(max_candidates, 200))]
            return [
                {
                    "concept_id": match.concept_id,
                    "matched_label": match.matched_label,
                    "match_kind": match.match_kind.name,
                    "is_standard": match.is_standard,
                    "is_active": match.is_active,
                }
                for match in matches
            ]

        @server.tool()
        def shortest_paths(
            source_concept_id: int,
            target_concept_id: int,
            predicate_classes: Optional[list[str]] = None,
            max_depth: int = 6,
            max_paths: int = 10,
        ) -> dict:
            classes = _parse_predicate_classes(predicate_classes)
            paths, trace = find_shortest_paths(
                graph,
                source=source_concept_id,
                target=target_concept_id,
                predicate_kinds=classes,
                max_depth=max(1, min(max_depth, 12)),
                max_paths=max(1, min(max_paths, 50)),
                traced=True,
            )

            payload_paths = []
            for path in paths:
                payload_paths.append(
                    {
                        "length": len(path),
                        "nodes": list(path.nodes()),
                        "steps": [
                            {
                                "source": step.subject.concept_id,
                                "predicate_id": step.predicate,
                                "target": step.object.concept_id,
                            }
                            for step in path.steps
                        ],
                    }
                )

            return {
                "num_paths": len(payload_paths),
                "paths": payload_paths,
                "trace_terminated_reason": trace.terminated_reason if trace else None,
            }

        @server.tool()
        def explore_connections(
            seed_concept_id: int,
            predicate_classes: Optional[list[str]] = None,
            max_depth: int = 4,
            max_edges_per_expand: int = 100,
            max_total_expansions: int = 500,
        ) -> dict:
            classes = _parse_predicate_classes(predicate_classes)
            result = graph.explore_connections(
                seed_concept_id=seed_concept_id,
                predicate_kinds=classes,
                max_depth=max(1, min(max_depth, 8)),
                max_edges_per_expand=max(1, min(max_edges_per_expand, 500)),
                max_total_expansions=max(1, min(max_total_expansions, 5000)),
            )

            return {
                "visited_count": len(result.visited),
                "visited": list(result.visited),
                "truncated": result.truncated,
                "steps": [
                    {
                        "source": step.subject.concept_id,
                        "predicate_id": step.predicate,
                        "target": step.object.concept_id,
                        "depth": step.depth,
                    }
                    for step in result.steps
                ],
            }


def main() -> None:
    """Run the KG MCP server as a standalone process."""
    server = KGServer()
    app = server.build()
    app.run()


if __name__ == "__main__":
    main()
