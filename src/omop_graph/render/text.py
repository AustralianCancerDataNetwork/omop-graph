from __future__ import annotations

from omop_graph.graph.paths import GraphPath
from omop_graph.graph.scoring import PathExplanation
from omop_graph.graph.traverse import Subgraph, GraphTrace


def subgraph_text(kg, sg: Subgraph) -> str:
    lines = [
        f"Subgraph:",
        f"  Nodes: {len(sg.nodes)}",
        f"  Edges: {len(sg.edges)}",
        "",
    ]

    for cid in sorted(sg.nodes):
        c = kg.concept_view(cid)
        lines.append(f"- {c.concept_name} ({c.vocabulary_id}:{c.concept_code})")

    return "\n".join(lines)


def trace_text(kg, trace: GraphTrace) -> str:
    lines: list[str] = []

    for step in trace.steps:
        c = kg.concept_view(step.node)
        lines.append(f"[depth {step.depth}] {c.concept_name}")

        by_pred = {}
        for e in step.expanded_edges:
            by_pred.setdefault(e.predicate_id, []).append(e)

        for pid, edges in by_pred.items():
            pname = kg.predicate_name(pid)
            lines.append(f"    └─ {pname}")

            for e in edges[:6]:
                obj = kg.concept_view(e.object_id)
                lines.append(f"        → {obj.concept_name}")

            if len(edges) > 6:
                lines.append(f"        … {len(edges) - 6} more")

    if trace.terminated_reason:
        lines.append(f"[terminated: {trace.terminated_reason}]")

    return "\n".join(lines)

def path_text(kg, path: GraphPath) -> str:
    parts = []
    for step in path.steps:
        s = kg.concept_view(step.subject)
        o = kg.concept_view(step.object)
        parts.append(f"{s.concept_name} --[{step.predicate}]--> {o.concept_name}")
    return "\n".join(parts)


def explained_path_text(kg, explanation: PathExplanation) -> str:
    lines = [f"Path score: {explanation.profile.path_rank():.2f}", "Steps:"]

    for s in explanation.steps:
        subj = kg.concept_view(s.step.subject)
        obj = kg.concept_view(s.step.object)
        lines.append(
            f"- {subj.concept_name} --[{s.step.predicate}]--> {obj.concept_name} "
            f"({s.predicate_kind.name}, "#Δ={s.score_delta:+.2f}) "
            f"{s.reason}"
        )

    return "\n".join(lines)
