from __future__ import annotations

from omop_graph.graph.traverse import Subgraph
from omop_graph.graph.paths import GraphPath

def subgraph_mermaid(kg, sg: Subgraph) -> str:
    lines = ["graph TD"]

    # nodes
    for cid in sg.nodes:
        c = kg.concept_view(cid)
        label = f"{c.concept_name}\\n{c.vocabulary_id}:{c.concept_code}"
        lines.append(f'  {cid}["{label}"]')

    # edges
    for e in sg.edges:
        pname = kg.predicate_name(e.predicate_id)
        lines.append(f"  {e.subject_id} -->|{pname}| {e.object_id}")

    return "\n".join(lines)


def path_mermaid(kg, path: GraphPath) -> str:
    lines = ["graph LR"]

    for step in path.steps:
        s = kg.concept_view(step.subject)
        o = kg.concept_view(step.object)
        lines.append(
            f'{step.subject}["{s.concept_name}"] -->|{step.predicate}| '
            f'{step.object}["{o.concept_name}"]'
        )

    return "\n".join(lines)
