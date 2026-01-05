from __future__ import annotations

from html import escape

from omop_graph.graph.nodes import ConceptView
from omop_graph.graph.traverse import Subgraph, GraphTrace
from omop_graph.graph.paths import GraphPath
from omop_graph.graph.scoring import PathExplanation


def concept_card(c: ConceptView) -> str:
    status = "✅" if c.invalid_reason is None else "❌"
    return f"""
    <div style="border:1px solid #ddd; padding:8px; border-radius:6px;">
        <b>{escape(c.concept_name)}</b> {status}<br/>
        <code>{c.vocabulary_id}:{c.concept_code}</code><br/>
        <small>
            Domain: {c.domain_id} · Class: {c.concept_class_id}
        </small>
    </div>
    """

def concept_card_compact(c: ConceptView) -> str:
    status = "✅" if c.invalid_reason is None else "❌"
    return f"""
    <div style="
        border:1px solid #ddd;
        border-radius:6px;
        padding:6px 8px;
        background:#fafafa;
        display:inline-block;
        max-width:420px;
    ">
        <div style="font-weight:600;">
            {escape(c.concept_name)} {status}
        </div>
        <div style="font-size:0.85em; color:#555;">
            <code>{c.vocabulary_id}:{c.concept_code}</code>
            · {c.domain_id}
            · {c.concept_class_id}
        </div>
    </div>
    """


def subgraph_html(kg, sg: Subgraph) -> str:
    nodes = [kg.concept_view(cid) for cid in sg.nodes]
    node_html = "".join(concept_card(n) for n in nodes[:20])

    return f"""
    <div>
        <h4>Subgraph</h4>
        <p>Nodes: {len(sg.nodes)} · Edges: {len(sg.edges)}</p>
        <div style="display:grid; grid-template-columns:repeat(2,1fr); gap:8px;">
            {node_html}
        </div>
    </div>
    """


def trace_html_with_cards(kg, trace: GraphTrace) -> str:
    blocks: list[str] = []

    for step in trace.steps:
        c = kg.concept_view(step.node)
        blocks.append(f"""
        <div style="margin-bottom:16px;">
            <div style="margin-bottom:6px; color:#666;">
                [depth {step.depth}]
            </div>
            {concept_card_compact(c)}
        """)

        by_pred: dict[str, list] = {}
        for e in step.expanded_edges:
            by_pred.setdefault(e.predicate_id, []).append(e)

        for pid, edges in by_pred.items():
            pname = escape(kg.predicate_name(pid))
            blocks.append(f"""
            <div style="margin-left:20px; margin-top:6px;">
                <span style="color:#555;">└─ {pname}</span>
            </div>
            """)

            MAX = 5
            for e in edges[:MAX]:
                obj = kg.concept_view(e.object_id)
                blocks.append(f"""
                <div style="margin-left:40px;">
                    → {escape(obj.concept_name)}
                </div>
                """)

            if len(edges) > MAX:
                blocks.append(f"""
                <div style="margin-left:40px; color:#888;">
                    … {len(edges) - MAX} more
                </div>
                """)

        blocks.append("</div>")

    if trace.terminated_reason:
        blocks.append(
            f"<div style='color:#a00;'>[terminated: {escape(trace.terminated_reason)}]</div>"
        )

    return f"""
    <div style="font-family: system-ui, sans-serif; line-height:1.4;">
        {''.join(blocks)}
    </div>
    """

def path_html(kg, path: GraphPath) -> str:
    lines = []
    for step in path.steps:
        s = kg.concept_view(step.subject)
        o = kg.concept_view(step.object)
        lines.append(
            f"{escape(s.concept_name)} "
            f"<b>--[{escape(step.predicate)}]--></b> "
            f"{escape(o.concept_name)}"
        )
    return "<br/>".join(lines)


def explained_path_html(kg, explanation: PathExplanation) -> str:
    rows = []
    for s in explanation.steps:
        subj = kg.concept_view(s.step.subject)
        obj = kg.concept_view(s.step.object)
        rows.append(f"""
        <tr>
            <td>{escape(subj.concept_name)}</td>
            <td>{escape(s.step.predicate)}</td>
            <td>{escape(obj.concept_name)}</td>
            <td>{s.predicate_kind.name}</td>
            <td>{escape(s.reason)}</td>
        </tr>
        """)

    return f"""
    <div>
        <h4>Path explanation (score={explanation.profile.path_rank():.2f})</h4>
        <table border="1" cellpadding="4" cellspacing="0">
            <tr>
                <th>From</th>
                <th>Predicate</th>
                <th>To</th>
                <th>Kind</th>
                <th>Δ</th>
                <th>Reason</th>
            </tr>
            {''.join(rows)}
        </table>
    </div>
    """
