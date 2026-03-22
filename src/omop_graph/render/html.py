from __future__ import annotations

from collections import Counter
from html import escape

from omop_graph.graph.nodes import ConceptView
from omop_graph.graph.scoring import StandardConceptWithScore, _textual_similarity_score
from omop_graph.graph.traverse import Subgraph, GraphTrace
from omop_graph.graph.paths import GraphPath, PathExplanation, find_shortest_paths
from omop_graph.extensions.omop_alchemy import ClassIDEnum
from omop_graph.reasoning.resolvers import CandidateHit


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
    if not path.steps:
        return "<div>[no traversal needed]</div>"
    lines = []
    for step in path.steps:
        s = kg.concept_view(step.subject.concept_id)
        o = kg.concept_view(step.object.concept_id)
        lines.append(
            f"{escape(s.concept_name)} "
            f"<b>--[{escape(step.predicate)}]--></b> "
            f"{escape(o.concept_name)}"
        )
    return "<br/>".join(lines)


def explained_path_html(kg, explanation: PathExplanation) -> str:
    rows = []
    for s in explanation.steps:
        subj = kg.concept_view(s.step.subject.concept_id)
        obj = kg.concept_view(s.step.object.concept_id)
        rows.append(f"""
        <tr>
            <td>{escape(subj.concept_name)}</td>
            <td>{escape(s.step.predicate)}</td>
            <td>{escape(obj.concept_name)}</td>
            <td>{s.predicate_kind.name}</td>
            <td>{escape(str(s.traversal_depth) if s.traversal_depth is not None else "-")}</td>
            <td>{escape(s.reason)}</td>
        </tr>
        """)

    return f"""
    <div>
        <h4>Path explanation for {escape(explanation.profile.concept_name)}</h4>
        <table border="1" cellpadding="4" cellspacing="0">
            <tr>
                <th>From</th>
                <th>Predicate</th>
                <th>To</th>
                <th>Kind</th>
                <th>Trace depth</th>
                <th>Reason</th>
            </tr>
            {''.join(rows)}
        </table>
    </div>
    """


def candidate_hits_html(
    kg,
    hits: list[CandidateHit],
    *,
    title: str = "Candidate Hits",
) -> str:
    if not hits:
        return f"""
        <div style="font-family:system-ui,sans-serif;background:#f8fafc;color:#111827;
                    border:1px solid #cbd5e1;border-radius:14px;padding:18px 20px;">
            <h3>{escape(title)}</h3>
            <div>No resolver hits.</div>
        </div>
        """

    resolver_counts = Counter(hit.resolver_confidence.name for hit in hits)
    count_badges = "".join(
        f"<span style='display:inline-block;padding:4px 8px;border-radius:999px;"
        f"background:#dbeafe;color:#0f172a;border:1px solid #93c5fd;margin-right:6px;margin-bottom:6px;'>"
        f"{escape(resolver)}: {count}</span>"
        for resolver, count in resolver_counts.items()
    )

    rows = []
    for idx, hit in enumerate(hits, start=1):
        concept = kg.concept_view(hit.concept_id)
        rows.append(f"""
        <tr>
            <td>{idx}</td>
            <td><code>{hit.concept_id}</code></td>
            <td>{escape(concept.concept_name)}</td>
            <td>{escape(concept.domain_id)}</td>
            <td>{escape(concept.vocabulary_id)}</td>
            <td>{escape(hit.resolver_confidence.name)}</td>
            <td><code>{escape(hit.matched_label)}</code></td>
        </tr>
        """)

    return f"""
    <div style="font-family:system-ui,sans-serif;line-height:1.45;color:#111827;
                background:#f8fafc;border:1px solid #cbd5e1;border-radius:14px;
                padding:18px 20px;box-shadow:0 1px 2px rgba(15,23,42,0.06);">
        <h3 style="margin-bottom:10px;">{escape(title)}</h3>
        <div style="margin-bottom:12px;">{count_badges}</div>
        <table style="border-collapse:collapse;min-width:900px;background:#ffffff;">
            <thead>
                <tr style="background:#e5e7eb;color:#111827;">
                    <th style="text-align:left;padding:8px;border:1px solid #9ca3af;">#</th>
                    <th style="text-align:left;padding:8px;border:1px solid #9ca3af;">Concept ID</th>
                    <th style="text-align:left;padding:8px;border:1px solid #9ca3af;">Concept</th>
                    <th style="text-align:left;padding:8px;border:1px solid #9ca3af;">Domain</th>
                    <th style="text-align:left;padding:8px;border:1px solid #9ca3af;">Vocabulary</th>
                    <th style="text-align:left;padding:8px;border:1px solid #9ca3af;">Resolver</th>
                    <th style="text-align:left;padding:8px;border:1px solid #9ca3af;">Matched label</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
    </div>
    """


def grounding_review_html(
    kg,
    query_text: str,
    results: list[StandardConceptWithScore],
    *,
    max_near_winners: int = 3,
    max_also_rans: int = 5,
    near_winner_delta: float = 0.05,
    path_max_depth: int = 4,
) -> str:
    if not results:
        return f"""
        <div style="font-family:system-ui,sans-serif;background:#f8fafc;color:#111827;
                    border:1px solid #cbd5e1;border-radius:14px;padding:18px 20px;">
            <h3>Grounding review</h3>
            <div><b>Query:</b> {escape(query_text)}</div>
            <div style="margin-top:8px;">No grounded candidates passed the constraints.</div>
        </div>
        """

    winner = results[0]
    near_winners = [
        result for result in results[1:]
        if winner.total_score - result.total_score <= near_winner_delta
    ][:max_near_winners]
    also_rans = [
        result for result in results[1:]
        if result not in near_winners
    ][:max_also_rans]

    sections = [
        _grounding_result_card_html(
            kg,
            query_text,
            winner,
            rank=1,
            score_gap=0.0,
            tier_label="Winner",
            path_max_depth=path_max_depth,
            highlight=True,
        )
    ]

    if near_winners:
        sections.append("<h4 style='margin:20px 0 10px;color:#0f172a;'>Close contenders</h4>")
        sections.append(
            _grounding_summary_table_html(
                kg,
                query_text,
                winner,
                near_winners,
                start_rank=2,
                path_max_depth=path_max_depth,
            )
        )

    if also_rans:
        sections.append("<h4 style='margin:20px 0 10px;color:#0f172a;'>Remaining candidates</h4>")
        sections.append(
            _grounding_summary_table_html(
                kg,
                query_text,
                winner,
                also_rans,
                start_rank=2 + len(near_winners),
                path_max_depth=path_max_depth,
            )
        )

    return f"""
    <div style="font-family:system-ui,sans-serif;line-height:1.45;color:#111827;
                background:#f8fafc;border:1px solid #cbd5e1;border-radius:14px;
                padding:18px 20px;box-shadow:0 1px 2px rgba(15,23,42,0.06);">
        <h3 style="margin-bottom:8px;">Grounding review</h3>
        <div style="margin-bottom:8px;"><b>Query:</b> {escape(query_text)}</div>
        <div style="margin-bottom:16px;color:#374151;">
            Score = relevance - parsimony penalty + broadness bonus
        </div>
        {''.join(sections)}
    </div>
    """


def _grounding_result_card_html(
    kg,
    query_text: str,
    result: StandardConceptWithScore,
    *,
    rank: int,
    score_gap: float,
    tier_label: str,
    path_max_depth: int,
    highlight: bool = False,
) -> str:
    textual_similarity = _textual_similarity_score(query_text, result.matched_label)
    path_string = _candidate_anchor_path_html(kg, result, path_max_depth=path_max_depth)
    border = "#1d4ed8" if highlight else "#94a3b8"
    background = "#dbeafe" if highlight else "#ffffff"

    return f"""
    <div style="border:2px solid {border};background:{background};border-radius:12px;
                padding:14px 16px;margin-bottom:12px;">
        <div style="display:flex;justify-content:space-between;gap:12px;align-items:flex-start;">
            <div>
                <div style="font-size:0.82rem;color:#334155;text-transform:uppercase;letter-spacing:0.04em;">
                    {escape(tier_label)} · rank {rank}
                </div>
                <div style="font-size:1.05rem;font-weight:700;margin-top:2px;">
                    {escape(result.concept_name)} <span style="font-weight:500;color:#334155;">({result.concept_id})</span>
                </div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:1.05rem;font-weight:700;">total {result.total_score:.4f}</div>
                <div style="color:#334155;">gap to winner {score_gap:.4f}</div>
            </div>
        </div>
        <div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin-top:12px;">
            <div style="padding:8px;background:#f8fafc;border:1px solid #cbd5e1;border-radius:8px;">relevance<br/><b>{result.relevance:.4f}</b></div>
            <div style="padding:8px;background:#f8fafc;border:1px solid #cbd5e1;border-radius:8px;">parsimony penalty<br/><b>{result.parsimony_penalty:.4f}</b></div>
            <div style="padding:8px;background:#f8fafc;border:1px solid #cbd5e1;border-radius:8px;">broadness bonus<br/><b>{result.broadness_bonus:.4f}</b></div>
            <div style="padding:8px;background:#f8fafc;border:1px solid #cbd5e1;border-radius:8px;">embedding score<br/><b>{escape(str(result.embedding_score))}</b></div>
            <div style="padding:8px;background:#f8fafc;border:1px solid #cbd5e1;border-radius:8px;">textual similarity<br/><b>{textual_similarity:.4f}</b></div>
            <div style="padding:8px;background:#f8fafc;border:1px solid #cbd5e1;border-radius:8px;">hierarchy separation<br/><b>{result.separation}</b></div>
        </div>
        <div style="margin-top:12px;">
            <b>Candidate provenance:</b>
            <div style="margin-top:4px;">
                original candidate <code>{result.original_id}</code> = {escape(result.original_name)}
            </div>
            <div style="margin-top:2px;">
                resolver {escape(result.resolver_confidence.name)} via matched label <code>{escape(result.matched_label)}</code>
            </div>
        </div>
        <div style="margin-top:12px;">
            <b>Anchor path from candidate to grounded concept:</b>
            <div style="margin-top:6px;padding:10px;background:#0f172a;color:#e2e8f0;border-radius:8px;
                        font-family:ui-monospace,SFMono-Regular,Menlo,monospace;white-space:pre-wrap;">{path_string}</div>
        </div>
    </div>
    """


def _grounding_summary_table_html(
    kg,
    query_text: str,
    winner: StandardConceptWithScore,
    results: list[StandardConceptWithScore],
    *,
    start_rank: int,
    path_max_depth: int,
) -> str:
    rows = []
    for idx, result in enumerate(results, start=start_rank):
        textual_similarity = _textual_similarity_score(query_text, result.matched_label)
        rows.append(f"""
        <tr>
            <td style="padding:8px;border:1px solid #9ca3af;">{idx}</td>
            <td style="padding:8px;border:1px solid #9ca3af;"><code>{result.concept_id}</code></td>
            <td style="padding:8px;border:1px solid #9ca3af;">{escape(result.concept_name)}</td>
            <td style="padding:8px;border:1px solid #9ca3af;">{result.total_score:.4f}</td>
            <td style="padding:8px;border:1px solid #9ca3af;">{winner.total_score - result.total_score:.4f}</td>
            <td style="padding:8px;border:1px solid #9ca3af;">{escape(str(result.embedding_score))}</td>
            <td style="padding:8px;border:1px solid #9ca3af;">{textual_similarity:.4f}</td>
            <td style="padding:8px;border:1px solid #9ca3af;">{result.separation}</td>
            <td style="padding:8px;border:1px solid #9ca3af;">{escape(result.resolver_confidence.name)}</td>
            <td style="padding:8px;border:1px solid #9ca3af;">{escape(result.matched_label)}</td>
            <td style="padding:8px;border:1px solid #9ca3af;">{escape(_candidate_anchor_path_summary_text(kg, result, path_max_depth=path_max_depth))}</td>
        </tr>
        """)

    return f"""
    <table style="border-collapse:collapse;min-width:1200px;background:#ffffff;">
        <thead>
            <tr style="background:#e5e7eb;color:#111827;">
                <th style="text-align:left;padding:8px;border:1px solid #9ca3af;">#</th>
                <th style="text-align:left;padding:8px;border:1px solid #9ca3af;">Concept ID</th>
                <th style="text-align:left;padding:8px;border:1px solid #9ca3af;">Concept</th>
                <th style="text-align:left;padding:8px;border:1px solid #9ca3af;">Total</th>
                <th style="text-align:left;padding:8px;border:1px solid #9ca3af;">Gap</th>
                <th style="text-align:left;padding:8px;border:1px solid #9ca3af;">Embedding</th>
                <th style="text-align:left;padding:8px;border:1px solid #9ca3af;">Text</th>
                <th style="text-align:left;padding:8px;border:1px solid #9ca3af;">Sep</th>
                <th style="text-align:left;padding:8px;border:1px solid #9ca3af;">Resolver</th>
                <th style="text-align:left;padding:8px;border:1px solid #9ca3af;">Matched label</th>
                <th style="text-align:left;padding:8px;border:1px solid #9ca3af;">Candidate → winner path</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    """


def _candidate_anchor_path_html(
    kg,
    result: StandardConceptWithScore,
    *,
    path_max_depth: int,
) -> str:
    if result.original_id == result.concept_id:
        return "[candidate already equals winning concept]"

    paths, _ = find_shortest_paths(
        kg,
        source=result.original_id,
        target=result.concept_id,
        predicate_kinds=frozenset({ClassIDEnum.IDENTITY}),
        max_depth=path_max_depth,
        max_paths=1,
        traced=False,
    )
    if not paths:
        return "[no identity-path reconstruction found]"
    return escape(_path_text_block(kg, paths[0]))


def _candidate_anchor_path_summary_text(
    kg,
    result: StandardConceptWithScore,
    *,
    path_max_depth: int,
) -> str:
    if result.original_id == result.concept_id:
        return "candidate==winner"

    paths, _ = find_shortest_paths(
        kg,
        source=result.original_id,
        target=result.concept_id,
        predicate_kinds=frozenset({ClassIDEnum.IDENTITY}),
        max_depth=path_max_depth,
        max_paths=1,
        traced=False,
    )
    if not paths or not paths[0].steps:
        return "no identity path"

    return " -> ".join(
        kg.concept_view(step.object.concept_id).concept_name for step in paths[0].steps
    )


def _path_text_block(kg, path: GraphPath) -> str:
    if not path.steps:
        return "[no traversal needed]"
    return "\n".join(
        (
            f"{kg.concept_view(step.subject.concept_id).concept_name} "
            f"--[{step.predicate}]--> "
            f"{kg.concept_view(step.object.concept_id).concept_name}"
        )
        for step in path.steps
    )
