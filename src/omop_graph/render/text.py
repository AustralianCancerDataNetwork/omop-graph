from __future__ import annotations

from collections import Counter

from omop_graph.extensions.omop_alchemy import ClassIDEnum
from omop_graph.graph.paths import GraphPath, PathExplanation, find_shortest_paths
from omop_graph.graph.scoring import StandardConceptWithScore, _textual_similarity_score
from omop_graph.graph.traverse import Subgraph, GraphTrace
from omop_graph.reasoning.resolvers import CandidateHit


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
    if not path.steps:
        return "[no traversal needed]"
    parts = []
    for step in path.steps:
        s = kg.concept_view(step.subject.concept_id)
        o = kg.concept_view(step.object.concept_id)
        parts.append(f"{s.concept_name} --[{step.predicate}]--> {o.concept_name}")
    return "\n".join(parts)


def explained_path_text(kg, explanation: PathExplanation) -> str:
    lines = [f"Explained path to {explanation.profile.concept_name}", "Steps:"]

    for s in explanation.steps:
        subj = kg.concept_view(s.step.subject.concept_id)
        obj = kg.concept_view(s.step.object.concept_id)
        lines.append(
            f"- {subj.concept_name} --[{s.step.predicate}]--> {obj.concept_name} "
            f"({s.predicate_kind.name}) "
            f"{s.reason}"
        )

    return "\n".join(lines)


def candidate_hits_text(
    kg,
    hits: list[CandidateHit],
    *,
    title: str = "Candidate Hits",
) -> str:
    lines = [title, "-" * len(title)]
    if not hits:
        lines.append("No resolver hits.")
        return "\n".join(lines)

    resolver_counts = Counter(hit.resolver_confidence.name for hit in hits)
    lines.append(
        "Resolver mix: "
        + ", ".join(
            f"{resolver}={count}" for resolver, count in resolver_counts.items()
        )
    )
    lines.append("")

    for idx, hit in enumerate(hits, start=1):
        concept = kg.concept_view(hit.concept_id)
        lines.append(
            f"#{idx} {hit.concept_id} | {concept.concept_name} | "
            f"{concept.domain_id} | {concept.vocabulary_id}"
        )
        lines.append(
            f"  resolver={hit.resolver_confidence.name} | matched_label={hit.matched_label!r}"
        )

    return "\n".join(lines)


def grounding_review_text(
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
        return f"Query: {query_text}\nNo grounded candidates passed the constraints."

    winner = results[0]
    near_winners = [
        result for result in results[1:]
        if winner.total_score - result.total_score <= near_winner_delta
    ][:max_near_winners]
    also_rans = [
        result for result in results[1:]
        if result not in near_winners
    ][:max_also_rans]

    lines = [
        f"Query: {query_text}",
        f"Grounded candidates: {len(results)}",
        "",
        "Winner",
        "------",
    ]
    lines.extend(
        _format_grounded_result_text(
            kg,
            query_text,
            winner,
            rank=1,
            score_gap=0.0,
            path_max_depth=path_max_depth,
        )
    )

    if near_winners:
        lines.extend(["", "Close Contenders", "----------------"])
        lines.extend(
            _format_grounding_summary_table_text(
                kg,
                query_text,
                winner,
                near_winners,
                start_rank=2,
                path_max_depth=path_max_depth,
            )
        )

    if also_rans:
        lines.extend(["", "Remaining Candidates", "--------------------"])
        lines.extend(
            _format_grounding_summary_table_text(
                kg,
                query_text,
                winner,
                also_rans,
                start_rank=2 + len(near_winners),
                path_max_depth=path_max_depth,
            )
        )

    return "\n".join(lines)


def _format_grounded_result_text(
    kg,
    query_text: str,
    result: StandardConceptWithScore,
    *,
    rank: int,
    score_gap: float,
    path_max_depth: int,
) -> list[str]:
    textual_similarity = _textual_similarity_score(query_text, result.matched_label)
    path_string = _candidate_anchor_path_text(
        kg,
        result,
        path_max_depth=path_max_depth,
    )
    return [
        f"#{rank} {result.concept_id} | {result.concept_name}",
        (
            f"  total={result.total_score:.4f} | relevance={result.relevance:.4f} | "
            f"parsimony_penalty={result.parsimony_penalty:.4f} | "
            f"broadness_bonus={result.broadness_bonus:.4f} | gap_to_winner={score_gap:.4f}"
        ),
        (
            f"  embedding_score={result.embedding_score} | textual_similarity={textual_similarity:.4f} | "
            f"resolver={result.resolver_confidence.name} | matched_label={result.matched_label!r}"
        ),
        (
            f"  original_candidate={result.original_id}:{result.original_name} | "
            f"hierarchy_separation={result.separation}"
        ),
        "  anchor_path:",
        f"    {path_string.replace(chr(10), chr(10) + '    ')}",
    ]


def _candidate_anchor_path_text(
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
    return path_text(kg, paths[0])


def _format_grounding_summary_table_text(
    kg,
    query_text: str,
    winner: StandardConceptWithScore,
    results: list[StandardConceptWithScore],
    *,
    start_rank: int,
    path_max_depth: int,
) -> list[str]:
    header = (
        "rank  concept_id  concept_name                          total    gap      "
        "embed    text     sep  resolver          path"
    )
    divider = (
        "----  ----------  -----------------------------------  -------  -------  "
        "-------  -------  ---  ----------------  ------------------------------"
    )
    lines = [header, divider]

    for idx, result in enumerate(results, start=start_rank):
        textual_similarity = _textual_similarity_score(query_text, result.matched_label)
        path_summary = _candidate_anchor_path_summary_text(
            kg,
            result,
            path_max_depth=path_max_depth,
        )
        embedding_score = (
            f"{result.embedding_score:.4f}"
            if isinstance(result.embedding_score, float)
            else str(result.embedding_score)
        )
        lines.append(
            f"{idx:<4}  "
            f"{result.concept_id:<10}  "
            f"{result.concept_name[:35]:<35}  "
            f"{result.total_score:>7.4f}  "
            f"{(winner.total_score - result.total_score):>7.4f}  "
            f"{embedding_score[:7]:>7}  "
            f"{textual_similarity:>7.4f}  "
            f"{result.separation:>3}  "
            f"{result.resolver_confidence.name[:16]:<16}  "
            f"{path_summary}"
        )

    return lines


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
    )[:80]
