"""
Trace a single grounding example step-by-step and optionally generate an SVG figure.

Commands
--------
trace  Run one case through every resolver, capture per-stage results, output JSON.
svg    Read a trace JSON and render a Whimsical-importable flowchart SVG.
"""

from __future__ import annotations

import json
import logging
from html import escape
from pathlib import Path
from typing import Annotated, Dict, List, Optional, Tuple

import typer

from omop_graph.config import OmopGraphConfig
from omop_graph.db.session import make_engine
from omop_graph.extensions.emb import get_embedding_writer_interface
from omop_graph.graph.constraints import SearchConstraintConcept
from omop_graph.graph.kg import KnowledgeGraph, KnowledgeGraphEmbeddingConfiguration
from omop_graph.graph.nodes import LabelMatchKind
from omop_graph.reasoning.grounding import GroundingConstraints, ground_term
from omop_graph.reasoning.resolvers.resolver_pipeline import ResolverPipeline
from omop_graph.reasoning.resolvers.resolvers import (
    EmbeddingResolver,
    ExactLabelResolver,
    ExactSynonymResolver,
    FullTextResolver,
    FullTextSynonymResolver,
    PartialLabelResolver,
    PartialSynonymResolver,
)

logger = logging.getLogger("omop_graph.trace_example")

app = typer.Typer(help="Trace a single OMOP grounding example and generate an SVG figure.")


@app.callback()
def _main(
    verbose: Annotated[
        int,
        typer.Option("--verbose", "-v", count=True, help="Increase log verbosity (-v INFO, -vv DEBUG). Must come before the subcommand name."),
    ] = 0,
) -> None:
    OmopGraphConfig.configure_logging(verbosity=verbose)

# Display name for each resolver (inferred from match_kind + synonym flag)
RESOLVER_DISPLAY_NAMES = {
    (LabelMatchKind.EXACT, False): "Exact",
    (LabelMatchKind.EXACT, True): "Exact Synonym",
    (LabelMatchKind.PARTIAL, False): "Partial",
    (LabelMatchKind.PARTIAL, True): "Partial Synonym",
    (LabelMatchKind.FTS, False): "Full Text",
    (LabelMatchKind.FTS, True): "Full Text Synonym",
    (LabelMatchKind.EMBEDDING, False): "Embedding",
}

ORDERED_RESOLVERS = [
    ExactLabelResolver(),
    ExactSynonymResolver(),
    PartialLabelResolver(),
    PartialSynonymResolver(),
    FullTextResolver(),
    FullTextSynonymResolver(),
    EmbeddingResolver(),
]

# Groups synonym + non-synonym resolvers for the SVG column view
RESOLVER_GROUPS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("Exact",     ("Exact", "Exact Synonym")),
    ("Partial",   ("Partial", "Partial Synonym")),
    ("Full Text", ("Full Text", "Full Text Synonym")),
    ("Embedding", ("Embedding",)),
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _build_kg(metric_type=None, embedding_model: Optional[str] = None) -> KnowledgeGraph:
    cdm_engine = make_engine()
    try:
        from omop_emb.config import MetricType, OmopEmbConfig
        from omop_emb.embeddings import EmbeddingClient

        resolved_metric = metric_type if metric_type is not None else MetricType.COSINE
        emb_cfg = OmopEmbConfig.get_config()
        client = EmbeddingClient(
            model=embedding_model or emb_cfg.embedding_model,
            api_base=emb_cfg.api_base,
            api_key=emb_cfg.api_key,
            provider_type=emb_cfg.provider_type,
        )
        canonical_model = client.provider.canonical_model_name(embedding_model or emb_cfg.embedding_model)
        emb_config = KnowledgeGraphEmbeddingConfiguration(
            client=client,
            model_name=canonical_model,
            metric_type=resolved_metric,
        )
        logger.info("Embedding config loaded (model=%s, metric=%s).", canonical_model, resolved_metric.value)
        return KnowledgeGraph(cdm_engine=cdm_engine, emb_config=emb_config)
    except Exception as exc:
        logger.warning("Error occurred while loading embedding config:\n%s.\nRunning without embedding.", exc)
        return KnowledgeGraph(cdm_engine=cdm_engine)


def _resolver_label(resolver) -> str:
    return RESOLVER_DISPLAY_NAMES.get((resolver.match_kind, resolver.synonym), type(resolver).__name__)


def _resolve_embedding_model(embedding_model: Optional[str]) -> str:
    """Resolve the canonical embedding model name (override or config default).

    Mirrors the canonicalisation `_build_kg` does, but without opening a DB
    connection — usable from `svg`, which never builds a KnowledgeGraph.
    """
    from omop_emb.config import OmopEmbConfig
    from omop_emb.embeddings import EmbeddingClient

    emb_cfg = OmopEmbConfig.get_config()
    raw_model = embedding_model or emb_cfg.embedding_model
    client = EmbeddingClient(
        model=raw_model,
        api_base=emb_cfg.api_base,
        api_key=emb_cfg.api_key,
        provider_type=emb_cfg.provider_type,
    )
    return client.provider.canonical_model_name(raw_model)


def _model_dir_name(embedding_model: Optional[str]) -> str:
    """Filesystem-safe directory name for a (canonical) embedding model name.

    `embedding_model` is None when no embedding is configured for this run
    (not an omop-emb availability issue — that's a hard requirement of this
    script — but a legitimate "ran without embedding" outcome).
    """
    if embedding_model is None:
        return "no_embedding"
    from omop_emb.model_registry.model_registry_manager import RegistryManager

    return RegistryManager.safe_model_name(embedding_model)


def _concept_info(kg: KnowledgeGraph, concept_id: int) -> Dict:
    try:
        cv = kg.concept_view(concept_id)
        return {
            "concept_id": cv.concept_id,
            "concept_name": cv.concept_name,
            "vocabulary_id": cv.vocabulary_id,
            "concept_code": cv.concept_code,
            "domain_id": cv.domain_id,
        }
    except Exception:
        return {"concept_id": concept_id, "concept_name": f"[concept {concept_id}]"}


# ─── Core trace logic ─────────────────────────────────────────────────────────

def _run_trace(
    kg: KnowledgeGraph,
    query: str,
    case_id: str,
    parent_ids: Tuple[int, ...],
    search_constraint: Optional[SearchConstraintConcept],
    expected_concept_id: Optional[int],
    top_n: int,
) -> Dict:
    # Stage 0: compute query embedding once (shared by individual resolvers + ground_term)
    query_embedding = None
    try:
        from omop_emb.embeddings import EmbeddingRole
        writer = get_embedding_writer_interface(kg)
        if writer is not None:
            query_embedding = writer.embed_texts(texts=(query,), embedding_role=EmbeddingRole.QUERY)
            logger.info("Query embedding computed (shape=%s).", query_embedding.shape)
        else:
            logger.info("No embedding writer available; EmbeddingResolver will be skipped.")
    except Exception as exc:
        logger.warning("Could not compute query embedding: %s", exc)

    # Stage 1: run every resolver individually to capture per-resolver attribution
    resolver_stages = []
    seen: set = set()
    concept_to_resolvers: Dict[int, List[str]] = {}

    for resolver in ORDERED_RESOLVERS:
        label = _resolver_label(resolver)
        hit_count = 0
        previously_unseen_hits = 0
        try:
            for hit in resolver.resolve(kg=kg, query=query, constraints=search_constraint,
                                        query_embedding=query_embedding):
                hit_count += 1
                if hit.concept_id not in seen:
                    previously_unseen_hits += 1
                    seen.add(hit.concept_id)
                concept_to_resolvers.setdefault(hit.concept_id, []).append(label)
        except Exception as exc:
            typer.echo(f"  {label}: skipped ({exc})", err=True)
        resolver_stages.append({"resolver": label, "hits": hit_count, "previously_unseen_hits": previously_unseen_hits})

    # Stage 2: full grounding pipeline (hierarchy validation + scoring + embedding ranking)
    pipeline = ResolverPipeline(resolvers=tuple(ORDERED_RESOLVERS))
    grounding_constraints = GroundingConstraints(
        parent_ids=parent_ids,
        search_constraint=search_constraint,
    )
    scored = ground_term(
        resolver_pipeline=pipeline,
        kg=kg,
        query=query,
        query_embedding=query_embedding,
        constraints=grounding_constraints,
    )

    # Reconstruct hierarchy_validation from scored vs resolver hits (for SVG compatibility)
    scored_by_id = {sc.concept_id: sc for sc in scored}
    hierarchy_validation = []
    for cid, resolvers_list in concept_to_resolvers.items():
        info = _concept_info(kg, cid)
        if cid in scored_by_id:
            sc = scored_by_id[cid]
            hierarchy_validation.append({
                "candidate_concept_id": cid,
                "candidate_concept_name": info["concept_name"],
                "resolver": resolvers_list[0],
                "passed": True,
                "standard_concept_id": sc.concept_id,
                "standard_concept_name": sc.concept_name,
                "separation": sc.separation,
                "parent_id": parent_ids[0] if parent_ids else None,
            })
        else:
            hierarchy_validation.append({
                "candidate_concept_id": cid,
                "candidate_concept_name": info["concept_name"],
                "resolver": resolvers_list[0],
                "passed": False,
            })

    if not scored:
        empty_group_scoring = []
        for group_name, members in RESOLVER_GROUPS:
            group_hits = sum(s["hits"] for s in resolver_stages if s["resolver"] in members)
            empty_group_scoring.append({
                "group": group_name, "members": list(members),
                "hit_count": group_hits, "scored_count": 0,
                "top1": None, "target_rank": None, "target_entry": None,
            })
        return {
            "case_id": case_id,
            "query": query,
            "constraints": _constraints_dict(search_constraint, parent_ids, kg),
            "resolver_stages": resolver_stages,
            "hierarchy_validation": hierarchy_validation,
            "resolver_scoring": [],
            "resolver_group_scoring": empty_group_scoring,
            "scoring": [],
            "target_rank": None,
            "top_n_results": [],
        }

    # Stage 2b: per-resolver view — partition the global scored list by resolver attribution.
    # Scores are per-concept and pool-independent, so this is equivalent to running ground_term
    # per resolver without any extra DB calls.
    resolver_scoring = []
    for resolver in ORDERED_RESOLVERS:
        label = _resolver_label(resolver)
        resolver_subset = [sc for sc in scored
                           if label in concept_to_resolvers.get(sc.concept_id, [])]
        if not resolver_subset:
            continue
        top1 = resolver_subset[0]
        resolver_target_rank = None
        target_sc = None
        if expected_concept_id is not None:
            for i, sc in enumerate(resolver_subset):
                if sc.concept_id == expected_concept_id:
                    resolver_target_rank = i + 1
                    target_sc = sc
                    break
        resolver_scoring.append({
            "resolver": label,
            "hit_count": sum(1 for rlist in concept_to_resolvers.values() if label in rlist),
            "scored_count": len(resolver_subset),
            "top1": {
                "concept_id": top1.concept_id,
                "concept_name": top1.concept_name,
                "total_score": round(top1.total_score, 4),
                "is_target": top1.concept_id == expected_concept_id,
            },
            "target_rank": resolver_target_rank,
            "target_entry": {
                "concept_id": target_sc.concept_id,
                "concept_name": target_sc.concept_name,
                "total_score": round(target_sc.total_score, 4),
            } if target_sc and target_sc.concept_id != top1.concept_id else None,
        })

    # Stage 2c: grouped view — synonym + non-synonym merged, for SVG columns
    resolver_group_scoring = []
    for group_name, members in RESOLVER_GROUPS:
        group_subset = [sc for sc in scored
                        if any(m in concept_to_resolvers.get(sc.concept_id, []) for m in members)]
        group_hits = sum(s["hits"] for s in resolver_stages if s["resolver"] in members)
        if not group_subset:
            resolver_group_scoring.append({
                "group": group_name, "members": list(members),
                "hit_count": group_hits, "scored_count": 0,
                "top1": None, "target_rank": None, "target_entry": None,
            })
            continue
        top1 = group_subset[0]
        group_target_rank = None
        target_sc = None
        if expected_concept_id is not None:
            for i, sc in enumerate(group_subset):
                if sc.concept_id == expected_concept_id:
                    group_target_rank = i + 1
                    target_sc = sc
                    break
        resolver_group_scoring.append({
            "group": group_name,
            "members": list(members),
            "hit_count": group_hits,
            "scored_count": len(group_subset),
            "top1": {
                "concept_id": top1.concept_id,
                "concept_name": top1.concept_name,
                "total_score": round(top1.total_score, 4),
                "is_target": top1.concept_id == expected_concept_id,
            },
            "target_rank": group_target_rank,
            "target_entry": {
                "concept_id": target_sc.concept_id,
                "concept_name": target_sc.concept_name,
                "total_score": round(target_sc.total_score, 4),
            } if target_sc and target_sc.concept_id != top1.concept_id else None,
        })

    # Stage 3: build output — annotate each result with which resolvers found it
    target_rank: Optional[int] = None
    if expected_concept_id is not None:
        for i, sc in enumerate(scored):
            if sc.concept_id == expected_concept_id:
                target_rank = i + 1
                break

    scoring_list = []
    for rank, sc in enumerate(scored[:top_n], start=1):
        scoring_list.append({
            "rank": rank,
            "concept_id": sc.concept_id,
            "concept_name": sc.concept_name,
            "relevance": round(sc.relevance, 4),
            "parsimony_penalty": round(sc.parsimony_penalty, 4),
            "broadness_bonus": round(sc.broadness_bonus, 4),
            "total_score": round(sc.total_score, 4),
            "is_target": sc.concept_id == expected_concept_id,
            "resolvers": concept_to_resolvers.get(sc.concept_id, []),
        })
    # Always include target even when it falls outside top_n
    if expected_concept_id and target_rank and target_rank > top_n:
        sc = scored[target_rank - 1]
        scoring_list.append({
            "rank": target_rank,
            "concept_id": sc.concept_id,
            "concept_name": sc.concept_name,
            "relevance": round(sc.relevance, 4),
            "parsimony_penalty": round(sc.parsimony_penalty, 4),
            "broadness_bonus": round(sc.broadness_bonus, 4),
            "total_score": round(sc.total_score, 4),
            "is_target": True,
            "resolvers": concept_to_resolvers.get(sc.concept_id, []),
        })

    top_n_results = []
    for entry in scoring_list:
        info = _concept_info(kg, entry["concept_id"])
        top_n_results.append({
            "rank": entry["rank"],
            "concept_id": entry["concept_id"],
            "concept_name": entry["concept_name"],
            "vocabulary_id": info.get("vocabulary_id", ""),
            "concept_code": info.get("concept_code", ""),
            "total_score": entry["total_score"],
            "is_target": entry["is_target"],
            "resolvers": entry["resolvers"],
        })

    return {
        "case_id": case_id,
        "query": query,
        "constraints": _constraints_dict(search_constraint, parent_ids, kg),
        "resolver_stages": resolver_stages,
        "hierarchy_validation": hierarchy_validation,
        "resolver_scoring": resolver_scoring,
        "resolver_group_scoring": resolver_group_scoring,
        "scoring": scoring_list,
        "target_rank": target_rank,
        "top_n_results": top_n_results,
    }


def _constraints_dict(
    sc: Optional[SearchConstraintConcept],
    parent_ids: Tuple[int, ...],
    kg: KnowledgeGraph,
) -> Dict:
    parent_name = None
    if parent_ids:
        try:
            parent_name = kg.concept_view(parent_ids[0]).concept_name
        except Exception:
            pass
    return {
        "parent_ids": list(parent_ids),
        "parent_concept_name": parent_name,
        "domain": sc.domains[0] if sc and sc.domains else None,
        "vocabularies": list(sc.vocabularies) if sc and sc.vocabularies else None,
        "max_depth": 6,
    }


# ─── SVG generation ───────────────────────────────────────────────────────────

def _svg(trace: Dict, title: str) -> str:
    groups = trace.get("resolver_group_scoring", [])
    n = max(len(groups), 1)

    MARGIN = 30
    W = max(960, 260 * n)
    CW = W - 2 * MARGIN

    HEADER_BG = "#1a3a5c"
    HEADER_FG = "#ffffff"
    BODY_BG = "#f8f9fa"
    HIT_BG = "#d4edda"
    HIT_BORDER = "#28a745"
    MISS_BG = "#f0f0f0"
    MISS_BORDER = "#ced4da"
    AMBER_BG = "#fff8e1"
    AMBER_BORDER = "#856404"
    ARROW_COL = "#495057"
    OUTPUT_BG = "#eaf2ff"
    OUTPUT_BORDER = "#1a3a5c"
    COL_HDR_BG = "#1a6b2e"   # dark green for resolver column header bars
    TEXT_DARK = "#212529"
    TEXT_MID = "#495057"
    TEXT_LIGHT = "#6c757d"
    TEXT_OK = "#1a6b2e"
    FONT = "system-ui, -apple-system, Arial, sans-serif"

    FS_XS  = 10   # eyebrow labels ("INPUT", "GLOBAL OUTPUT")
    FS_SM  = 12   # metadata / secondary text
    FS_MID = 14   # normal body text / concept names
    FS_HDR = 13   # blue bar primary label (group name)

    COL_GAP = 10
    col_w = (CW - (n - 1) * COL_GAP) // n
    col_xs = [MARGIN + i * (col_w + COL_GAP) for i in range(n)]
    max_chars = max(8, col_w // 8)   # ~14px font ≈ 8px/char

    COL_HDR_H  = 86   # column header total (bar 38 + two stats lines + padding)
    COL_BAR_H  = 38   # green bar inside column header
    ARROW_H    = 22
    BLOCK_H    = 80   # result block height
    INPUT_BAR_H = 48  # blue bar inside INPUT block
    OUT_BAR_H  = 44   # blue bar inside GLOBAL OUTPUT block

    parts: List[str] = []
    y = 12   # small top margin; no title text

    def rect(x, yy, w, h, fill, stroke, rx=6, sw=1.5):
        return f'<rect x="{x}" y="{yy}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'

    def text(x, yy, s, fs=FS_MID, fill=TEXT_DARK, anchor="start", bold=False):
        fw = "bold" if bold else "normal"
        return f'<text x="{x}" y="{yy}" font-family="{FONT}" font-size="{fs}" fill="{fill}" text-anchor="{anchor}" font-weight="{fw}">{escape(str(s))}</text>'

    def col_arrow(cx, yy, h=ARROW_H):
        return (
            f'<line x1="{cx}" y1="{yy}" x2="{cx}" y2="{yy+h-8}" stroke="{ARROW_COL}" stroke-width="1.5"/>'
            f'<polygon points="{cx-5},{yy+h-8} {cx+5},{yy+h-8} {cx},{yy+h}" fill="{ARROW_COL}"/>'
        )

    def trunc(s, n_chars):
        return s[:n_chars] + "…" if len(s) > n_chars else s

    def field_row(x, yy, fields):
        """One <text> element: bold label + normal value pairs separated by  |  dividers."""
        tspans = []
        for i, (lbl, val) in enumerate(fields):
            if i > 0:
                tspans.append(f'<tspan fill="{TEXT_LIGHT}">  |  </tspan>')
            tspans.append(f'<tspan font-weight="bold" fill="{TEXT_DARK}">{escape(str(lbl))}</tspan>')
            tspans.append(f'<tspan fill="{TEXT_MID}"> {escape(str(val))}</tspan>')
        return f'<text x="{x}" y="{yy}" font-family="{FONT}" font-size="{FS_SM}">{"".join(tspans)}</text>'

    constraints = trace.get("constraints", {})
    parent_id = (constraints.get("parent_ids") or [None])[0]
    parent_name = constraints.get("parent_concept_name") or (f"concept {parent_id}" if parent_id else "—")

    # Resolve expected concept for the TARGET block
    target_result  = next((r for r in trace.get("top_n_results", []) if r.get("is_target")), None)
    expected_name  = trace.get("expected_concept_name") or (target_result.get("concept_name") if target_result else None)
    expected_id    = target_result.get("concept_id") if target_result else None
    has_target_blk = expected_name is not None or expected_id is not None

    # ── 1. INPUT + TARGET ──
    input_h  = 104   # bar(48) + two body rows at 22px each + 12px padding
    input_w  = int(CW * 0.58) if has_target_blk else CW
    target_w = CW - input_w - 12   # 12px gap

    # INPUT outer box
    parts.append(rect(MARGIN, y, input_w, input_h, BODY_BG, HEADER_BG, rx=6, sw=2))
    # blue bar: rounded top, flat bottom
    parts.append(rect(MARGIN, y, input_w, INPUT_BAR_H, HEADER_BG, HEADER_BG, rx=6, sw=0))
    parts.append(f'<rect x="{MARGIN}" y="{y + INPUT_BAR_H // 2}" width="{input_w}" height="{INPUT_BAR_H // 2}" fill="{HEADER_BG}"/>')
    # eyebrow + query in bar
    parts.append(text(MARGIN + 12, y + 17, "INPUT", FS_XS, HEADER_FG))
    parts.append(text(MARGIN + 12, y + 37, trunc(f'"{trace.get("query", "")}"', input_w // 8), FS_MID, HEADER_FG, bold=True))
    # body: single compact row
    domain_str = constraints.get("domain") or "—"
    vocabs_str = ", ".join(constraints.get("vocabularies") or [])
    parent_str = f"{parent_name}  (concept {parent_id})" if parent_id else "—"
    parts.append(field_row(MARGIN + 14, y + INPUT_BAR_H + 20, [
        ("Domain:", domain_str),
        ("Vocabularies:", vocabs_str),
    ]))
    parts.append(field_row(MARGIN + 14, y + INPUT_BAR_H + 42, [
        ("Parent:", parent_str),
    ]))

    # TARGET block (golden, right of INPUT)
    if has_target_blk:
        tx = MARGIN + input_w + 12
        parts.append(rect(tx, y, target_w, input_h, AMBER_BG, AMBER_BORDER, rx=6, sw=2))
        parts.append(rect(tx, y, target_w, INPUT_BAR_H, AMBER_BORDER, AMBER_BORDER, rx=6, sw=0))
        parts.append(f'<rect x="{tx}" y="{y + INPUT_BAR_H // 2}" width="{target_w}" height="{INPUT_BAR_H // 2}" fill="{AMBER_BORDER}"/>')
        parts.append(text(tx + 12, y + 17, "TARGET", FS_XS, HEADER_FG))
        parts.append(text(tx + 12, y + 37, trunc(expected_name or "—", target_w // 8), FS_MID, HEADER_FG, bold=True))
        if expected_id:
            parts.append(field_row(tx + 14, y + INPUT_BAR_H + 22, [("Concept ID:", str(expected_id))]))

    y += input_h + 16

    # ── 2. PER-GROUP COLUMNS (Exact / Partial / Full Text / Embedding) ──
    col_y = y
    col_bottoms = []

    for i, group_entry in enumerate(groups):
        cx = col_xs[i]
        ccx = cx + col_w // 2
        cy = col_y

        label = group_entry["group"]
        hits = group_entry.get("hit_count", 0)
        scored_cnt = group_entry.get("scored_count", 0)
        has_results = scored_cnt > 0

        # ── Column header: blue bar + stats body ──
        hdr_border = HIT_BORDER if has_results else MISS_BORDER
        parts.append(rect(cx, cy, col_w, COL_HDR_H, BODY_BG, hdr_border, rx=6, sw=1.5))
        if has_results:
            parts.append(rect(cx, cy, col_w, COL_BAR_H, COL_HDR_BG, COL_HDR_BG, rx=6, sw=0))
            parts.append(f'<rect x="{cx}" y="{cy + COL_BAR_H // 2}" width="{col_w}" height="{COL_BAR_H // 2}" fill="{COL_HDR_BG}"/>')
            parts.append(text(ccx, cy + 25, label, FS_HDR, HEADER_FG, "middle", bold=True))
        else:
            parts.append(rect(cx, cy, col_w, COL_BAR_H, MISS_BG, MISS_BORDER, rx=6, sw=0))
            parts.append(f'<rect x="{cx}" y="{cy + COL_BAR_H // 2}" width="{col_w}" height="{COL_BAR_H // 2}" fill="{MISS_BG}"/>')
            parts.append(text(ccx, cy + 25, label, FS_HDR, TEXT_MID, "middle", bold=True))
        stats_y = cy + COL_BAR_H + 18
        parts.append(text(ccx, stats_y, f"{hits:,} candidates", FS_SM, TEXT_MID, "middle"))
        parts.append(
            f'<text x="{ccx}" y="{stats_y + 22}" text-anchor="middle"'
            f' font-family="{FONT}" font-size="{FS_SM}">'
            f'<tspan font-weight="bold" fill="{TEXT_DARK}">{scored_cnt:,}</tspan>'
            f'<tspan fill="{TEXT_MID}"> passed hierarchy</tspan></text>'
        )
        cy += COL_HDR_H

        # ── Arrow ──
        parts.append(col_arrow(ccx, cy))
        cy += ARROW_H

        # ── Result block: top-1 concept + optional target rank footer ──
        if not has_results:
            parts.append(rect(cx, cy, col_w, BLOCK_H, MISS_BG, MISS_BORDER, rx=5, sw=1))
            parts.append(text(ccx, cy + 36, "—", 16, MISS_BORDER, "middle", bold=True))
            parts.append(text(ccx, cy + 60, "no results passed hierarchy", FS_XS, TEXT_LIGHT, "middle"))
        else:
            top1 = group_entry.get("top1") or {}
            is_tgt = top1.get("is_target", False)
            b1_fill = HIT_BG if is_tgt else BODY_BG
            b1_border = HIT_BORDER if is_tgt else "#adb5bd"
            rank_col = TEXT_OK if is_tgt else TEXT_DARK
            parts.append(rect(cx, cy, col_w, BLOCK_H, b1_fill, b1_border, rx=5, sw=1.5))
            parts.append(text(ccx, cy + 22, f"Rank 1  |  {top1.get('concept_id', '')}", FS_SM, rank_col, "middle", bold=True))
            parts.append(text(ccx, cy + 46, trunc(top1.get("concept_name", "—"), max_chars), FS_MID, TEXT_DARK, "middle"))
            if group_entry.get("target_entry"):
                trank = group_entry.get("target_rank", "?")
                parts.append(text(ccx, cy + 66, f"Target Rank: {trank}", FS_SM, AMBER_BORDER, "middle", bold=True))
        cy += BLOCK_H

        col_bottoms.append(cy)

    max_bottom = max(col_bottoms) if col_bottoms else col_y
    y = max_bottom + 24

    # ── 3. GLOBAL OUTPUT ──
    top_results = trace.get("top_n_results", [])
    top_out = top_results[0] if top_results else {}
    target_rank = trace.get("target_rank")
    out_h = 100
    parts.append(rect(MARGIN, y, CW, out_h, OUTPUT_BG, OUTPUT_BORDER, rx=8, sw=3))
    # blue bar: rounded top, flat bottom
    parts.append(rect(MARGIN, y, CW, OUT_BAR_H, OUTPUT_BORDER, OUTPUT_BORDER, rx=8, sw=0))
    parts.append(f'<rect x="{MARGIN}" y="{y + OUT_BAR_H // 2}" width="{CW}" height="{OUT_BAR_H // 2}" fill="{OUTPUT_BORDER}"/>')
    parts.append(text(MARGIN + 14, y + 16, "GLOBAL OUTPUT", FS_XS, HEADER_FG))
    if top_out:
        name_full = top_out.get("concept_name", "")
        parts.append(text(MARGIN + 14, y + 34, trunc(name_full, CW // 8), FS_MID, HEADER_FG, bold=True))
        rank_color = TEXT_OK if target_rank == 1 else (AMBER_BORDER if target_rank else "#b02a37")
        rank_val   = str(target_rank) if target_rank else "not grounded"
        tspans: List[str] = []
        for lbl, val in [
            ("Vocabulary:", top_out.get("vocabulary_id", "—")),
            ("Code:",       top_out.get("concept_code", "—")),
            ("Concept:",    str(top_out.get("concept_id", "—"))),
        ]:
            if tspans:
                tspans.append(f'<tspan fill="{TEXT_LIGHT}">  |  </tspan>')
            tspans.append(f'<tspan font-weight="bold" fill="{TEXT_DARK}">{escape(lbl)}</tspan>')
            tspans.append(f'<tspan fill="{TEXT_MID}"> {escape(val)}</tspan>')
        tspans.append(f'<tspan fill="{TEXT_LIGHT}">  |  </tspan>')
        tspans.append(f'<tspan font-weight="bold" fill="{TEXT_DARK}">Target rank:</tspan>')
        tspans.append(f'<tspan font-weight="bold" fill="{rank_color}"> {escape(rank_val)}</tspan>')
        parts.append(
            f'<text x="{MARGIN + 14}" y="{y + OUT_BAR_H + 24}" font-family="{FONT}" font-size="{FS_MID}">{"".join(tspans)}</text>'
        )
    else:
        parts.append(text(MARGIN + 14, y + 34, "—", FS_MID, HEADER_FG, bold=True))
        parts.append(text(MARGIN + 14, y + OUT_BAR_H + 24, "No result — query could not be grounded.", FS_SM, "#b02a37"))

    y += out_h + 20

    # ── Assemble ──
    svg_body = "\n".join(parts)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{y}" viewBox="0 0 {W} {y}">
  <rect width="{W}" height="{y}" fill="white"/>
  {svg_body}
</svg>"""


# ─── CLI commands ──────────────────────────────────────────────────────────────

@app.command()
def trace(
    cases_file: Annotated[Optional[str], typer.Option("--cases-file", "-c",
        help="JSON benchmark cases file. Runs all cases unless --case-ids is given.")] = None,
    case_ids: Annotated[Optional[List[str]], typer.Option("--case-id", "-i",
        help="Optional case IDs to run. For multiple IDs, repeat the option (e.g., -i case1 -i case2). Default: None (all cases run)")] = None,
    query: Annotated[Optional[str], typer.Option("--query", "-q",
        help="Query text for on-demand evaluation, circumventing the cases file.")] = None,
    parent_ids: Annotated[Optional[List[int]], typer.Option("--parent-id", "-G",
        help="Provide parent concept IDs for hierarchy anchoring. For multiple IDs, repeat the option (e.g., -G 443392 -G 413015).")] = None,
    domains: Annotated[Optional[List[str]], typer.Option("--domain", "-D",
        help="OMOP domain filter. For multiple domains, repeat the option (e.g., -D Condition -D Procedure).")] = None,
    vocabularies: Annotated[Optional[List[str]], typer.Option("--vocabulary", "-V",
        help="Vocabulary filter. For multiple vocabularies, repeat the option (e.g., -V SNOMED -V ICDO3).")] = None,
    query_expected_concept_id: Annotated[Optional[int], typer.Option("--expected-concept-id", "-e",
        help="Expected concept ID for on-demand evaluation.")] = None,
    top_n: Annotated[int, typer.Option("--top-n", "-n",
        help="Number of top results to include per case.")] = 5,
    out_dir: Annotated[Optional[str], typer.Option("--out-dir", "-o",
        help="Base output directory. Writes one file per case to <out-dir>/<safe_model_name>/trace_<case_id>.json. If omitted, prints combined JSON to stdout.")] = None,
    metric_type: Annotated[str, typer.Option("--metric-type", "-m",
        help="Distance metric used when embedding was indexed (cosine or l2). Must match the metric registered for the model.")] = "cosine",
    embedding_model: Annotated[Optional[str], typer.Option("--embedding-model", "-E",
        help="Embedding model name to use, overriding the one configured in config.toml (e.g. to compare multiple ingested models).")] = None,
):
    """Trace grounding cases through every resolver stage. Outputs {\"cases\": [...]}."""
    # Build the list of cases to run
    cases_to_run: List[Dict] = []
    
    if cases_file:
        payload = json.loads(Path(cases_file).read_text())
        rows = payload if isinstance(payload, list) else [r for bucket in payload.values() for r in bucket]
        if case_ids:
            rows = [r for r in rows if r.get("id") in case_ids]
            missing = set(case_ids) - {r.get("id") for r in rows}
            for mid in sorted(missing):
                typer.echo(f"Warning: case '{mid}' not found in {cases_file}.", err=True)
        cases_to_run = rows
    elif query:
        if not parent_ids:
            typer.echo("--parent-ids required with --query.", err=True)
            raise typer.Exit(1)
        cases_to_run = [{
            "id": "on-demand",
            "text": query,
            "expected_concept_id": query_expected_concept_id,
            "expected_concept_name": None,
        }]
    else:
        typer.echo("Provide --cases-file or --query.", err=True)
        raise typer.Exit(1)

    if not cases_to_run:
        typer.echo("No cases to run.", err=True)
        raise typer.Exit(1)

    typer.echo("Building knowledge graph…", err=True)
    from omop_emb.config import parse_metric_type
    kg = _build_kg(metric_type=parse_metric_type(metric_type), embedding_model=embedding_model)
    used_embedding_model = (
        kg.embedding_configuration.model_name if kg.embedding_configuration else None
    )

    results = []
    for case in cases_to_run:
        case_id = case.get("id")
        case_query = case.get("text")
        case_domains = domains or ([case.get("domain")] if case.get("domain") else None) 
        case_vocab = vocabularies or ([case.get("vocabulary")] if case.get("vocabulary") else None)
        case_parent_ids = parent_ids or case.get("parent_ids")
        case_expected = query_expected_concept_id or case.get("expected_concept_id")
        case_expected_name = case.get("expected_concept_name")

        if case_query is None:
            typer.echo(f"Case '{case_id}': no query text, skipping.", err=True)
            continue
        if case_id is None:
            typer.echo(f"Case with query '{case_query}': no case ID, skipping.", err=True)
            continue
        if case_parent_ids is None:
            typer.echo(f"Case '{case_id}': no parent IDs, skipping.", err=True)
            continue

        search_constraint = SearchConstraintConcept(
            domains=tuple([str(c) for c in case_domains]) if isinstance(case_domains, list) else None,
            vocabularies=tuple([str(v) for v in case_vocab]) if isinstance(case_vocab, list) else None,
            require_standard=False,
        )

        typer.echo(f"  [{case_id}] '{case_query}'", err=True)
        result = _run_trace(
            case_id=case_id,
            kg=kg,
            query=case_query,
            parent_ids=tuple(case_parent_ids),
            search_constraint=search_constraint,
            expected_concept_id=case_expected,
            top_n=top_n,
        )
        result["case_id"] = case_id
        result["expected_concept_name"] = case_expected_name
        results.append(result)

    hits = [r for r in results if r.get("target_rank") == 1]

    if out_dir:
        out_dir_path = Path(out_dir) / _model_dir_name(used_embedding_model)
        out_dir_path.mkdir(parents=True, exist_ok=True)
        for result in results:
            cid = result["case_id"]
            case_output = json.dumps(
                {"cases": [result], "embedding_model": used_embedding_model},
                indent=2,
                ensure_ascii=False,
            )
            (out_dir_path / f"trace_{cid}.json").write_text(case_output, encoding="utf-8")

        typer.echo(f"Trace written to {out_dir_path} ({len(results)} case(s)).", err=True)
        typer.echo(f"Target rank 1: {len(hits)}/{len(results)}", err=True)
    else:
        output = json.dumps(
            {"cases": results, "embedding_model": used_embedding_model},
            indent=2,
            ensure_ascii=False,
        )
        print(output)


@app.command()
def svg(
    trace_dir: Annotated[str, typer.Option("--trace-dir", "-t",
        help="Base output directory passed to 'trace --out-dir'. The model-specific "
             "subdirectory is resolved the same way 'trace' does, via --embedding-model "
             "or config.toml.")],
    embedding_model: Annotated[Optional[str], typer.Option("--embedding-model", "-E",
        help="Embedding model name used to resolve the subdirectory, overriding "
             "config.toml (same semantics as 'trace --embedding-model').")] = None,
    case_id: Annotated[Optional[str], typer.Option("--case-id", "-i",
        help="Case ID to render. Omitted, empty, or 'all' renders every case found.")] = None,
    title: Annotated[str, typer.Option("--title",
        help="Figure title shown at the top.")] = "omop-graph Concept Grounding",
):
    """Generate flowchart SVG(s) from trace JSON files written by the trace command. Wirtes them to plots/ subdirectory of
    the same directory as the traces."""
    resolved_model = _resolve_embedding_model(embedding_model)
    trace_dir_pl = Path(trace_dir) / _model_dir_name(resolved_model)

    json_files = sorted(trace_dir_pl.glob("*.json"))
    if not json_files:
        typer.echo(f"No JSON trace files found in {trace_dir_pl}.", err=True)
        raise typer.Exit(1)
    cases: List[Dict] = []
    for json_file in json_files:
        payload = json.loads(json_file.read_text(encoding="utf-8"))
        cases.extend(payload.get("cases", [payload]))

    if case_id and case_id.lower() != "all":
        selected = [c for c in cases if (c.get("case_id") or "ad-hoc") == case_id]
        if not selected:
            typer.echo(f"Case '{case_id}' not found in {trace_dir_pl}.", err=True)
            raise typer.Exit(1)
    else:
        selected = cases

    out_dir = trace_dir_pl / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    for trace_data in selected:
        cid = trace_data.get("case_id") or "ad-hoc"
        svg_content = _svg(trace_data, title)
        out_file = out_dir / f"trace_{cid}.svg"
        out_file.write_text(svg_content, encoding="utf-8")
        typer.echo(f"SVG written to {out_file}")


if __name__ == "__main__":
    app()
