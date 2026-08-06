"""
Trace a single grounding example step-by-step and optionally generate SVG figures.

Commands
--------
trace         Run one case through every resolver, capture per-stage results, output JSON.
pipeline-svg  Read a trace JSON and render a Whimsical-importable resolver-pipeline flowchart SVG.
panel-svg     Render relationship-classification edge filtering (Panel A) and hierarchy-
              constrained candidate culling (Panel B) as summary dashboards.
graph-svg     Render the actual graph traversal: starting candidates hopping via a real
              Identity edge to a standard concept, then either reaching the hierarchy anchor
              (kept, highlighted) or dead-ending (culled), labeled by resolver.
"""

from __future__ import annotations

import json
import logging
from html import escape
from pathlib import Path
from typing import Annotated, Dict, List, Optional, Tuple

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from omop_graph.config import OmopGraphConfig
from omop_graph.db.session import make_engine
from omop_graph.extensions.emb import get_embedding_writer_interface
from omop_graph.extensions.omop_alchemy import PredicateKind
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

# Shared console for logging and progress bar
console = Console(stderr=True)

app = typer.Typer(help="Trace a single OMOP grounding example and generate an SVG figure.")


@app.callback()
def _main(
    verbose: Annotated[
        int,
        typer.Option("--verbose", "-v", count=True, help="Increase log verbosity (-v INFO, -vv DEBUG). Must come before the subcommand name."),
    ] = 0,
) -> None:
    OmopGraphConfig.configure_logging(verbosity=verbose, console=console)

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
    """Build a KG with embedding support, resolved from OmopGraphConfig."""
    cdm_engine = make_engine()
    try:
        from oa_configurator import Resolver
        from omop_emb.backends import resolve_backend_from_resolved_vector_store
        from omop_emb.config import MetricType

        resolved_metric = metric_type if metric_type is not None else MetricType.COSINE
        cfg = OmopGraphConfig.get_config()
        resolved_model_name = embedding_model or cfg.embedding_model_name
        if resolved_model_name is None or cfg.vector_store_name is None:
            raise RuntimeError(
                "No embedding model/vector store configured. Set embedding_model_name and "
                "vector_store_name via `omop-config configure omop_graph`, or pass --embedding-model."
            )
        resolver = Resolver.from_active_config()
        resolved_model = resolver.resolve_model(resolved_model_name)
        resolved_vector_store = resolver.resolve_vector_store(cfg.vector_store_name)
        backend = resolve_backend_from_resolved_vector_store(resolved_vector_store)

        emb_config = KnowledgeGraphEmbeddingConfiguration(
            metric_type=resolved_metric,
            backend=backend,
            resolved_model=resolved_model,
            write=True,
        )
        logger.info("Embedding config loaded (model=%s, metric=%s).", resolved_model.model, resolved_metric.value)
        return KnowledgeGraph(cdm_engine=cdm_engine, emb_config=emb_config)
    except Exception as exc:
        logger.warning("Error occurred while loading embedding config:\n%s.\nRunning without embedding.", exc)
        return KnowledgeGraph(cdm_engine=cdm_engine)


def _resolver_label(resolver) -> str:
    return RESOLVER_DISPLAY_NAMES.get((resolver.match_kind, resolver.synonym), type(resolver).__name__)


def _resolve_embedding_model(embedding_model: Optional[str]) -> str:
    """Resolve the canonical embedding model name (override or config default).

    Mirrors the canonicalisation `_build_kg` does, but without opening a DB
    connection or resolving a vector store; usable from `svg`, which never
    builds a KnowledgeGraph.
    """
    from oa_configurator import Resolver
    from omop_llm import build_model_backend_from_resolved

    cfg = OmopGraphConfig.get_config()
    resolved_name = embedding_model or cfg.embedding_model_name
    if resolved_name is None:
        raise RuntimeError(
            "No embedding model configured. Set embedding_model_name via "
            "`omop-config configure omop_graph`, or pass --embedding-model."
        )
    resolved_model = Resolver.from_active_config().resolve_model(resolved_name)
    return build_model_backend_from_resolved(resolved_model).model


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
        from omop_emb.interface import EmbeddingRole
        writer = get_embedding_writer_interface(kg)
        if writer is not None:
            query_embedding = writer.embed_texts(texts=(query,), role=EmbeddingRole.QUERY)
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
            console.print(f"  {label}: skipped ({exc})")
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

    # Reconstruct hierarchy_validation from scored vs resolver hits (for SVG compatibility).
    # Indexed by both the standard (post-promotion) id and the original (pre-promotion) id,
    # since a candidate that required an Identity-hop promotion is looked up by its original id.
    scored_by_id = {}
    for sc in scored:
        scored_by_id[sc.concept_id] = sc
        scored_by_id[sc.original_id] = sc
    hierarchy_validation = []
    for cid, resolvers_list in concept_to_resolvers.items():
        info = _concept_info(kg, cid)
        if cid in scored_by_id:
            sc = scored_by_id[cid]
            hierarchy_validation.append({
                "candidate_concept_id": cid,
                "candidate_concept_name": info["concept_name"],
                "resolver": resolvers_list[0],
                "resolvers": resolvers_list,
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
                "resolvers": resolvers_list,
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


# ─── Panel dashboard (Panel A: relationship classification, Panel B: hierarchy culling) ──────

# Render order: Identity first (the only walkable-by-default class for grounding), then the
# excluded-by-default classes.
PREDICATE_KIND_ORDER = (
    PredicateKind.IDENTITY,
    PredicateKind.HIERARCHY,
    PredicateKind.COMPOSITION,
    PredicateKind.ASSOCIATION,
    PredicateKind.ATTRIBUTE,
)


def _relationship_classification(kg: KnowledgeGraph, concept_id: int) -> Dict:
    """Group a concept's outgoing edges by predicate_kind for the Panel A diagram.

    One representative edge (relationship + target concept name) per class, plus the total
    count in that class, is enough to show which classes are walkable for grounding (Identity)
    vs. excluded by default (everything else) without enumerating every edge.
    """
    edges = kg.edges(concept_ids=concept_id, direction="out", active_only=True, within_domain=False)
    by_kind: Dict[PredicateKind, List] = {}
    for edge in edges:
        by_kind.setdefault(edge.predicate_kind, []).append(edge)

    object_ids = tuple({edge.object_id for edge in edges})
    names = {cv.concept_id: cv.concept_name for cv in kg.concept_views(object_ids)} if object_ids else {}

    groups = []
    for kind in PREDICATE_KIND_ORDER:
        kind_edges = by_kind.get(kind, [])
        if not kind_edges:
            groups.append({"kind": kind.value, "count": 0, "example": None})
            continue
        rep = kind_edges[0]
        groups.append({
            "kind": kind.value,
            "count": len(kind_edges),
            "example": {
                "predicate_id": rep.predicate_id,
                "predicate_name": kg.predicate_name(rep.predicate_id),
                "object_id": rep.object_id,
                "object_name": names.get(rep.object_id, f"[concept {rep.object_id}]"),
            },
        })

    concept_name = kg.concept_view(concept_id).concept_name
    return {"concept_id": concept_id, "concept_name": concept_name, "groups": groups}


def _pick_hierarchy_examples(hierarchy_validation: List[Dict], passed: bool, limit: int = 3) -> List[Dict]:
    """Pick up to `limit` distinct-by-name hierarchy_validation entries for Panel B.

    For failures, Embedding-only hits are surfaced first: an Embedding candidate that falls
    outside the hierarchy anchor is the most narratively useful failure mode for this figure
    (it is what the constrained traversal protects against), so it's preferred over lexical
    near-duplicates that also failed for unrelated reasons.
    """
    candidates = [e for e in hierarchy_validation if e.get("passed") == passed]
    if not passed:
        candidates = sorted(candidates, key=lambda e: e.get("resolver") != "Embedding")
    seen_names: set = set()
    picked = []
    for entry in candidates:
        name = entry.get("candidate_concept_name", "")
        if name in seen_names:
            continue
        seen_names.add(name)
        picked.append(entry)
        if len(picked) >= limit:
            break
    return picked


def _panel_svg(rel_class: Dict, trace: Dict, title: str) -> str:
    """Render the two-panel dashboard: Panel A (relationship classification / edge filtering)
    above Panel B (hierarchy-constrained candidate culling)."""
    MARGIN = 30
    W = 1040
    CW = W - 2 * MARGIN
    FONT = "system-ui, -apple-system, Arial, sans-serif"

    HEADER_BG = "#1a3a5c"
    HEADER_FG = "#ffffff"
    BODY_BG = "#f8f9fa"
    HIT_BG = "#d4edda"
    HIT_BORDER = "#28a745"
    MISS_BG = "#f0f0f0"
    MISS_BORDER = "#ced4da"
    FAIL_BORDER = "#b02a37"
    ARROW_COL = "#495057"
    COL_HDR_BG_WALK = "#1a6b2e"     # green: walkable for grounding (Identity)
    COL_HDR_BG_EXCL = "#6c757d"    # gray: excluded by default (everything else)
    TEXT_DARK = "#212529"
    TEXT_MID = "#495057"
    TEXT_LIGHT = "#6c757d"
    TEXT_OK = "#1a6b2e"

    FS_XS, FS_SM, FS_MID, FS_HDR = 10, 12, 14, 13

    parts: List[str] = []

    def rect(x, yy, w, h, fill, stroke, rx=6, sw=1.5):
        return f'<rect x="{x}" y="{yy}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'

    def text(x, yy, s, fs=FS_MID, fill=TEXT_DARK, anchor="start", bold=False):
        fw = "bold" if bold else "normal"
        return f'<text x="{x}" y="{yy}" font-family="{FONT}" font-size="{fs}" fill="{fill}" text-anchor="{anchor}" font-weight="{fw}">{escape(str(s))}</text>'

    def trunc(s, n_chars):
        return s[:n_chars] + "…" if len(s) > n_chars else s

    def fan_arrows(src_cx, src_y, col_xs, col_w, dst_y):
        """Draw a bus line fanning out from one source point down into N columns."""
        bus_y = src_y + (dst_y - src_y) // 2
        out = [f'<line x1="{src_cx}" y1="{src_y}" x2="{src_cx}" y2="{bus_y}" stroke="{ARROW_COL}" stroke-width="1.5"/>']
        col_centers = [cx + col_w // 2 for cx in col_xs]
        out.append(f'<line x1="{min(col_centers)}" y1="{bus_y}" x2="{max(col_centers)}" y2="{bus_y}" stroke="{ARROW_COL}" stroke-width="1.5"/>')
        for ccx in col_centers:
            out.append(f'<line x1="{ccx}" y1="{bus_y}" x2="{ccx}" y2="{dst_y-8}" stroke="{ARROW_COL}" stroke-width="1.5"/>')
            out.append(f'<polygon points="{ccx-5},{dst_y-8} {ccx+5},{dst_y-8} {ccx},{dst_y}" fill="{ARROW_COL}"/>')
        return "\n".join(out)

    y = 12

    # ── Panel A: relationship classification ──
    parts.append(text(MARGIN, y + 8, "PANEL A — Relationship classification (edge filtering)", FS_SM, TEXT_LIGHT, bold=True))
    y += 18

    node_h = 56
    parts.append(rect(MARGIN, y, CW, node_h, BODY_BG, HEADER_BG, rx=6, sw=2))
    parts.append(rect(MARGIN, y, CW, 28, HEADER_BG, HEADER_BG, rx=6, sw=0))
    parts.append(f'<rect x="{MARGIN}" y="{y+14}" width="{CW}" height="14" fill="{HEADER_BG}"/>')
    parts.append(text(MARGIN + 12, y + 19, "CONCEPT", FS_XS, HEADER_FG))
    parts.append(text(MARGIN + 12, y + 48, trunc(f'{rel_class["concept_name"]}  (concept {rel_class["concept_id"]})', CW // 8), FS_MID, TEXT_DARK, bold=True))
    concept_box_bottom = y + node_h
    concept_cx = MARGIN + CW // 2
    y = concept_box_bottom + 36

    groups = rel_class["groups"]
    n = len(groups)
    col_gap = 10
    col_w = (CW - (n - 1) * col_gap) // n
    col_xs = [MARGIN + i * (col_w + col_gap) for i in range(n)]

    parts.append(fan_arrows(concept_cx, concept_box_bottom, col_xs, col_w, y))

    col_h = 110
    for i, group in enumerate(groups):
        cx = col_xs[i]
        ccx = cx + col_w // 2
        is_identity = group["kind"] == PredicateKind.IDENTITY.value
        has_edges = group["count"] > 0
        hdr_bg = COL_HDR_BG_WALK if (is_identity and has_edges) else COL_HDR_BG_EXCL
        body_border = HIT_BORDER if (is_identity and has_edges) else MISS_BORDER
        body_bg = HIT_BG if (is_identity and has_edges) else BODY_BG if has_edges else MISS_BG

        parts.append(rect(cx, y, col_w, col_h, body_bg, body_border, rx=6, sw=1.5))
        parts.append(rect(cx, y, col_w, 30, hdr_bg, hdr_bg, rx=6, sw=0))
        parts.append(f'<rect x="{cx}" y="{y+15}" width="{col_w}" height="15" fill="{hdr_bg}"/>')
        parts.append(text(ccx, y + 20, group["kind"], FS_HDR, HEADER_FG, "middle", bold=True))
        parts.append(text(ccx, y + 46, f'{group["count"]} edges', FS_SM, TEXT_MID, "middle"))
        if group["example"]:
            ex = group["example"]
            edge_label = trunc(f'-[{ex["predicate_name"]}]-> {ex["object_name"]}', max(8, col_w // 7))
            parts.append(text(ccx, y + 66, edge_label, FS_SM, TEXT_DARK, "middle"))
        else:
            parts.append(text(ccx, y + 66, "no edges", FS_SM, TEXT_LIGHT, "middle"))
        tag = "walkable for grounding" if is_identity else "excluded by default"
        tag_col = TEXT_OK if is_identity else TEXT_LIGHT
        parts.append(text(ccx, y + 92, tag, FS_XS, tag_col, "middle", bold=is_identity))

    y += col_h + 40

    # ── Panel B: hierarchy-constrained culling ──
    constraints = trace.get("constraints", {})
    parent_id = (constraints.get("parent_ids") or [None])[0]
    parent_name = constraints.get("parent_concept_name") or (f"concept {parent_id}" if parent_id else "—")

    parts.append(text(MARGIN, y + 8, "PANEL B — Hierarchy-constrained culling", FS_SM, TEXT_LIGHT, bold=True))
    y += 18

    anchor_h = 56
    parts.append(rect(MARGIN, y, CW, anchor_h, BODY_BG, HEADER_BG, rx=6, sw=2))
    parts.append(rect(MARGIN, y, CW, 28, HEADER_BG, HEADER_BG, rx=6, sw=0))
    parts.append(f'<rect x="{MARGIN}" y="{y+14}" width="{CW}" height="14" fill="{HEADER_BG}"/>')
    parts.append(text(MARGIN + 12, y + 19, "ANCHOR", FS_XS, HEADER_FG))
    parts.append(text(MARGIN + 12, y + 48, trunc(f'{parent_name}  (concept {parent_id})' if parent_id else "—", CW // 8), FS_MID, TEXT_DARK, bold=True))
    anchor_box_bottom = y + anchor_h
    anchor_cx = MARGIN + CW // 2
    y = anchor_box_bottom + 36

    hierarchy_validation = trace.get("hierarchy_validation", [])
    pass_examples = _pick_hierarchy_examples(hierarchy_validation, passed=True)
    fail_examples = _pick_hierarchy_examples(hierarchy_validation, passed=False)

    pcol_w = (CW - col_gap) // 2
    pcol_xs = [MARGIN, MARGIN + pcol_w + col_gap]
    row_h = 36
    pcol_h = 44 + row_h * max(len(pass_examples), len(fail_examples), 1)

    parts.append(fan_arrows(anchor_cx, anchor_box_bottom, pcol_xs, pcol_w, y))

    for col_idx, (label, examples, is_pass) in enumerate((
        ("PASSED hierarchy anchor", pass_examples, True),
        ("FAILED hierarchy anchor", fail_examples, False),
    )):
        cx = pcol_xs[col_idx]
        ccx = cx + pcol_w // 2
        hdr_bg = HIT_BORDER if is_pass else FAIL_BORDER
        body_border = HIT_BORDER if is_pass else FAIL_BORDER
        parts.append(rect(cx, y, pcol_w, pcol_h, "#ffffff", body_border, rx=6, sw=1.5))
        parts.append(rect(cx, y, pcol_w, 28, hdr_bg, hdr_bg, rx=6, sw=0))
        parts.append(f'<rect x="{cx}" y="{y+14}" width="{pcol_w}" height="14" fill="{hdr_bg}"/>')
        parts.append(text(ccx, y + 19, label, FS_HDR, HEADER_FG, "middle", bold=True))
        if not examples:
            parts.append(text(ccx, y + 28 + row_h // 2 + 4, "no examples", FS_SM, TEXT_LIGHT, "middle"))
        for ri, entry in enumerate(examples):
            ry = y + 28 + ri * row_h + row_h // 2
            name = trunc(entry.get("candidate_concept_name", ""), max(8, pcol_w // 7))
            cid = entry.get("candidate_concept_id")
            resolver = entry.get("resolver", "?")
            fill = TEXT_OK if is_pass else FAIL_BORDER
            parts.append(text(cx + 10, ry, f'{name}  ({cid})', FS_SM, TEXT_DARK, bold=False))
            detail = f'sep {entry["separation"]} | found by: {resolver}' if is_pass else f'found by: {resolver} | outside anchor'
            parts.append(text(cx + 10, ry + 15, detail, FS_XS, fill))

    y += pcol_h + 24

    svg_body = "\n".join(parts)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{y}" viewBox="0 0 {W} {y}">
  <rect width="{W}" height="{y}" fill="white"/>
  {svg_body}
</svg>"""


# ─── Funnel diagram: candidates -> standard concepts -> hierarchy-validated -> scored/winner ──

# Resolver-attribution stripe colors (distinct from outcome colors so the two channels never
# clash). Keyed by the canonical RESOLVER_GROUPS group name.
RESOLVER_STRIPE_COLORS = {
    "Exact": "#0d6efd",
    "Partial": "#6f42c1",
    "Full Text": "#0aa394",
    "Embedding": "#fd7e14",
}


def _resolver_groups_for(resolvers: List[str]) -> List[str]:
    """Map a candidate's full resolver list to its distinct, canonically-ordered group names
    (e.g. ["Exact", "Exact Synonym"] -> ["Exact"]; preserves RESOLVER_GROUPS order)."""
    resolver_set = set(resolvers)
    return [group_name for group_name, members in RESOLVER_GROUPS if resolver_set & set(members)]


def _resolve_identity_hop(kg: KnowledgeGraph, candidate_id: int, standard_id: Optional[int]) -> Dict:
    """Resolve the real Identity edge (if any) a candidate uses to reach a standard concept.

    Returns ``{"already_standard": bool, "predicate_name": str|None, "object_id": int|None,
    "object_name": str|None, "vocabulary_id": str, "invalid_reason": str|None}``. When the
    candidate is already standard there is no hop (`object_id` is None). Otherwise, prefers the
    real edge landing on `standard_id` (when known, i.e. the candidate passed); falls back to the
    candidate's first Identity edge so a failed candidate still shows where Identity mapping
    *would* have led. `invalid_reason` (None when current/active) lets callers tell a genuine,
    no-mapping-exists dead end apart from a deprecated concept whose mappings were dropped from
    the live vocabulary snapshot -- both render as "no Identity mapping found" but only the
    former is a clean illustrative example.
    """
    cv = kg.concept_view(candidate_id)
    base = {"vocabulary_id": cv.vocabulary_id, "invalid_reason": cv.invalid_reason}
    if cv.standard_concept:
        return {**base, "already_standard": True, "predicate_name": None, "object_id": None, "object_name": None}

    edges = kg.edges(
        concept_ids=candidate_id, direction="out",
        predicate_kinds=frozenset({PredicateKind.IDENTITY}),
        active_only=True, within_domain=False,
    )
    target_edge = None
    if standard_id is not None:
        target_edge = next((e for e in edges if e.object_id == standard_id), None)
    if target_edge is None and edges:
        target_edge = edges[0]
    if target_edge is None:
        return {**base, "already_standard": False, "predicate_name": None, "object_id": None, "object_name": None}

    object_name = kg.concept_view(target_edge.object_id).concept_name
    return {
        **base,
        "already_standard": False,
        "predicate_name": kg.predicate_name(target_edge.predicate_id),
        "object_id": target_edge.object_id,
        "object_name": object_name,
    }


def _select_funnel_candidates(trace: Dict, max_passed: int = 6, max_failed: int = 12) -> List[Dict]:
    """Pick a small, representative set of `hierarchy_validation` entries for the funnel
    diagram's starting (stage 1) column -- enough to show real candidates getting eliminated
    at each stage, without enumerating every candidate the resolvers found.

    Passed entries are deduplicated by their resulting standard concept first (so two
    candidates that converge on the same answer don't both eat a slot), then capped, always
    keeping whichever one resolves to the case's rank-1 winner. A few failed entries are added
    for contrast, preferring resolver diversity (Embedding-only fails are the most narratively
    useful failure mode for this figure, per the project's established preference).
    """
    hierarchy_validation = trace.get("hierarchy_validation", [])
    top_n_results = trace.get("top_n_results", [])
    winner_result = next((r for r in top_n_results if r.get("rank") == 1), None)
    winner_id = winner_result.get("concept_id") if winner_result else None

    passed = [e for e in hierarchy_validation if e.get("passed")]
    failed = [e for e in hierarchy_validation if not e.get("passed")]

    by_standard: Dict[int, List[Dict]] = {}
    for e in passed:
        by_standard.setdefault(e["standard_concept_id"], []).append(e)

    passed_reps = []
    for sid, entries in by_standard.items():
        # Prefer a representative that required a real Identity hop (more illustrative than a
        # candidate that already *is* the standard concept), unless that candidate is the hop
        # target itself with no alternative.
        rep = next((e for e in entries if e["candidate_concept_id"] != sid), entries[0])
        passed_reps.append(rep)

    winner_rep = next((e for e in passed_reps if e["standard_concept_id"] == winner_id), None)
    other_reps = sorted(
        (e for e in passed_reps if e is not winner_rep),
        key=lambda e: e.get("separation", 99),
    )
    selected_passed = ([winner_rep] if winner_rep else []) + other_reps
    selected_passed = selected_passed[:max_passed]

    failed_sorted = sorted(failed, key=lambda e: "Embedding" not in (e.get("resolvers") or [e.get("resolver")]))
    # Exclude failed candidates sharing a name with an already-selected passed one: these are
    # almost always the same surface form under a different (duplicate vocabulary source)
    # concept ID, which renders confusingly -- a "failed" lane whose live Identity-edge lookup
    # coincidentally lands on the very standard concept that the matching passed lane already
    # reached, since we don't know which of a failed candidate's edges the algorithm actually
    # tried, only that none of them worked.
    seen_names: set = {e.get("candidate_concept_name", "") for e in selected_passed}
    selected_failed = []
    for e in failed_sorted:
        name = e.get("candidate_concept_name", "")
        if name in seen_names:
            continue
        seen_names.add(name)
        selected_failed.append(e)
        if len(selected_failed) >= max_failed:
            break

    return selected_passed + selected_failed


def _build_funnel_nodes(kg: KnowledgeGraph, trace: Dict, entries: List[Dict]) -> List[Dict]:
    """Resolve each selected candidate's real Identity hop and attach resolver/score metadata,
    ready for the funnel renderer."""
    top_n_results = trace.get("top_n_results", [])
    winner_result = next((r for r in top_n_results if r.get("rank") == 1), None)
    winner_id = winner_result.get("concept_id") if winner_result else None
    score_by_id = {s["concept_id"]: s["total_score"] for s in trace.get("scoring", [])}

    nodes = []
    for entry in entries:
        candidate_id = entry["candidate_concept_id"]
        candidate_name = entry["candidate_concept_name"]
        passed = bool(entry.get("passed"))
        standard_id = entry.get("standard_concept_id") if passed else None
        hop = _resolve_identity_hop(kg, candidate_id, standard_id)

        if hop["already_standard"]:
            final_standard_id, final_standard_name = candidate_id, candidate_name
        elif standard_id is not None:
            final_standard_id, final_standard_name = standard_id, entry.get("standard_concept_name")
        elif hop["object_id"] is not None:
            final_standard_id, final_standard_name = hop["object_id"], hop["object_name"]
        else:
            final_standard_id, final_standard_name = None, None

        resolvers = entry.get("resolvers") or [entry.get("resolver")]
        assert isinstance(resolvers, list) and all(isinstance(r, str) for r in resolvers)
        nodes.append({
            "candidate_id": candidate_id,
            "candidate_name": candidate_name,
            "candidate_vocabulary_id": hop["vocabulary_id"],
            "candidate_invalid_reason": hop["invalid_reason"],
            "resolver_groups": _resolver_groups_for(resolvers),  # type: ignore
            "already_standard": hop["already_standard"],
            "edge_predicate": hop["predicate_name"],
            "standard_id": final_standard_id,
            "standard_name": final_standard_name,
            "passed": passed,
            "separation": entry.get("separation") if passed else None,
            "score": score_by_id.get(final_standard_id) if final_standard_id is not None else None,
            "is_winner": passed and final_standard_id is not None and final_standard_id == winner_id,
        })

    # Drop failed nodes whose live-Identity-edge fallback happens to land on a standard concept
    # that another lane independently confirmed *passes* the anchor -- internally contradictory
    # (if that edge were the real reason it failed, the matching passed lane wouldn't exist) and
    # confusing to render (a "culled" arrow pointing at a box another arrow shows as kept).
    passed_standard_ids = {n["standard_id"] for n in nodes if n["passed"]}
    nodes = [n for n in nodes if n["passed"] or n["standard_id"] not in passed_standard_ids]
    return nodes


def _trim_funnel_nodes(nodes: List[Dict], max_passed: int = 3, max_no_mapping: int = 1, max_hierarchy_fail: int = 1) -> List[Dict]:
    """Cut the built node list down to a small, clean set for the figure: the winner plus a
    couple of other genuinely-scored alternatives, and one example of each failure flavor (no
    Identity mapping at all vs. a real mapping that falls outside the hierarchy anchor) -- so
    the diagram doesn't grow tall with redundant or unscored rows."""
    winner = [n for n in nodes if n["is_winner"]]
    other_passed = sorted(
        (n for n in nodes if n["passed"] and not n["is_winner"] and n["score"] is not None),
        key=lambda n: n["score"],
        reverse=True,
    )
    selected_passed = (winner + other_passed)[:max_passed]

    # Prefer a currently-active concept for the "no mapping" example: a deprecated/updated
    # concept's missing Identity edge is often just a side-effect of the vocabulary refresh
    # dropping mappings for invalid concepts, not a genuine "this never maps anywhere" dead end.
    no_mapping = sorted(
        (n for n in nodes if not n["passed"] and n["standard_id"] is None),
        key=lambda n: n["candidate_invalid_reason"] is not None,
    )[:max_no_mapping]
    hierarchy_fail = [n for n in nodes if not n["passed"] and n["standard_id"] is not None][:max_hierarchy_fail]

    return selected_passed + no_mapping + hierarchy_fail


def _resolve_anchor_concepts(kg: KnowledgeGraph, trace: Dict) -> List[Dict]:
    """Resolve every hierarchy-anchor parent ID's concept name (not just the first) for the
    funnel diagram's top banner."""
    parent_ids = trace.get("constraints", {}).get("parent_ids") or []
    if not parent_ids:
        return []
    views = kg.concept_views(tuple(parent_ids))
    name_by_id = {v.concept_id: v.concept_name for v in views}
    return [{"concept_id": pid, "concept_name": name_by_id.get(pid, f"[concept {pid}]")} for pid in parent_ids]


def _funnel_svg(nodes: List[Dict], trace: Dict, anchor_concepts: List[Dict]) -> str:
    """Render the 4-stage left-to-right funnel: candidates -> standard concepts (Identity hop)
    -> hierarchy-validated -> scored/winner. Candidates are stacked vertically; nodes that
    converge on the same standard concept merge into one box in later stages. Eliminated
    candidates gray out and dead-end exactly at the stage where they're culled."""
    MARGIN = 30
    COL_W, COL_GAP = 230, 70
    NODE_H, NODE_GAP = 78, 22
    NAME_MAX_CHARS = 28
    FONT = "system-ui, -apple-system, Arial, sans-serif"

    BODY_BG = "#f8f9fa"
    WIN_BG, WIN_BORDER = "#d4edda", "#28a745"
    PASS_BG, PASS_BORDER = "#eaf2ff", "#5b8def"
    FAIL_BG, FAIL_BORDER = "#f8d7da", "#b02a37"
    NEUTRAL_BORDER = "#ced4da"
    ANCHOR_BG, ANCHOR_BORDER = "#fff3cd", "#856404"
    SEARCH_BG, SEARCH_BORDER = "#e9ecef", "#495057"
    TEXT_DARK, TEXT_MID, TEXT_LIGHT = "#212529", "#495057", "#6c757d"
    FS_XS, FS_SM, FS_MID = 10, 12, 13
    SEARCH_BOX_Y, SEARCH_BOX_H = 10, 30
    ANCHOR_LINE_H = 16

    W = MARGIN * 2 + COL_W * 4 + COL_GAP * 3

    def wrap_words(s, max_chars, max_lines=2):
        """Greedy word-wrap into up to `max_lines` lines; overflow gets an ellipsis. Pads with
        empty strings up to `max_lines` so callers can index positionally."""
        words = str(s).split()
        lines, cur = [], ""
        for w in words:
            trial = f"{cur} {w}".strip()
            if len(trial) <= max_chars or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            lines[-1] = lines[-1].rstrip() + "…"
        while len(lines) < max_lines:
            lines.append("")
        return lines

    # ── Hierarchy anchor banner sizing (computed up front so the grid leaves it enough room) ──
    ANCHOR_LABEL = "Hierarchy anchor: "
    if anchor_concepts:
        anchor_value = ", ".join(f'{a["concept_name"]} [{a["concept_id"]}]' for a in anchor_concepts)
    else:
        anchor_value = "—"
    anchor_chars_per_line = max(20, int((W - 2 * MARGIN - 24) / 7.3))
    anchor_lines = [line for line in wrap_words(ANCHOR_LABEL + anchor_value, anchor_chars_per_line, max_lines=4) if line] or ["—"]
    ANCHOR_BOX_H = 16 + len(anchor_lines) * ANCHOR_LINE_H

    # Anchor banner sits directly under the search-term banner with no gap; headers/grid start
    # below it, so a longer (multi-row) anchor list pushes the whole grid down rather than
    # overflowing.
    ANCHOR_BOX_Y = SEARCH_BOX_Y + SEARCH_BOX_H
    HEADER_Y = ANCHOR_BOX_Y + ANCHOR_BOX_H + 24
    TOP_Y = HEADER_Y + 9
    BOTTOM_MARGIN = 7        # gap below the "Found by" legend to the canvas edge
    LEGEND_RECT_H = 14       # the colored resolver-swatch height in the legend
    GRID_LEGEND_GAP = 16     # gap between the last candidate row and the "Found by" legend
    FOOTER_RESERVED = GRID_LEGEND_GAP + LEGEND_RECT_H + BOTTOM_MARGIN

    n = max(len(nodes), 1)
    total_h = n * NODE_H + (n - 1) * NODE_GAP
    H = max(420, TOP_Y + total_h + FOOTER_RESERVED)

    parts: List[str] = []

    def rect(x, yy, w, h, fill, stroke, rx=6, sw=1.5, dashed=False):
        dash = ' stroke-dasharray="6,4"' if dashed else ""
        return f'<rect x="{x:.1f}" y="{yy:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{dash}/>'

    def text(x, yy, s, fs=FS_MID, fill=TEXT_DARK, anchor="start", bold=False):
        fw = "bold" if bold else "normal"
        return f'<text x="{x:.1f}" y="{yy:.1f}" font-family="{FONT}" font-size="{fs}" fill="{fill}" text-anchor="{anchor}" font-weight="{fw}">{escape(str(s))}</text>'

    def text_label_value(x, yy, label, value, fs=FS_SM, fill=TEXT_DARK):
        """A bold label followed by a non-bold value, e.g. 'Search term: <query>'."""
        return (
            f'<text x="{x:.1f}" y="{yy:.1f}" font-family="{FONT}" font-size="{fs}" fill="{fill}" text-anchor="start">'
            f'<tspan font-weight="bold">{escape(str(label))}</tspan>'
            f'<tspan font-weight="normal">{escape(str(value))}</tspan>'
            f'</text>'
        )

    def split_label(label):
        """Split an edge label into two halves (by word count) for above/below the arrow."""
        if not label:
            return None
        words = label.split(" ")
        if len(words) == 1:
            return (label, None)
        mid = (len(words) + 1) // 2
        return (" ".join(words[:mid]), " ".join(words[mid:]))

    def arrow(x1, y1, x2, y2, color, dashed=False, bold=False, label=None):
        sw = 2.5 if bold else 1.5
        dash = ' stroke-dasharray="5,4"' if dashed else ""
        dx, dy = x2 - x1, y2 - y1
        dist = max(1.0, (dx * dx + dy * dy) ** 0.5)
        ux, uy = dx / dist, dy / dist
        head = 8
        lx2, ly2 = x2 - ux * head, y2 - uy * head
        out = [f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{lx2:.1f}" y2="{ly2:.1f}" stroke="{color}" stroke-width="{sw}"{dash}/>']
        px, py = -uy, ux
        p1 = (x2 - ux * head + px * 4, y2 - uy * head + py * 4)
        p2 = (x2 - ux * head - px * 4, y2 - uy * head - py * 4)
        out.append(f'<polygon points="{p1[0]:.1f},{p1[1]:.1f} {p2[0]:.1f},{p2[1]:.1f} {x2:.1f},{y2:.1f}" fill="{color}"/>')
        if label:
            mx, my = (x1 + lx2) / 2, (y1 + ly2) / 2
            if isinstance(label, tuple):
                l1, l2 = label
                out.append(text(mx, my - 6, l1, FS_XS, color, "middle", bold=bold))
                if l2:
                    out.append(text(mx, my + 10, l2, FS_XS, color, "middle", bold=bold))
            else:
                out.append(text(mx, my - 4, label, FS_XS, color, "middle", bold=bold))
        return "\n".join(out)

    def stripe(x, yy, w, groups):
        """Segmented resolver-attribution stripe along a box's top edge."""
        if not groups:
            return ""
        seg_w = w / len(groups)
        out = []
        for i, g in enumerate(groups):
            color = RESOLVER_STRIPE_COLORS.get(g, TEXT_LIGHT)
            out.append(f'<rect x="{x + i*seg_w:.1f}" y="{yy:.1f}" width="{seg_w:.1f}" height="6" fill="{color}"/>')
        return "\n".join(out)

    def name_id_box(x, y, name, cid):
        """Box content: name + '[concept_id]' in **at most 2 lines, always**. The name fills
        line 1 (word-wrapped); '[concept_id]' is appended to line 2 if there's room, otherwise
        line 2 is the name's remaining words truncated (with an ellipsis) to make room for it --
        so the ID is always visible without ever needing a 3rd line. Returns (parts, next_y),
        the y-coordinate immediately below the rendered lines, for any additional content a
        caller wants to add beneath the name."""
        id_str = f"[{cid}]"
        words = str(name).split()
        line1, i = "", 0
        for w in words:
            trial = f"{line1} {w}".strip()
            if len(trial) <= NAME_MAX_CHARS or not line1:
                line1, i = trial, i + 1
            else:
                break
        remaining = " ".join(words[i:])
        if not remaining:
            if len(f"{line1} {id_str}") <= NAME_MAX_CHARS:
                line1, line2 = f"{line1} {id_str}", None
            else:
                line2 = id_str
        elif len(f"{remaining} {id_str}") <= NAME_MAX_CHARS:
            line2 = f"{remaining} {id_str}"
        else:
            budget = max(0, NAME_MAX_CHARS - len(id_str) - 2)
            truncated = remaining[:budget].rstrip()
            line2 = f"{truncated}… {id_str}" if truncated else f"…{id_str}"
        name_lines = [line1] + ([line2] if line2 else [])

        out = []
        yy = y + 24
        for line in name_lines:
            out.append(text(x + 10, yy, line, FS_SM, TEXT_DARK, bold=True))
            yy += 16
        return out, yy

    def outcome_colors(passed, winner):
        if winner:
            return WIN_BG, WIN_BORDER
        if passed:
            return PASS_BG, PASS_BORDER
        return FAIL_BG, FAIL_BORDER

    col_xs = [MARGIN + i * (COL_W + COL_GAP) for i in range(4)]
    headers = ["CANDIDATES", "STANDARD CONCEPT", "HIERARCHY-VALIDATED", "SCORED / WINNER"]

    # ── Search-term banner (full width, at the very top) ──
    query = trace.get("query") or "—"
    parts.append(rect(MARGIN, SEARCH_BOX_Y, W - 2 * MARGIN, SEARCH_BOX_H, SEARCH_BG, SEARCH_BORDER, rx=6, sw=1.5))
    parts.append(text_label_value(
        MARGIN + 12, SEARCH_BOX_Y + SEARCH_BOX_H / 2 + 4, "Search term: ", f'"{query}"', FS_SM, SEARCH_BORDER,
    ))

    # ── Hierarchy anchor banner (full width, directly below the search-term banner, no gap) ──
    parts.append(rect(MARGIN, ANCHOR_BOX_Y, W - 2 * MARGIN, ANCHOR_BOX_H, ANCHOR_BG, ANCHOR_BORDER, rx=6, sw=1.5))
    yy = ANCHOR_BOX_Y + 8 + 12  # 8px top padding + ~12px baseline-from-top for a 12px font
    for i, line in enumerate(anchor_lines):
        if i == 0 and line.startswith(ANCHOR_LABEL):
            parts.append(text_label_value(MARGIN + 12, yy, ANCHOR_LABEL, line[len(ANCHOR_LABEL):], FS_SM, ANCHOR_BORDER))
        else:
            parts.append(text(MARGIN + 12, yy, line, FS_SM, ANCHOR_BORDER))
        yy += ANCHOR_LINE_H

    for i, hdr in enumerate(headers):
        parts.append(text(col_xs[i] + COL_W / 2, HEADER_Y, hdr, FS_SM, TEXT_LIGHT, "middle", bold=True))

    n_nodes = len(nodes)
    start_y = TOP_Y + max(0, (H - TOP_Y - FOOTER_RESERVED - total_h) / 2)
    node_y = {i: start_y + i * (NODE_H + NODE_GAP) for i in range(n_nodes)}
    node_cy = {i: node_y[i] + NODE_H / 2 for i in range(n_nodes)}

    # ── Stage 1: candidates ──
    for i, node in enumerate(nodes):
        x, y = col_xs[0], node_y[i]
        parts.append(rect(x, y, COL_W, NODE_H, BODY_BG, NEUTRAL_BORDER, rx=6, sw=1.5))
        parts.append(stripe(x, y, COL_W, node["resolver_groups"]))
        box_parts, next_y = name_id_box(x, y, node["candidate_name"], node["candidate_id"])
        parts.extend(box_parts)
        parts.append(text(x + 10, next_y, f'Vocabulary: {node["candidate_vocabulary_id"]}', FS_XS, TEXT_LIGHT))

    # ── Stage 1 -> 2: Identity hop (or dead end if no mapping at all) ──
    std_order: List[int] = []
    std_indices: Dict[int, List[int]] = {}
    for i, node in enumerate(nodes):
        sid = node["standard_id"]
        if sid is None:
            x1, y1 = col_xs[0] + COL_W, node_cy[i]
            parts.append(arrow(x1, y1, x1 + 26, y1, FAIL_BORDER, dashed=True))
            parts.append(text(x1 + 32, y1 + 4, "✗ no Identity mapping found", FS_XS, FAIL_BORDER, bold=True))
            continue
        std_indices.setdefault(sid, [])
        if sid not in std_order:
            std_order.append(sid)
        std_indices[sid].append(i)

    std_cy = {sid: sum(node_cy[i] for i in idxs) / len(idxs) for sid, idxs in std_indices.items()}
    std_passed = {sid: any(nodes[i]["passed"] for i in idxs) for sid, idxs in std_indices.items()}
    std_winner = {sid: any(nodes[i]["is_winner"] for i in idxs) for sid, idxs in std_indices.items()}
    std_name = {sid: next(nodes[i]["standard_name"] for i in idxs) for sid, idxs in std_indices.items()}
    std_sep = {
        sid: min((nodes[i]["separation"] for i in idxs if nodes[i]["separation"] is not None), default=None)
        for sid, idxs in std_indices.items()
    }
    std_score = {
        sid: next((nodes[i]["score"] for i in idxs if nodes[i]["score"] is not None), None)
        for sid, idxs in std_indices.items()
    }
    std_edge_label = {}
    for sid, idxs in std_indices.items():
        rep = nodes[idxs[0]]
        std_edge_label[sid] = "Already standard (no hop)" if rep["already_standard"] else (rep["edge_predicate"] or "Identity mapping")

    for i, node in enumerate(nodes):
        sid = node["standard_id"]
        if sid is None:
            continue
        x1, y1 = col_xs[0] + COL_W, node_cy[i]
        x2, y2 = col_xs[1], std_cy[sid]
        color = WIN_BORDER if node["is_winner"] else (PASS_BORDER if node["passed"] else FAIL_BORDER)
        parts.append(arrow(x1, y1, x2, y2, color, dashed=not node["passed"], bold=node["is_winner"]))

    for sid in std_order:
        x, y = col_xs[1], std_cy[sid] - NODE_H / 2
        bg, border = outcome_colors(std_passed[sid], std_winner[sid])
        parts.append(rect(x, y, COL_W, NODE_H, bg, border, rx=6, sw=2 if std_winner[sid] else 1.5, dashed=not std_passed[sid]))
        box_parts, next_y = name_id_box(x, y, std_name[sid], sid)
        parts.extend(box_parts)
        parts.append(text(x + 10, next_y, f'via: {std_edge_label[sid]}', FS_XS, TEXT_LIGHT))

    # ── Stage 2 -> 3: hierarchy validation (kept vs. culled) ──
    for sid in std_order:
        x1, y1 = col_xs[1] + COL_W, std_cy[sid]
        if not std_passed[sid]:
            parts.append(arrow(x1, y1, x1 + 26, y1, FAIL_BORDER, dashed=True))
            parts.append(text(x1 + 32, y1 + 4, "✗ outside anchor", FS_XS, FAIL_BORDER, bold=True))
            continue
        x2 = col_xs[2]
        color = WIN_BORDER if std_winner[sid] else PASS_BORDER
        parts.append(arrow(x1, y1, x2, y1, color, bold=std_winner[sid]))
        bg, border = outcome_colors(True, std_winner[sid])
        parts.append(rect(x2, y1 - NODE_H / 2, COL_W, NODE_H, bg, border, rx=6, sw=2 if std_winner[sid] else 1.5))
        ccx = x2 + COL_W / 2
        parts.append(text(ccx, y1 - 4, "✓ Passed hierarchy constraint", FS_SM, border, "middle", bold=True))
        dist_str = f'Distance to Hierarchy Anchor: {std_sep[sid]}' if std_sep[sid] is not None else ""
        parts.append(text(ccx, y1 + 14, dist_str, FS_XS, TEXT_LIGHT, "middle"))

    # ── Stage 3 -> 4: scoring (final rank) ──
    for sid in std_order:
        if not std_passed[sid]:
            continue
        y = std_cy[sid]
        x1 = col_xs[2] + COL_W
        x2 = col_xs[3]
        color = WIN_BORDER if std_winner[sid] else PASS_BORDER
        parts.append(arrow(x1, y, x2, y, color, bold=std_winner[sid]))
        bg, border = outcome_colors(True, std_winner[sid])
        box_y = y - NODE_H / 2
        parts.append(rect(x2, box_y, COL_W, NODE_H, bg, border, rx=6, sw=2 if std_winner[sid] else 1.5))
        box_parts, next_y = name_id_box(x2, box_y, std_name[sid], sid)
        parts.extend(box_parts)
        score_str = f'{std_score[sid]:.3f}' if std_score[sid] is not None else "n/a"
        if std_winner[sid]:
            status = f"★ WINNER (rank 1), Score: {score_str}"
        else:
            status = f"Score: {score_str}"
        parts.append(text(x2 + 10, next_y, status, FS_XS, WIN_BORDER if std_winner[sid] else TEXT_LIGHT, bold=std_winner[sid]))

    # ── Legend: resolver-attribution stripe colors ──
    legend_y = H - BOTTOM_MARGIN - 4
    lx = MARGIN
    parts.append(text(lx, legend_y + 1, "Found by:", FS_XS, TEXT_MID, bold=True))
    lx += 58
    for g in ("Exact", "Partial", "Full Text", "Embedding"):
        color = RESOLVER_STRIPE_COLORS[g]
        parts.append(f'<rect x="{lx}" y="{legend_y-10}" width="14" height="14" rx="3" fill="{color}"/>')
        parts.append(text(lx + 20, legend_y + 1, g, FS_XS, TEXT_MID))
        lx += 9 * len(g) + 50

    svg_body = "\n".join(parts)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
  <rect width="{W}" height="{H}" fill="white"/>
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
    parent_id_level: Annotated[Optional[int], typer.Option("--parent-id-level", "-L",
        help="Use case['parent_ids_by_level'][N] as the anchor instead of case['parent_ids'] "
             "(see enrich_gold_standard.py). Falls back to case['parent_ids'] for cases/files "
             "without per-level data. Also nests output under parent_id_level_<N>/.")] = None,
):
    """Trace grounding cases through every resolver stage. Outputs {\"cases\": [...]}."""
    # Build the list of cases to run
    cases_to_run: List[Dict] = []
    
    if cases_file:
        payload = json.loads(Path(cases_file).read_text())
        rows = payload if isinstance(payload, list) else [r for bucket in payload.values() for r in bucket]
        if case_ids:
            case_ids_str = {str(cid) for cid in case_ids}
            rows = [r for r in rows if str(r.get("id")) in case_ids_str]
            missing = case_ids_str - {str(r.get("id")) for r in rows}
            for mid in sorted(missing):
                console.print(f"Warning: case '{mid}' not found in {cases_file}.")
        cases_to_run = rows
    elif query:
        if not parent_ids:
            console.print("--parent-ids required with --query.")
            raise typer.Exit(1)
        cases_to_run = [{
            "id": "on-demand",
            "text": query,
            "expected_concept_id": query_expected_concept_id,
            "expected_concept_name": None,
        }]
    else:
        console.print("Provide --cases-file or --query.")
        raise typer.Exit(1)

    if not cases_to_run:
        console.print("No cases to run.")
        raise typer.Exit(1)

    console.print("Building knowledge graph…")
    from omop_emb.config import parse_metric_type
    kg = _build_kg(metric_type=parse_metric_type(metric_type), embedding_model=embedding_model)
    used_embedding_model = (
        kg.embedding_configuration.model_name if kg.embedding_configuration else None
    )

    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("Tracing cases…", total=len(cases_to_run))
        for case in cases_to_run:
            case_id = case.get("id")
            case_query = case.get("text")
            case_domains = domains or ([case.get("domain")] if case.get("domain") else None)
            case_vocab = vocabularies or ([case.get("vocabulary")] if case.get("vocabulary") else None)
            if parent_ids:
                case_parent_ids = parent_ids
            elif parent_id_level is not None:
                by_level = case.get("parent_ids_by_level")
                if by_level is None:
                    console.print(f"Case '{case_id}': no parent_ids_by_level, falling back to parent_ids.")
                    case_parent_ids = case.get("parent_ids")
                else:
                    case_parent_ids = by_level.get(str(parent_id_level))
            else:
                case_parent_ids = case.get("parent_ids")
            case_expected = query_expected_concept_id or case.get("expected_concept_id")
            case_expected_name = case.get("expected_concept_name")

            if case_query is None:
                console.print(f"Case '{case_id}': no query text, skipping.")
                progress.advance(task_id)
                continue
            if case_id is None:
                console.print(f"Case with query '{case_query}': no case ID, skipping.")
                progress.advance(task_id)
                continue
            if case_parent_ids is None:
                console.print(f"Case '{case_id}': no parent IDs, skipping.")
                progress.advance(task_id)
                continue

            search_constraint = SearchConstraintConcept(
                domains=tuple([str(c) for c in case_domains]) if isinstance(case_domains, list) else None,
                vocabularies=tuple([str(v) for v in case_vocab]) if isinstance(case_vocab, list) else None,
                require_standard=False,
            )

            progress.update(task_id, description=f"[{case_id}] '{case_query[:40]}'")
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
            progress.advance(task_id)

    hits = [r for r in results if r.get("target_rank") == 1]

    if out_dir:
        out_dir_path = Path(out_dir) / _model_dir_name(used_embedding_model)
        if parent_id_level is not None:
            out_dir_path = out_dir_path / f"parent_id_level_{parent_id_level}"
        out_dir_path.mkdir(parents=True, exist_ok=True)
        for result in results:
            cid = result["case_id"]
            case_output = json.dumps(
                {
                    "cases": [result],
                    "embedding_model": used_embedding_model,
                    "parent_id_level": parent_id_level,
                },
                indent=2,
                ensure_ascii=False,
            )
            (out_dir_path / f"trace_{cid}.json").write_text(case_output, encoding="utf-8")

        console.print(f"Trace written to {out_dir_path} ({len(results)} case(s)).")
        console.print(f"Target rank 1: {len(hits)}/{len(results)}")
    else:
        output = json.dumps(
            {"cases": results, "embedding_model": used_embedding_model},
            indent=2,
            ensure_ascii=False,
        )
        print(output)


@app.command(name="pipeline-svg")
def pipeline_svg(
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
    parent_id_level: Annotated[Optional[int], typer.Option("--parent-id-level", "-L",
        help="Same --parent-id-level passed to 'trace' -- resolves the matching "
             "parent_id_level_<N>/ subdirectory.")] = None,
):
    """Generate resolver-pipeline flowchart SVG(s) from trace JSON files written by the trace
    command. Writes them to the plots/pipeline/ subdirectory of the same directory as the traces."""
    resolved_model = _resolve_embedding_model(embedding_model)
    trace_dir_pl = Path(trace_dir) / _model_dir_name(resolved_model)
    if parent_id_level is not None:
        trace_dir_pl = trace_dir_pl / f"parent_id_level_{parent_id_level}"

    json_files = sorted(trace_dir_pl.glob("*.json"))
    if not json_files:
        typer.echo(f"No JSON trace files found in {trace_dir_pl}.", err=True)
        raise typer.Exit(1)
    cases: List[Dict] = []
    for json_file in json_files:
        payload = json.loads(json_file.read_text(encoding="utf-8"))
        cases.extend(payload.get("cases", [payload]))

    if case_id and case_id.lower() != "all":
        selected = [c for c in cases if str(c.get("case_id") or "ad-hoc") == case_id]
        if not selected:
            typer.echo(f"Case '{case_id}' not found in {trace_dir_pl}.", err=True)
            raise typer.Exit(1)
    else:
        selected = cases

    out_dir = trace_dir_pl / "plots" / "pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)
    for trace_data in selected:
        cid = trace_data.get("case_id") or "ad-hoc"
        svg_content = _svg(trace_data, title)
        out_file = out_dir / f"trace_{cid}.svg"
        out_file.write_text(svg_content, encoding="utf-8")
        typer.echo(f"SVG written to {out_file}")


@app.command(name="panel-svg")
def panel_svg(
    trace_dir: Annotated[str, typer.Option("--trace-dir", "-t",
        help="Base output directory passed to 'trace --out-dir'. The model-specific "
             "subdirectory is resolved the same way 'trace' does, via --embedding-model "
             "or config.toml.")],
    embedding_model: Annotated[Optional[str], typer.Option("--embedding-model", "-E",
        help="Embedding model name used to resolve the subdirectory, overriding "
             "config.toml (same semantics as 'trace --embedding-model').")] = None,
    case_id: Annotated[Optional[str], typer.Option("--case-id", "-i",
        help="Case ID to render. Omitted, empty, or 'all' renders every case found.")] = None,
    concept_id: Annotated[Optional[int], typer.Option("--concept-id",
        help="Override the concept whose outgoing edges Panel A classifies. Defaults to the "
             "case's winning (target) concept. Only meaningful when rendering a single case.")] = None,
    title: Annotated[str, typer.Option("--title",
        help="Figure title (kept for CLI parity with pipeline-svg; not currently rendered).")] = "omop-graph Graph Traversal",
    parent_id_level: Annotated[Optional[int], typer.Option("--parent-id-level", "-L",
        help="Same --parent-id-level passed to 'trace' -- resolves the matching "
             "parent_id_level_<N>/ subdirectory.")] = None,
):
    """Generate the Panel A/B dashboard SVG(s): Panel A shows a concept's outgoing edges grouped
    by relationship class (Identity = walkable for grounding vs. everything else = excluded by
    default); Panel B shows real candidates that passed or failed the hierarchy anchor, pulled
    from the trace JSON's existing `hierarchy_validation` data. Writes them to the plots/panel/
    subdirectory of the same directory as the traces."""
    resolved_model = _resolve_embedding_model(embedding_model)
    trace_dir_pl = Path(trace_dir) / _model_dir_name(resolved_model)
    if parent_id_level is not None:
        trace_dir_pl = trace_dir_pl / f"parent_id_level_{parent_id_level}"

    json_files = sorted(trace_dir_pl.glob("*.json"))
    if not json_files:
        typer.echo(f"No JSON trace files found in {trace_dir_pl}.", err=True)
        raise typer.Exit(1)
    cases: List[Dict] = []
    for json_file in json_files:
        payload = json.loads(json_file.read_text(encoding="utf-8"))
        cases.extend(payload.get("cases", [payload]))

    if case_id and case_id.lower() != "all":
        selected = [c for c in cases if str(c.get("case_id") or "ad-hoc") == case_id]
        if not selected:
            typer.echo(f"Case '{case_id}' not found in {trace_dir_pl}.", err=True)
            raise typer.Exit(1)
    else:
        selected = cases

    kg = KnowledgeGraph(cdm_engine=make_engine())
    rel_class_cache: Dict[int, Dict] = {}

    out_dir = trace_dir_pl / "plots" / "panel"
    out_dir.mkdir(parents=True, exist_ok=True)
    for trace_data in selected:
        cid = trace_data.get("case_id") or "ad-hoc"
        target_result = next((r for r in trace_data.get("top_n_results", []) if r.get("is_target")), None)
        panel_a_concept_id = concept_id or (target_result.get("concept_id") if target_result else None)
        if panel_a_concept_id is None:
            typer.echo(f"Case '{cid}': no winning target concept to classify edges for, skipping.", err=True)
            continue
        if panel_a_concept_id not in rel_class_cache:
            rel_class_cache[panel_a_concept_id] = _relationship_classification(kg, panel_a_concept_id)
        svg_content = _panel_svg(rel_class_cache[panel_a_concept_id], trace_data, title)
        out_file = out_dir / f"panel_{cid}.svg"
        out_file.write_text(svg_content, encoding="utf-8")
        typer.echo(f"SVG written to {out_file}")


@app.command(name="graph-svg")
def graph_svg(
    trace_dir: Annotated[str, typer.Option("--trace-dir", "-t",
        help="Base output directory passed to 'trace --out-dir'. The model-specific "
             "subdirectory is resolved the same way 'trace' does, via --embedding-model "
             "or config.toml.")],
    embedding_model: Annotated[Optional[str], typer.Option("--embedding-model", "-E",
        help="Embedding model name used to resolve the subdirectory, overriding "
             "config.toml (same semantics as 'trace --embedding-model').")] = None,
    case_id: Annotated[Optional[str], typer.Option("--case-id", "-i",
        help="Case ID to render. Omitted, empty, or 'all' renders every case found.")] = None,
    parent_id_level: Annotated[Optional[int], typer.Option("--parent-id-level", "-L",
        help="Same --parent-id-level passed to 'trace' -- resolves the matching "
             "parent_id_level_<N>/ subdirectory.")] = None,
):
    """Generate the 4-stage funnel diagram SVG(s): candidates (stacked vertically, left, each
    striped by which resolver(s) found it) -> standard concepts (via a real, live-looked-up
    Identity edge, or dead-ending right here if no Identity mapping exists) -> hierarchy-
    validated (kept if a path to the anchor exists within depth, culled otherwise) -> scored,
    with the rank-1 winner highlighted green. Writes them to the plots/graph/ subdirectory of
    the same directory as the traces."""
    resolved_model = _resolve_embedding_model(embedding_model)
    trace_dir_pl = Path(trace_dir) / _model_dir_name(resolved_model)
    if parent_id_level is not None:
        trace_dir_pl = trace_dir_pl / f"parent_id_level_{parent_id_level}"

    json_files = sorted(trace_dir_pl.glob("*.json"))
    if not json_files:
        typer.echo(f"No JSON trace files found in {trace_dir_pl}.", err=True)
        raise typer.Exit(1)
    cases: List[Dict] = []
    for json_file in json_files:
        payload = json.loads(json_file.read_text(encoding="utf-8"))
        cases.extend(payload.get("cases", [payload]))

    if case_id and case_id.lower() != "all":
        selected = [c for c in cases if str(c.get("case_id") or "ad-hoc") == case_id]
        if not selected:
            typer.echo(f"Case '{case_id}' not found in {trace_dir_pl}.", err=True)
            raise typer.Exit(1)
    else:
        selected = cases

    kg = KnowledgeGraph(cdm_engine=make_engine())

    out_dir = trace_dir_pl / "plots" / "graph"
    out_dir.mkdir(parents=True, exist_ok=True)
    for trace_data in selected:
        cid = trace_data.get("case_id") or "ad-hoc"
        candidates = _select_funnel_candidates(trace_data)
        if not candidates:
            typer.echo(f"Case '{cid}': no resolver hits to build a funnel diagram from, skipping.", err=True)
            continue
        nodes = _trim_funnel_nodes(_build_funnel_nodes(kg, trace_data, candidates))
        anchor_concepts = _resolve_anchor_concepts(kg, trace_data)
        svg_content = _funnel_svg(nodes, trace_data, anchor_concepts)
        out_file = out_dir / f"graph_{cid}.svg"
        out_file.write_text(svg_content, encoding="utf-8")
        typer.echo(f"SVG written to {out_file}")


if __name__ == "__main__":
    app()
