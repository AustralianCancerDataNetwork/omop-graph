"""Grounding-focused benchmark tailored for poster/showcase outputs.

This script evaluates configuration ablations using ``ground_term`` end-to-end
(resolver + hierarchy anchoring + scored ranking) and surfaces representative
case-level improvements for communication in poster figures/tables.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from omop_llm import LLMClient

from omop_graph.extensions.emb import EmbeddingBackendType, MissingExtensionError
from omop_graph.extensions.omop_alchemy import ClassIDEnum
from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.reasoning.grounding import GroundingConstraints, ground_term
from omop_graph.reasoning.resolvers.resolver_pipeline import ResolverPipeline
from omop_graph.reasoning.resolvers.resolvers import (
    CandidateResolver,
    EmbeddingResolver,
    ExactLabelResolver,
    ExactSynonymResolver,
    FullTextResolver,
    FullTextSynonymResolver,
    PartialLabelResolver,
    PartialSynonymResolver,
)
from benchmark_base import (  # type: ignore
    BenchmarkCase,
    build_embedding_knowledge_graph,
    build_knowledge_graph,
    case_constraints,
    load_cases,
    mcnemar,
    ranking_metrics,
)


@dataclass(frozen=True)
class PosterConfig:
    """One ablation configuration for grounding-based evaluation."""

    name: str
    resolvers: Tuple[CandidateResolver, ...]
    requires_embedding: bool = False


def _build_configs() -> Tuple[PosterConfig, ...]:
    """Build resolver ablations for grounded benchmarking."""

    basic = (
        ExactLabelResolver(),
        ExactSynonymResolver(),
    )
    extended = (
        *basic,
        PartialLabelResolver(),
        PartialSynonymResolver(),
    )
    full_text = (
        *extended,
        FullTextResolver(),
        FullTextSynonymResolver(),
    )
    full_text_with_embedding = (
        *full_text,
        EmbeddingResolver(),
    )

    return (
        PosterConfig(name="basic", resolvers=basic),
        PosterConfig(name="extended", resolvers=extended),
        PosterConfig(name="full_text", resolvers=full_text),
        PosterConfig(
            name="full_text_with_embedding",
            resolvers=full_text_with_embedding,
            requires_embedding=True,
        ),
    )


def _resolve_parent_ids(
    case: BenchmarkCase,
    default_parent_ids: Optional[Tuple[int, ...]],
) -> Optional[Tuple[int, ...]]:
    """Resolve grounding parent IDs from case-level or CLI-level defaults."""

    if case.parent_ids is not None:
        return case.parent_ids
    return default_parent_ids


def _evaluate_case(
    kg: KnowledgeGraph,
    case: BenchmarkCase,
    config: PosterConfig,
    k: int,
    default_parent_ids: Optional[Tuple[int, ...]],
    resolver_kwargs: Optional[Dict[str, Any]] = None,
    grounding_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, float | int | bool | str]:
    """Evaluate one case using ``ground_term`` and return poster-friendly metrics."""

    search_constraint = case_constraints(case)
    parent_ids = _resolve_parent_ids(case, default_parent_ids)
    if parent_ids is None:
        raise ValueError(
            f"Case '{case.id}' has no parent_ids. Provide case.parent_ids or --grounding-parent-id."
        )

    resolver_pipeline = ResolverPipeline(resolvers=config.resolvers)
    resolver_kwargs = resolver_kwargs or {}
    grounding_kwargs = grounding_kwargs or {}

    grounded = ground_term(
        resolver_pipeline=resolver_pipeline,
        kg=kg,
        text=case.text,
        text_embedding=grounding_kwargs.get("text_embedding"),
        text_embedding_model=grounding_kwargs.get("text_embedding_model"),
        embedding_client=grounding_kwargs.get("embedding_client"),
        constraints=GroundingConstraints(
            parent_ids=parent_ids,
            search_constraint=search_constraint,
            max_depth=6,
            predicate_kinds=frozenset({ClassIDEnum.IDENTITY}),
        ),
        max_candidates=None,
        metric_type=grounding_kwargs.get("metric_type"),
        index_type=grounding_kwargs.get("index_type"),
    )

    predictions = [sc.concept_id for sc in grounded]
    expected = case.expected_concept_id
    metrics = ranking_metrics(predictions=predictions, expected=expected, k=k)

    false_grounding = 0.0
    safe_null = 0.0
    if expected is None:
        safe_null = 1.0 if len(predictions) == 0 else 0.0
        false_grounding = 1.0 if len(predictions) > 0 else 0.0
    elif expected not in predictions and len(predictions) > 0:
        false_grounding = 1.0

    target_rank = 0
    target_total_score = 0.0
    target_relevance = 0.0
    target_embedding_score = 0.0
    if expected is not None and expected in predictions:
        target_rank = predictions.index(expected) + 1
        target = grounded[target_rank - 1]
        target_total_score = float(target.total_score)
        target_relevance = float(target.relevance)
        target_embedding_score = (
            float(target.embedding_score) if target.embedding_score is not None else 0.0
        )

    top1_concept_id = predictions[0] if predictions else -1
    top1_total_score = float(grounded[0].total_score) if grounded else 0.0

    return {
        "case_id": case.id,
        "text": case.text,
        "bucket": case.bucket,
        "config": config.name,
        "expected": -1 if expected is None else expected,
        "expected_concept_name": case.expected_concept_name or "",
        "pred_count": len(predictions),
        "top1_correct": metrics["top1_correct"],
        "mrr": metrics["mrr"],
        "recall_at_k": metrics["recall_at_k"],
        "false_grounding": false_grounding,
        "safe_null": safe_null,
        "target_rank": float(target_rank),
        "target_total_score": target_total_score,
        "target_relevance": target_relevance,
        "target_embedding_score": target_embedding_score,
        "top1_concept_id": float(top1_concept_id),
        "top1_total_score": top1_total_score,
    }


def _summarise(rows: Sequence[Dict[str, float | int | bool | str]], label: str) -> Dict[str, float | str]:
    """Aggregate case-level grounded metrics for one configuration."""

    if not rows:
        return {"config": label, "count": 0}

    return {
        "config": label,
        "count": float(len(rows)),
        "top1_accuracy": sum(float(r["top1_correct"]) for r in rows) / len(rows),
        "mrr": sum(float(r["mrr"]) for r in rows) / len(rows),
        "recall_at_k": sum(float(r["recall_at_k"]) for r in rows) / len(rows),
        "false_grounding_rate": sum(float(r["false_grounding"]) for r in rows) / len(rows),
        "safe_null_rate": sum(float(r["safe_null"]) for r in rows) / len(rows),
        "target_rank_mean": sum(float(r["target_rank"]) for r in rows) / len(rows),
        "target_total_score_mean": sum(float(r["target_total_score"]) for r in rows) / len(rows),
        "target_relevance_mean": sum(float(r["target_relevance"]) for r in rows) / len(rows),
        "target_embedding_score_mean": sum(float(r["target_embedding_score"]) for r in rows) / len(rows),
    }


def _build_representative_cases(
    per_config: Dict[str, List[Dict[str, float | int | bool | str]]],
    baseline: str,
    target: str,
    limit: int,
) -> List[Dict[str, float | int | str]]:
    """Return high-signal case-level improvements for poster tables."""

    if baseline not in per_config or target not in per_config:
        return []

    base_map = {str(r["case_id"]): r for r in per_config[baseline]}
    target_map = {str(r["case_id"]): r for r in per_config[target]}

    rows: List[Dict[str, float | int | str]] = []
    for case_id, base_row in base_map.items():
        if case_id not in target_map:
            continue

        target_row = target_map[case_id]
        base_mrr = float(base_row["mrr"])
        target_mrr = float(target_row["mrr"])

        rows.append(
            {
                "case_id": case_id,
                "text": str(base_row["text"]),
                "bucket": str(base_row["bucket"]),
                "expected": int(float(base_row["expected"])),
                "baseline_top1": int(float(base_row["top1_concept_id"])),
                "target_top1": int(float(target_row["top1_concept_id"])),
                "baseline_rank": int(float(base_row["target_rank"])),
                "target_rank": int(float(target_row["target_rank"])),
                "baseline_target_score": float(base_row["target_total_score"]),
                "target_target_score": float(target_row["target_total_score"]),
                "delta_mrr": target_mrr - base_mrr,
            }
        )

    rows.sort(key=lambda r: float(r["delta_mrr"]), reverse=True)
    return rows[:limit]


def run(
    cases_path: Path,
    k: int,
    database_url: Optional[str] = None,
    embedding_backend: Optional[EmbeddingBackendType] = None,
    embedding_storage_base_dir: Optional[str] = None,
    embedding_model: Optional[str] = None,
    embedding_api_base: Optional[str] = None,
    embedding_api_key: Optional[str] = None,
    embedding_metric_type: str = "cosine",
    embedding_index_type: str = "flat",
    domain_filter: Optional[set[str]] = None,
    vocab_filter: Optional[set[str]] = None,
    grounding_parent_ids: Optional[Tuple[int, ...]] = None,
    representative_limit: int = 12,
) -> Dict[str, object]:
    """Run grounded poster benchmark and return report payload."""

    cases = load_cases(cases_path)
    if domain_filter:
        cases = [c for c in cases if c.domain in domain_filter]
    if vocab_filter:
        cases = [c for c in cases if c.vocabulary in vocab_filter]

    if grounding_parent_ids is None and all(c.parent_ids is None for c in cases):
        raise RuntimeError(
            "No grounding parent IDs provided. Set --grounding-parent-id or add parent_ids per case."
        )

    embedding_client = None
    embedding_kg = None
    query_embeddings: Dict[str, np.ndarray] = {}
    if embedding_model is not None and embedding_api_base is not None:
        embedding_client = LLMClient(
            model=embedding_model,
            api_base=embedding_api_base,
            api_key=embedding_api_key or "ollama",
        )
        embedding_kg = build_embedding_knowledge_graph(
            database_url=database_url,
            embedding_backend=embedding_backend,
            embedding_client=embedding_client,
            embedding_storage_base_dir=embedding_storage_base_dir,
        )
        query_embeddings = {
            case.id: embedding_kg.emb.embed_texts(case.text)
            for case in cases
        }

    kg = build_knowledge_graph(database_url)
    configs = _build_configs()

    per_config: Dict[str, List[Dict[str, float | int | bool | str]]] = {}
    summaries: Dict[str, Dict[str, float | str]] = {}
    bucket_summaries: Dict[str, Dict[str, Dict[str, float | str]]] = {}
    errors: Dict[str, str] = {}

    for config in configs:
        try:
            if config.requires_embedding and embedding_kg is None:
                raise MissingExtensionError(
                    "Embedding config requires omop-emb plus embedding model/api settings."
                )

            active_kg = embedding_kg if embedding_kg is not None else kg
            rows: List[Dict[str, float | int | bool | str]] = []

            for case in cases:
                resolver_kwargs: Optional[Dict[str, Any]] = None
                grounding_kwargs: Optional[Dict[str, Any]] = None

                if config.requires_embedding:
                    resolver_kwargs = {
                        "text_embedding": query_embeddings.get(case.id),
                        "text_embedding_model": embedding_model,
                        "metric_type": embedding_metric_type,
                        "index_type": embedding_index_type,
                    }

                if embedding_kg is not None and embedding_model is not None:
                    grounding_kwargs = {
                        "text_embedding": query_embeddings.get(case.id),
                        "text_embedding_model": embedding_model,
                        "embedding_client": embedding_client,
                        "metric_type": embedding_metric_type,
                        "index_type": embedding_index_type,
                    }

                rows.append(
                    _evaluate_case(
                        kg=active_kg,
                        case=case,
                        config=config,
                        k=k,
                        default_parent_ids=grounding_parent_ids,
                        resolver_kwargs=resolver_kwargs,
                        grounding_kwargs=grounding_kwargs,
                    )
                )

        except Exception as exc:
            errors[config.name] = str(exc)
            continue

        per_config[config.name] = rows
        summaries[config.name] = _summarise(rows, config.name)

        buckets: Dict[str, List[Dict[str, float | int | bool | str]]] = {}
        for row in rows:
            bucket = str(row["bucket"])
            buckets.setdefault(bucket, []).append(row)
        bucket_summaries[config.name] = {
            bucket: _summarise(bucket_rows, f"{config.name}:{bucket}")
            for bucket, bucket_rows in buckets.items()
        }

    significance: Dict[str, Dict[str, float]] = {}
    if "basic" in per_config and "extended" in per_config:
        significance["basic_vs_extended"] = mcnemar(
            per_config["basic"],
            per_config["extended"],
        )
    if "extended" in per_config and "full_text" in per_config:
        significance["extended_vs_full_text"] = mcnemar(
            per_config["extended"],
            per_config["full_text"],
        )
    if "full_text" in per_config and "full_text_with_embedding" in per_config:
        significance["full_text_vs_full_text_with_embedding"] = mcnemar(
            per_config["full_text"],
            per_config["full_text_with_embedding"],
        )

    representative_cases = _build_representative_cases(
        per_config=per_config,
        baseline="basic",
        target="full_text_with_embedding",
        limit=representative_limit,
    )

    return {
        "cases_evaluated": len(cases),
        "k": k,
        "summaries": summaries,
        "bucket_summaries": bucket_summaries,
        "significance": significance,
        "representative_cases": representative_cases,
        "errors": errors,
        "database_url": database_url or os.getenv("OMOP_DATABASE_URL"),
        "embedding_model": embedding_model,
        "embedding_backend": embedding_backend,
        "embedding_storage_base_dir": embedding_storage_base_dir,
        "embedding_metric_type": embedding_metric_type,
        "embedding_index_type": embedding_index_type,
        "grounding_parent_ids": grounding_parent_ids,
        "representative_limit": representative_limit,
    }


def main() -> None:
    """CLI entry point for the poster-oriented grounding benchmark."""

    parser = argparse.ArgumentParser(
        description="Grounding benchmark for publication/poster showcases."
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path(__file__).with_name("resolver_cases.json"),
        help="Path to benchmark case JSON file.",
    )
    parser.add_argument(
        "--database-url",
        type=str,
        default=None,
        help="SQLAlchemy database URL for the local OMOP CDM.",
    )
    parser.add_argument(
        "--embedding-backend",
        type=str,
        default=os.getenv("OMOP_EMB_BACKEND"),
        help="Embedding backend. Defaults to OMOP_EMB_BACKEND.",
    )
    parser.add_argument(
        "--embedding-storage-base-dir",
        type=str,
        default=os.getenv("OMOP_EMB_BASE_STORAGE_DIR"),
        help="Optional base directory for file-backed embedding backends.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=os.getenv("OMOP_EMB_MODEL"),
        help="Embedding model name.",
    )
    parser.add_argument(
        "--embedding-api-base",
        type=str,
        default=os.getenv("OMOP_OLLAMA_API_BASE"),
        help="OpenAI-compatible API base for embedding calls.",
    )
    parser.add_argument(
        "--embedding-api-key",
        type=str,
        default=os.getenv("OMOP_OLLAMA_API_KEY"),
        help="Embedding API key.",
    )
    parser.add_argument(
        "--embedding-metric-type",
        type=str,
        default="cosine",
        help="Embedding similarity metric for retrieval/scoring.",
    )
    parser.add_argument(
        "--embedding-index-type",
        type=str,
        default="flat",
        help="Embedding index type for retrieval/scoring.",
    )
    parser.add_argument("--k", type=int, default=5, help="K for Recall@K.")
    parser.add_argument(
        "--grounding-parent-id",
        type=int,
        action="append",
        default=None,
        help="Grounding parent concept ID (repeatable).",
    )
    parser.add_argument(
        "--representative-limit",
        type=int,
        default=12,
        help="Number of representative improved cases to include.",
    )
    parser.add_argument("--domain", action="append", default=None, help="Optional domain filter (repeatable).")
    parser.add_argument("--vocabulary", action="append", default=None, help="Optional vocabulary filter (repeatable).")
    parser.add_argument("--out", type=Path, default=None, help="Optional output JSON path.")
    args = parser.parse_args()

    report = run(
        cases_path=args.cases,
        k=args.k,
        database_url=args.database_url,
        embedding_backend=args.embedding_backend,
        embedding_storage_base_dir=args.embedding_storage_base_dir,
        embedding_model=args.embedding_model,
        embedding_api_base=args.embedding_api_base,
        embedding_api_key=args.embedding_api_key,
        embedding_metric_type=args.embedding_metric_type,
        embedding_index_type=args.embedding_index_type,
        domain_filter=set(args.domain) if args.domain else None,
        vocab_filter=set(args.vocabulary) if args.vocabulary else None,
        grounding_parent_ids=(tuple(args.grounding_parent_id) if args.grounding_parent_id else None),
        representative_limit=args.representative_limit,
    )

    output = json.dumps(report, indent=2)
    print(output)
    if args.out is not None:
        args.out.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
