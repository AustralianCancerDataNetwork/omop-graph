"""Grounding-focused benchmark tailored for poster/showcase outputs.

This script evaluates configuration ablations using ``ground_term`` end-to-end
(resolver + hierarchy anchoring + scored ranking) and surfaces representative
case-level improvements for communication in poster figures/tables.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from omop_llm import LLMClient
# Requires the omop-emb package installed!
from omop_emb.config import (
    IndexType,
    BackendType,
    MetricType,
    parse_index_type,
    parse_metric_type,
)
from omop_graph.extensions.emb import MissingExtensionError
from omop_graph.extensions.omop_alchemy import ClassIDEnum
from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.cli import configure_logging_level
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
    build_engine,
    build_knowledge_graph,
    case_constraints,
    load_cases,
)


logger = logging.getLogger(__name__)


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

    if case.parent_ids:
        return case.parent_ids
    return default_parent_ids


def _bucket_sort_key(bucket: str) -> tuple[int, str]:
    """Sort buckets with easy first, then other buckets alphabetically."""

    normalised = bucket.lower()
    if normalised == "easy":
        return (0, normalised)
    return (1, normalised)


def _order_cases_for_report(cases: Sequence[BenchmarkCase]) -> List[BenchmarkCase]:
    """Order cases so the easy bucket is shown first."""

    return sorted(cases, key=lambda case: (_bucket_sort_key(case.bucket), case.id))


def _actual_payload(grounded: Sequence[Any]) -> Dict[str, object]:
    """Serialize the actual top grounded result for one config."""

    actual_concept = grounded[0] if grounded else None

    return {
        "actual": {
            "concept_id": int(actual_concept.concept_id) if actual_concept is not None else None,
            "concept_name": actual_concept.concept_name if actual_concept is not None else None,
            "total_score": float(actual_concept.total_score) if actual_concept is not None else 0.0,
        },
    }


def _evaluate_case(
    kg: KnowledgeGraph,
    case: BenchmarkCase,
    config: PosterConfig,
    default_parent_ids: Optional[Tuple[int, ...]],
    grounding_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, object]:
    """Evaluate one case using ``ground_term`` and return poster-friendly results."""

    search_constraint = case_constraints(case)
    parent_ids = _resolve_parent_ids(case, default_parent_ids)
    if parent_ids is None:
        raise ValueError(
            f"Case '{case.id}' has no parent_ids. Provide case.parent_ids or --grounding-parent-id."
        )

    resolver_pipeline = ResolverPipeline(resolvers=config.resolvers)
    grounding_kwargs = grounding_kwargs or {}

    metric_type = grounding_kwargs.get("metric_type")
    index_type = grounding_kwargs.get("index_type")

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
        max_candidates=10,
        metric_type=metric_type,
        index_type=index_type,
    )

    return {
        "case_id": case.id,
        "text": case.text,
        "bucket": case.bucket,
        "config": config.name,
        "target_concept_id": case.expected_concept_id,
        "target_concept_name": case.expected_concept_name,
        **_actual_payload(grounded),
        "target_idx_in_grounded": next(
            (i for i, concept in enumerate(grounded) if concept.concept_id == case.expected_concept_id),
            None,
        ),
        "grounded": [
            {
                "concept_id": int(concept.concept_id),
                "concept_name": concept.concept_name,
                "total_score": float(concept.total_score),
                "relevance": float(concept.relevance),
                "embedding_score": float(concept.embedding_score) if concept.embedding_score is not None else 0.0,
                "separation": int(concept.separation),
                "matched_label": concept.matched_label,
            }
            for concept in grounded
        ],

    }


def run(
    cases_path: Path,
    k: int,
    database_url: Optional[str] = None,
    embedding_backend: Optional[str | BackendType] = None,
    embedding_storage_base_dir: Optional[str] = None,
    embedding_model: Optional[str] = None,
    embedding_api_base: Optional[str] = None,
    embedding_api_key: Optional[str] = None,
    embedding_metric_type: str = "cosine",
    embedding_index_type: str = "flat",
    domain_filter: Optional[set[str]] = None,
    vocab_filter: Optional[set[str]] = None,
    grounding_parent_ids: Optional[Tuple[int, ...]] = None,
) -> Dict[str, object]:
    """Run grounded poster benchmark and return report payload."""

    logger.info("Starting poster benchmark run.")

    cases = load_cases(cases_path)
    if domain_filter:
        cases = [c for c in cases if c.domain in domain_filter]
    if vocab_filter:
        cases = [
            c
            for c in cases
            if c.vocabularies and any(vocabulary in vocab_filter for vocabulary in c.vocabularies)
        ]
    cases = _order_cases_for_report(cases)
    logger.info("Loaded %d benchmark cases after filters.", len(cases))

    if grounding_parent_ids is None and all(c.parent_ids is None for c in cases):
        raise RuntimeError(
            "No grounding parent IDs provided. Set --grounding-parent-id or add parent_ids per case."
        )

    embedding_client = None
    embedding_kg = None
    query_embeddings: Dict[str, np.ndarray] = {}
    engine = build_engine(database_url)
    resolved_embedding_index_type: Optional[IndexType] = None
    resolved_embedding_metric_type: Optional[MetricType] = None

    if embedding_model is not None and embedding_api_base is not None:
        resolved_embedding_index_type = parse_index_type(embedding_index_type)
        resolved_embedding_metric_type = parse_metric_type(embedding_metric_type)

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
        embedding_dim = embedding_client.embedding_dim
        if embedding_dim is None:
            raise RuntimeError("Embedding client did not expose an embedding dimension.")
        
        embedding_kg.emb.setup_and_register_model(
            engine=engine,
            model_name=embedding_model,
            dimensions=embedding_dim,
            index_type=IndexType(embedding_index_type),
        )

        query_embeddings = {
            case.id: embedding_kg.emb.embed_texts(case.text)
            for case in cases
        }

    kg = build_knowledge_graph(database_url)
    configs = _build_configs()

    per_config: Dict[str, List[Dict[str, float | int | bool | str]]] = {}
    errors: Dict[str, str] = {}

    case_reports: List[Dict[str, object]] = []
    active_kg = embedding_kg if embedding_kg is not None else kg
    for case in cases:
        config_results: List[Dict[str, object]] = []
        for config in configs:
            logger.info("Evaluating case '%s' with config '%s'.", case.id, config.name)
            try:
                if config.requires_embedding and embedding_kg is None:
                    raise MissingExtensionError(
                        "Embedding config requires omop-emb plus embedding model/api settings."
                    )

                grounding_kwargs: Optional[Dict[str, Any]] = None
                if embedding_kg is not None and embedding_model is not None:
                    grounding_kwargs = {
                        "text_embedding": query_embeddings.get(case.id),
                        "text_embedding_model": embedding_model,
                        "embedding_client": embedding_client,
                        "metric_type": resolved_embedding_metric_type,
                        "index_type": resolved_embedding_index_type,
                    }

                row = _evaluate_case(
                    kg=active_kg,
                    case=case,
                    config=config,
                    default_parent_ids=grounding_parent_ids,
                    grounding_kwargs=grounding_kwargs,
                )
                config_results.append(row)

            except Exception as exc:
                errors[f"{case.id}:{config.name}"] = str(exc)
                logger.exception("Case '%s' config '%s' failed: %s", case.id, config.name, exc)
                config_results.append(
                    {
                        "config": config.name,
                        "error": str(exc),
                        "predicted_top": {"concept_id": None, "concept_name": None, "total_score": 0.0, "relevance": 0.0, "embedding_score": 0.0},
                        "target_total_score": 0.0,
                    }
                )

        case_reports.append(
            {
                "case_id": case.id,
                "bucket": case.bucket,
                "text": case.text,
                "expected_concept_id": case.expected_concept_id,
                "expected_concept_name": case.expected_concept_name,
                "config_results": config_results,
            }
        )
    return {
        "cases_evaluated": len(cases),
        "cases": case_reports,
        "errors": errors,
        "database_url": database_url or os.getenv("OMOP_DATABASE_URL"),
        "embedding_model": embedding_model,
        "embedding_backend": embedding_backend,
        "embedding_storage_base_dir": embedding_storage_base_dir,
        "embedding_metric_type": embedding_metric_type,
        "embedding_index_type": embedding_index_type,
        "grounding_parent_ids": grounding_parent_ids,
        "k": k,
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
        "--domain",
        action="append",
        default=None,
        help="Optional domain filter (repeatable).",
    )
    parser.add_argument(
        "--vocabulary",
        action="append",
        default=None,
        help="Optional vocabulary filter (repeatable).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity; use -vv for DEBUG output.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional output JSON path.")
    args = parser.parse_args()

    configure_logging_level(args.verbose)

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
    )

    output = json.dumps(report, indent=2)
    print(output)
    if args.out is not None:
        args.out.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
