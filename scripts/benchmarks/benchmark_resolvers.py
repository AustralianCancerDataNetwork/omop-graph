"""Resolver benchmark against a live OMOP CDM database.

This benchmark measures how the resolver pipeline performs on real OMOP data.
Each case supplies an input phrase, a difficulty bucket, optional domain and
vocabulary constraints, and the expected OMOP concept ID. The benchmark does
not simulate resolver output; it runs the actual resolvers against the local
database configured by ``OMOP_DATABASE_URL`` or ``--database-url``.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import sqlalchemy as sa
from dotenv import load_dotenv
from sqlalchemy.orm import sessionmaker

from omop_graph.graph.constraints import SearchConstraintConcept
from omop_graph.extensions.emb import EmbeddingBackendType, MissingExtensionError
from omop_graph.graph.kg import KnowledgeGraph
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
from omop_llm import LLMClient


@dataclass(frozen=True)
class BenchmarkCase:
    """One real benchmark example and its expected ground-truth concept ID."""

    id: str
    text: str
    bucket: str
    domain: str
    vocabulary: str
    expected_concept_id: Optional[int]
    expected_concept_name: Optional[str] = None


@dataclass(frozen=True)
class BenchmarkConfig:
    """One resolver ablation evaluated by the benchmark."""

    name: str
    resolvers: Tuple[CandidateResolver, ...]
    requires_embedding: bool = False


def _load_cases(path: Path) -> List[BenchmarkCase]:
    """Load benchmark cases from JSON into typed dataclass instances.

    The loader accepts either a legacy flat list of cases or a bucketed mapping
    of bucket name to list of cases. Bucket names are attached to each case at
    load time so the evaluation/reporting code can stay unchanged.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(payload, list):
        return [BenchmarkCase(**row) for row in payload]

    if isinstance(payload, dict):
        cases: List[BenchmarkCase] = []
        for bucket, bucket_cases in payload.items():
            for row in bucket_cases:
                cases.append(BenchmarkCase(bucket=bucket, **row))
        return cases

    raise TypeError(f"Unsupported benchmark case file shape: {type(payload).__name__}")


def _build_session_factory(database_url: Optional[str]) -> sessionmaker:
    """Build a SQLAlchemy session factory for the configured OMOP database."""

    load_dotenv()
    resolved_url = database_url or os.getenv("OMOP_DATABASE_URL")
    if not resolved_url:
        raise RuntimeError(
            "No database URL provided. Pass --database-url or set OMOP_DATABASE_URL."
        )

    engine = sa.create_engine(resolved_url, future=True, echo=False)
    return sessionmaker(bind=engine, future=True)


def _build_knowledge_graph(database_url: Optional[str]) -> KnowledgeGraph:
    """Create a KnowledgeGraph backed by the live OMOP CDM database."""

    return KnowledgeGraph(session_factory=_build_session_factory(database_url))


def _build_embedding_knowledge_graph(
    database_url: Optional[str],
    embedding_backend: Optional[EmbeddingBackendType],
    embedding_client: Optional[LLMClient],
    embedding_storage_base_dir: Optional[str],
) -> KnowledgeGraph:
    """Create a KnowledgeGraph with embedding support configured when requested."""

    session_factory = _build_session_factory(database_url)
    return KnowledgeGraph(
        session_factory=session_factory,
        emb_backend=embedding_backend,
        emb_base_storage_dir=embedding_storage_base_dir,
        emb_client=embedding_client,
    )


def _case_constraints(case: BenchmarkCase) -> Optional[SearchConstraintConcept]:
    """Translate case metadata into OMOP search constraints when available."""

    if case.domain == "NA" and case.vocabulary == "NA":
        return None

    domains = (case.domain,) if case.domain != "NA" else None
    vocabularies = (case.vocabulary,) if case.vocabulary != "NA" else None

    return SearchConstraintConcept(
        domains=domains,
        vocabs=vocabularies,
        require_standard=False,
    )


def _build_configs() -> Tuple[BenchmarkConfig, ...]:
    """Build the real resolver ablations compared by the benchmark report."""

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
        BenchmarkConfig(name="basic", resolvers=basic),
        BenchmarkConfig(name="extended", resolvers=extended),
        BenchmarkConfig(name="full_text", resolvers=full_text),
        BenchmarkConfig(name="full_text_with_embedding", resolvers=full_text_with_embedding, requires_embedding=True),
    )


def _evaluate_case(
    kg: KnowledgeGraph,
    case: BenchmarkCase,
    resolvers: Tuple[CandidateResolver, ...],
    k: int,
    resolver_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, float | int | bool | str]:
    """Run one real benchmark case and derive ranking, safety, and pruning metrics."""

    constraints = _case_constraints(case)
    pipeline = ResolverPipeline(resolvers=resolvers)
    resolver_kwargs = resolver_kwargs or {}

    # Candidate pruning estimate: pre-dedup hits vs deduped pipeline output.
    raw_hits = 0
    for resolver in resolvers:
        raw_hits += len(tuple(resolver.resolve(kg, case.text, constraints=constraints, **resolver_kwargs)))

    t0 = time.perf_counter()
    predictions = [hit.concept_id for hit in pipeline.resolve(kg, case.text, constraints=constraints, **resolver_kwargs)]
    latency_ms = (time.perf_counter() - t0) * 1000.0

    expected = case.expected_concept_id
    top1_correct = False
    mrr = 0.0
    recall_at_k = 0.0
    false_grounding = False
    safe_null = False

    if expected is None:
        safe_null = len(predictions) == 0
        false_grounding = len(predictions) > 0
    else:
        if expected in predictions:
            rank = predictions.index(expected) + 1
            top1_correct = rank == 1
            mrr = 1.0 / rank
            recall_at_k = 1.0 if rank <= k else 0.0
        else:
            false_grounding = len(predictions) > 0

    unique_hits = len(predictions)
    pruning_ratio = (1.0 - (unique_hits / raw_hits)) if raw_hits > 0 else 0.0

    return {
        "case_id": case.id,
        "bucket": case.bucket,
        "expected": -1 if expected is None else expected,
        "expected_concept_name": case.expected_concept_name or "",
        "pred_count": len(predictions),
        "top1_correct": float(top1_correct),
        "mrr": mrr,
        "recall_at_k": recall_at_k,
        "false_grounding": float(false_grounding),
        "safe_null": float(safe_null),
        "latency_ms": latency_ms,
        "raw_hits": raw_hits,
        "unique_hits": unique_hits,
        "pruning_ratio": pruning_ratio,
    }


def _summarise(results: Sequence[Dict[str, float | int | bool | str]], label: str) -> Dict[str, float | str]:
    """Aggregate case-level measurements into one configuration summary."""

    if not results:
        return {"config": label, "count": 0}

    latencies = [float(r["latency_ms"]) for r in results]

    return {
        "config": label,
        "count": float(len(results)),
        "top1_accuracy": sum(float(r["top1_correct"]) for r in results) / len(results),
        "mrr": sum(float(r["mrr"]) for r in results) / len(results),
        "recall_at_k": sum(float(r["recall_at_k"]) for r in results) / len(results),
        "false_grounding_rate": sum(float(r["false_grounding"]) for r in results) / len(results),
        "safe_null_rate": sum(float(r["safe_null"]) for r in results) / len(results),
        "latency_median_ms": statistics.median(latencies),
        "latency_p95_ms": _percentile(latencies, 95),
        "pruning_ratio_mean": sum(float(r["pruning_ratio"]) for r in results) / len(results),
    }


def _percentile(values: List[float], p: int) -> float:
    """Return a simple nearest-rank percentile for a list of latencies."""

    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int((p / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


def _mcnemar(a: Sequence[Dict[str, float | int | bool | str]], b: Sequence[Dict[str, float | int | bool | str]]) -> Dict[str, float]:
    """Compute a lightweight paired comparison on top-1 correctness."""

    paired = [(float(x["top1_correct"]), float(y["top1_correct"])) for x, y in zip(a, b, strict=True)]
    b_only = sum(1 for x, y in paired if x == 0.0 and y == 1.0)
    a_only = sum(1 for x, y in paired if x == 1.0 and y == 0.0)
    denom = b_only + a_only
    chi2 = ((abs(b_only - a_only) - 1.0) ** 2 / denom) if denom > 0 else 0.0
    return {
        "a_only_correct": float(a_only),
        "b_only_correct": float(b_only),
        "mcnemar_chi2_cc": chi2,
    }


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
) -> Dict[str, object]:
    """Execute the benchmark and return a JSON-serialisable report object."""

    cases = _load_cases(cases_path)
    if domain_filter:
        cases = [c for c in cases if c.domain in domain_filter]
    if vocab_filter:
        cases = [c for c in cases if c.vocabulary in vocab_filter]

    embedding_client = None
    embedding_kg = None
    if embedding_model is not None and embedding_api_base is not None:
        embedding_client = LLMClient(
            model=embedding_model,
            api_base=embedding_api_base,
            api_key=embedding_api_key or "ollama",
        )
        embedding_kg = _build_embedding_knowledge_graph(
            database_url=database_url,
            embedding_backend=embedding_backend,
            embedding_client=embedding_client,
            embedding_storage_base_dir=embedding_storage_base_dir,
        )

    kg = _build_knowledge_graph(database_url)
    configs = _build_configs()

    per_config: Dict[str, List[Dict[str, float | int | bool | str]]] = {}
    summaries: Dict[str, Dict[str, float | str]] = {}
    bucket_summaries: Dict[str, Dict[str, Dict[str, float | str]]] = {}
    errors: Dict[str, str] = {}

    for config in configs:
        try:
            if config.requires_embedding and embedding_kg is None:
                raise MissingExtensionError(
                    "Embedding benchmark requires `omop-emb` plus `OMOP_EMB_BACKEND`, `--embedding-model`, and `--embedding-api-base`.")

            if config.requires_embedding:
                active_kg = embedding_kg
                assert active_kg is not None
                query_embeddings = {
                    case.id: active_kg.emb.embed_texts(case.text)
                    for case in cases
                }
                resolver_kwargs: Dict[str, Any] = {
                    "text_embedding": None,
                    "text_embedding_model": embedding_model,
                    "metric_type": embedding_metric_type,
                    "index_type": embedding_index_type,
                }

                rows = []
                for case in cases:
                    resolver_kwargs["text_embedding"] = query_embeddings[case.id]
                    rows.append(
                        _evaluate_case(
                            kg=active_kg,
                            case=case,
                            resolvers=config.resolvers,
                            k=k,
                            resolver_kwargs=resolver_kwargs,
                        )
                    )
            else:
                active_kg = kg
                rows = [
                    _evaluate_case(
                        kg=active_kg,
                        case=case,
                        resolvers=config.resolvers,
                        k=k,
                    )
                    for case in cases
                ]
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
        significance["basic_vs_extended"] = _mcnemar(per_config["basic"], per_config["extended"])
    if "extended" in per_config and "full_text" in per_config:
        significance["extended_vs_full_text"] = _mcnemar(per_config["extended"], per_config["full_text"])
    if "full_text" in per_config and "full_text_with_embedding" in per_config:
        significance["full_text_vs_full_text_with_embedding"] = _mcnemar(
            per_config["full_text"],
            per_config["full_text_with_embedding"],
        )

    return {
        "cases_evaluated": len(cases),
        "k": k,
        "summaries": summaries,
        "bucket_summaries": bucket_summaries,
        "significance": significance,
        "errors": errors,
        "database_url": database_url or os.getenv("OMOP_DATABASE_URL"),
        "embedding_model": embedding_model,
        "embedding_backend": embedding_backend,
        "embedding_storage_base_dir": embedding_storage_base_dir,
        "embedding_metric_type": embedding_metric_type,
        "embedding_index_type": embedding_index_type,
    }


def main() -> None:
    """CLI entry point for running the real OMOP benchmark from the shell."""

    parser = argparse.ArgumentParser(description="Resolver benchmark against a live OMOP CDM database.")
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
        help="SQLAlchemy database URL for the local OMOP CDM. Defaults to OMOP_DATABASE_URL.",
    )
    parser.add_argument(
        "--embedding-backend",
        type=str,
        default=os.getenv("OMOP_EMB_BACKEND"),
        help="Embedding backend to use for the benchmark. Defaults to OMOP_EMB_BACKEND.",
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
        help="Embedding model name for query embeddings and model lookup.",
    )
    parser.add_argument(
        "--embedding-api-base",
        type=str,
        default=os.getenv("OMOP_OLLAMA_API_BASE"),
        help="OpenAI-compatible API base used by the embedding client.",
    )
    parser.add_argument(
        "--embedding-api-key",
        type=str,
        default=os.getenv("OMOP_OLLAMA_API_KEY"),
        help="API key used by the embedding client. Defaults to the Ollama compatibility value.",
    )
    parser.add_argument(
        "--embedding-metric-type",
        type=str,
        default="cosine",
        help="Similarity metric used by embedding retrieval.",
    )
    parser.add_argument(
        "--embedding-index-type",
        type=str,
        default="flat",
        help="Embedding index type used by embedding retrieval.",
    )
    parser.add_argument("--k", type=int, default=5, help="K for Recall@K.")
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
    )

    output = json.dumps(report, indent=2)
    print(output)
    if args.out is not None:
        args.out.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
