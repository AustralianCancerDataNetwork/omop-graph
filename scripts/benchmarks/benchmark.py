"""Shared helpers for benchmark scripts.

This module centralizes common benchmark plumbing (case loading, DB/KG setup,
constraints, and metric helpers) so multiple benchmark entry points can share
consistent behavior and report semantics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast, Annotated
import typer

import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

import numpy as np
from omop_emb.config import (
    BackendType,
    IndexType,
    MetricType,
    parse_backend_type,
    parse_index_type,
    parse_metric_type,
)
from omop_emb.embeddings import (
    EmbeddingClient, 
    EmbeddingRole
)
from omop_emb.backends.index_config import index_config_from_index_type
from omop_graph.config import OmopGraphConfig
from omop_graph.extensions.emb import get_embedding_writer_interface, MissingExtensionError
from omop_graph.extensions.omop_alchemy import PredicateKind
from omop_graph.graph.constraints import SearchConstraintConcept
from omop_graph.graph.kg import KnowledgeGraph, KnowledgeGraphEmbeddingConfiguration
from omop_graph.graph.scoring import StandardConceptWithScore
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
from omop_graph.db.session import make_engine
app = typer.Typer()


@app.callback()
def _main(
    verbose: Annotated[
        int,
        typer.Option("--verbose", "-v", count=True, help="Increase log verbosity (-v INFO, -vv DEBUG). Must come before the subcommand name."),
    ] = 0,
) -> None:
    OmopGraphConfig.configure_logging(verbosity=verbose)


def _normalize_parent_ids(raw_parent_ids: object) -> Optional[Tuple[int, ...]]:
    """Normalize parent_ids from JSON into an optional tuple of ints."""

    if raw_parent_ids is None:
        return None
    if isinstance(raw_parent_ids, int):
        return (raw_parent_ids,)
    if isinstance(raw_parent_ids, (list, tuple)):
        return tuple(int(parent_id) for parent_id in raw_parent_ids)
    raise TypeError(
        "Invalid parent_ids value. Expected int, list[int], tuple[int, ...], or null."
    )


@dataclass(frozen=True)
class BenchmarkCase:
    """One benchmark example and its expected target concept.

    Parameters
    ----------
    id : str
        Stable case identifier used in outputs.
    text : str
        Query text sent to resolver/grounding pipelines.
    bucket : str
        Difficulty bucket used for grouped reporting.
    domain : str, optional
        OMOP domain constraint value. ``None`` means unconstrained.
    vocabularies : tuple[str, ...], optional
        OMOP vocabulary constraints. ``None`` or empty falls back to defaults.
    expected_concept_id : int, optional
        Gold concept ID. ``None`` indicates out-of-scope/null-grounding case.
    expected_concept_name : str, optional
        Human-readable expected concept label.
    parent_ids : tuple[int, ...], optional
        Optional grounding parent IDs for case-specific hierarchy constraints.
    """

    id: str
    text: str
    bucket: str
    domain: Optional[str] = None
    vocabularies: Optional[Tuple[str, ...]] = None
    expected_concept_id: Optional[int] = None
    expected_concept_name: Optional[str] = None
    parent_ids: Optional[Tuple[int, ...]] = None


@dataclass(frozen=True)
class GroundedBenchmarkConfig:
    """One ablation configuration for grounded benchmark evaluation."""

    name: str
    resolvers: Tuple[CandidateResolver, ...]
    requires_embedding: bool = False


def load_cases(path: Path) -> List[BenchmarkCase]:
    """Load benchmark cases from JSON into typed dataclass instances."""

    payload = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(payload, list):
        cases: List[BenchmarkCase] = []
        for row in payload:
            if "parent_ids" in row:
                row = {**row, "parent_ids": _normalize_parent_ids(row["parent_ids"])}
            if "vocabularies" not in row and "vocabulary" in row:
                legacy_vocab = row.get("vocabulary")
                if legacy_vocab is not None:
                    row = {**row, "vocabularies": (str(legacy_vocab),)}
                row = {k: v for k, v in row.items() if k != "vocabulary"}
            if "vocabularies" in row and row["vocabularies"] is not None:
                raw_vocabs = row["vocabularies"]
                if isinstance(raw_vocabs, str):
                    row = {**row, "vocabularies": (raw_vocabs,)}
                else:
                    row = {**row, "vocabularies": tuple(raw_vocabs)}
            cases.append(BenchmarkCase(**row))
        return cases

    if isinstance(payload, dict):
        cases: List[BenchmarkCase] = []
        for bucket, bucket_cases in payload.items():
            for row in bucket_cases:
                if "parent_ids" in row:
                    row = {**row, "parent_ids": _normalize_parent_ids(row["parent_ids"])}
                if "vocabularies" not in row and "vocabulary" in row:
                    legacy_vocab = row.get("vocabulary")
                    if legacy_vocab is not None:
                        row = {**row, "vocabularies": (str(legacy_vocab),)}
                    row = {k: v for k, v in row.items() if k != "vocabulary"}
                if "vocabularies" in row and row["vocabularies"] is not None:
                    raw_vocabs = row["vocabularies"]
                    if isinstance(raw_vocabs, str):
                        row = {**row, "vocabularies": (raw_vocabs,)}
                    else:
                        row = {**row, "vocabularies": tuple(raw_vocabs)}
                cases.append(BenchmarkCase(bucket=bucket, **row))
        return cases

    raise TypeError(f"Unsupported benchmark case file shape: {type(payload).__name__}")


def build_session_factory() -> sessionmaker:
    """Build a SQLAlchemy session factory via oa-configurator."""
    return sessionmaker(bind=make_engine(), future=True)


def build_engine() -> sa.Engine:
    """Build a SQLAlchemy engine via oa-configurator."""
    return make_engine()


def build_knowledge_graph() -> KnowledgeGraph:
    """Create a KnowledgeGraph backed by the live OMOP CDM database."""
    return KnowledgeGraph(cdm_engine=make_engine())


def build_embedding_knowledge_graph(
    embedding_metric: MetricType,
    embedding_model: Optional[str],
    embedding_backend: Optional[str | BackendType],
    embedding_client: Optional[EmbeddingClient],
) -> KnowledgeGraph:
    """Create a KnowledgeGraph with embedding support configured."""

    cdm_engine = make_engine()
    resolved_embedding_backend = parse_backend_type(embedding_backend) if embedding_backend is not None else None
    resolved_metric_type = parse_metric_type(embedding_metric)

    config = KnowledgeGraphEmbeddingConfiguration(
        metric_type=resolved_metric_type,
        backend_type=resolved_embedding_backend,
        client=embedding_client,
        compute_missing_embeddings=True,
        model_name=embedding_model,
    )
    return KnowledgeGraph(
        cdm_engine=cdm_engine,
        emb_config=config
    )


def case_constraints(case: BenchmarkCase) -> Optional[SearchConstraintConcept]:
    """Translate case metadata into OMOP search constraints when available."""

    domains = (case.domain,) if case.domain else None
    vocabularies = case.vocabularies or None

    return SearchConstraintConcept(
        domains=domains,
        vocabularies=vocabularies,
        require_standard=False,
    )


def percentile(values: List[float], p: int) -> float:
    """Return a simple nearest-rank percentile for numeric values."""

    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int((p / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


def mcnemar(
    a: Sequence[Dict[str, float | int | bool | str]],
    b: Sequence[Dict[str, float | int | bool | str]],
    *,
    field: str = "top1_correct",
) -> Dict[str, float]:
    """Compute continuity-corrected McNemar-style paired comparison."""

    paired = [(float(x[field]), float(y[field])) for x, y in zip(a, b, strict=True)]
    b_only = sum(1 for x, y in paired if x == 0.0 and y == 1.0)
    a_only = sum(1 for x, y in paired if x == 1.0 and y == 0.0)
    denom = b_only + a_only
    chi2 = ((abs(b_only - a_only) - 1.0) ** 2 / denom) if denom > 0 else 0.0
    return {
        "a_only_correct": float(a_only),
        "b_only_correct": float(b_only),
        "mcnemar_chi2_cc": chi2,
    }


def ranking_metrics(
    predictions: Sequence[int],
    expected: Optional[int],
    k: int,
) -> Dict[str, float]:
    """Compute top1/MRR/Recall@K for one ranked prediction list."""

    if expected is None:
        return {"top1_correct": 0.0, "mrr": 0.0, "recall_at_k": 0.0}

    if expected not in predictions:
        return {"top1_correct": 0.0, "mrr": 0.0, "recall_at_k": 0.0}

    rank = predictions.index(expected) + 1
    return {
        "top1_correct": 1.0 if rank == 1 else 0.0,
        "mrr": 1.0 / rank,
        "recall_at_k": 1.0 if rank <= k else 0.0,
    }


def build_grounded_configs() -> Tuple[GroundedBenchmarkConfig, ...]:
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
        GroundedBenchmarkConfig(name="basic", resolvers=basic),
        GroundedBenchmarkConfig(name="extended", resolvers=extended),
        GroundedBenchmarkConfig(name="full_text", resolvers=full_text),
        GroundedBenchmarkConfig(
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


def _summarise_config(rows: Sequence[Dict[str, Any]], label: str) -> Dict[str, Any]:
    """Aggregate case-level ranking metrics into one configuration summary."""

    if not rows:
        return {"config": label, "count": 0}
    n = len(rows)
    return {
        "config": label,
        "count": n,
        "top1_accuracy": sum(float(r["top1_correct"]) for r in rows) / n,
        "mrr": sum(float(r["mrr"]) for r in rows) / n,
        "recall_at_k": sum(float(r["recall_at_k"]) for r in rows) / n,
    }


def _print_summary_report(
    summaries: Dict[str, Dict[str, Any]],
    bucket_summaries: Dict[str, Dict[str, Dict[str, Any]]],
    significance: Dict[str, Dict[str, float]],
    k: int,
) -> None:
    """Print a formatted benchmark summary table to stdout."""

    rk_label = f"R@{k}"
    col_w = (35, 8, 8, 8, 6)
    header = f"{'Config':<{col_w[0]}} {'Top-1':>{col_w[1]}} {'MRR':>{col_w[2]}} {rk_label:>{col_w[3]}} {'N':>{col_w[4]}}"
    sep = "-" * len(header)

    def _row(name: str, s: Dict[str, Any]) -> str:
        return (
            f"{name:<{col_w[0]}} "
            f"{float(s.get('top1_accuracy', 0.0)):>{col_w[1]}.3f} "
            f"{float(s.get('mrr', 0.0)):>{col_w[2]}.3f} "
            f"{float(s.get('recall_at_k', 0.0)):>{col_w[3]}.3f} "
            f"{int(s.get('count', 0)):>{col_w[4]}}"
        )

    print("\n=== Benchmark Summary ===")
    print(header)
    print(sep)
    for config_name, summary in summaries.items():
        if int(summary.get("count", 0)) == 0:
            continue
        print(_row(config_name, summary))

    all_buckets = sorted(
        {b for bs in bucket_summaries.values() for b in bs},
        key=_bucket_sort_key,
    )
    if all_buckets:
        for bucket in all_buckets:
            print(f"\n  -- {bucket.upper()} --")
            print("  " + header)
            print("  " + sep)
            for config_name, by_bucket in bucket_summaries.items():
                if bucket not in by_bucket:
                    continue
                bs = by_bucket[bucket]
                if int(bs.get("count", 0)) == 0:
                    continue
                print("  " + _row(config_name, bs))

    if significance:
        print("\n--- McNemar significance tests ---")
        for pair, result in significance.items():
            print(
                f"  {pair}: χ²={result.get('mcnemar_chi2_cc', 0.0):.3f}"
                f"  (a_only={int(result.get('a_only_correct', 0))},"
                f" b_only={int(result.get('b_only_correct', 0))})"
            )

    print()


def _grounded_element_to_dict(
    concept: StandardConceptWithScore,
) -> Dict[str, object]:
    
    return {
        "concept_id": int(concept.concept_id),
        "concept_name": concept.concept_name,
        "total_score": float(concept.total_score),
        "relevance": float(concept.relevance),
        "embedding_score": float(concept.embedding_score) if concept.embedding_score is not None else 0.0,
        "separation": int(concept.separation),
        "matched_concept_label": concept.matched_concept_label,
        "match_kind": str(concept.match_kind),
        "synonym": concept.synonym,
    }


def _actual_payload(grounded: Sequence[StandardConceptWithScore]) -> Dict[str, object]:
    """Serialize the actual top grounded result for one config."""
    actual_concept = grounded[0] if grounded else None
    #return {
    #    "actual": _grounded_element_to_dict(actual_concept) if actual_concept else None
    #}
    return {
        "actual": {
            "concept_id": int(actual_concept.concept_id) if actual_concept else None,
            "concept_name": actual_concept.concept_name if actual_concept else None,
            "total_score": float(actual_concept.total_score) if actual_concept else 0.0,
        }
    }


def _evaluate_grounded_case(
    kg: KnowledgeGraph,
    case: BenchmarkCase,
    config: GroundedBenchmarkConfig,
    default_parent_ids: Optional[Tuple[int, ...]],
    grounding_kwargs: Optional[Dict[str, object]] = None,
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

    text_embedding = cast(Optional[np.ndarray], grounding_kwargs.get("text_embedding"))

    grounded = ground_term(
        resolver_pipeline=resolver_pipeline,
        kg=kg,
        query=case.text,
        query_embedding=text_embedding,
        constraints=GroundingConstraints(
            parent_ids=parent_ids,
            search_constraint=search_constraint,
            max_depth=6,
            predicate_kinds=frozenset({PredicateKind.IDENTITY}),
        ),
        max_candidates=10,
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
        "grounded": [_grounded_element_to_dict(concept) for concept in grounded],
    }


@app.command()
def run_benchmark(
    cases_file: Annotated[str, typer.Option(
        "--cases-file", "-c", 
        help="Path to the JSON file containing benchmark cases.")
    ],
    embedding_model: Annotated[Optional[str], typer.Option(
        "--embedding-model", "-m",
        help="Name of the embedding model to use (e.g., 'text-embedding-3-small'). Falls back to config.toml")
    ] = None,
    embedding_api_base_url: Annotated[Optional[str], typer.Option(
        "--embedding-api-base-url", "-u",
        help="Base URL for the embedding API (e.g., 'http://localhost:8000'). Falls back to config.toml.")
    ] = None,
    embedding_api_key: Annotated[Optional[str], typer.Option(
        "--embedding-api-key", "-k",
        help="API key for the embedding service, if required. Falls back to config.toml.")
    ] = None,
    embedding_metric_type: Annotated[MetricType, typer.Option(
        "--embedding-metric-type", "-M",
        help="Distance metric type for embedding similarity.")
    ] = MetricType.COSINE,
    embedding_index_type: Annotated[IndexType, typer.Option(
        "--embedding-index-type", "-I",
        help="Index type for embedding retrieval (e.g., 'flat'). Has to match the registered model.")
    ] = IndexType.FLAT,
    out_file: Annotated[Optional[str], typer.Option(
        "--out-file", "-o",
        help="Path to the output JSON file where results will be saved. If not provided, results will be printed to stdout.")
    ] = None,
    k: Annotated[int, typer.Option(
        "--k", "-K",
        help="Number of nearest neighbors to retrieve for each case.")
    ] = 5,
    domains: Annotated[Optional[List[str]], typer.Option(
        "--allowed-domains", "-D",
        help="Used to filter cases within the case file. For multiple domains, repeat the option (e.g., -D Condition -D Procedure).")
    ] = None,
    allowed_vocabularies: Annotated[Optional[List[str]], typer.Option(
        "--allowed-vocabularies", "-V",
        help="Used to filter cases within the case file. For multiple vocabularies, repeat the option (e.g., -V SNOMED -V ICDO3).")
    ] = None,
    parent_ids: Annotated[Optional[List[str]], typer.Option(
        "--grounding-parent-ids", "-G",
        help="Overwrites the parent_ids specified in individual cases. For multiple IDs, repeat the option (e.g., -G 443392 -G 413015).")
    ] = None,
    embedding_backend: Annotated[Optional[str], typer.Option(
        "--embedding-backend", "-e",
        help="Embedding backend to use (e.g., 'sqlite_vec' or 'pgvector'). Defaults to config.toml or OMOP_EMB_BACKEND environment variable.")
    ] = None,
):
    """Generalised benchmark interface."""
    cases = load_cases(Path(cases_file))

    
    if domains:
        domain_filter = set(domains)
        cases = [c for c in cases if c.domain in domain_filter]
    if allowed_vocabularies:
        vocab_filter = set(allowed_vocabularies)
        cases = [
            c
            for c in cases
            if c.vocabularies and any(vocabulary in vocab_filter for vocabulary in c.vocabularies)
        ]
    cases = _order_cases_for_report(cases)
    if parent_ids is not None:
        grounding_parent_ids = tuple(map(int, parent_ids))
    else:
        grounding_parent_ids = None
    if grounding_parent_ids is None and all(c.parent_ids is None for c in cases):
        raise RuntimeError(
            "No grounding parent IDs provided."
        )

    embedding_client = None
    embedding_kg = None
    query_embeddings: Dict[str, np.ndarray] = {}
    resolved_embedding_index_type: Optional[IndexType] = None
    resolved_embedding_metric_type: Optional[MetricType] = None

    if embedding_model is not None and embedding_api_base_url is not None:
        resolved_embedding_index_type = parse_index_type(embedding_index_type)
        resolved_embedding_metric_type = parse_metric_type(embedding_metric_type)

        embedding_client = EmbeddingClient(
            model=embedding_model,
            api_base=embedding_api_base_url,
            api_key=embedding_api_key or "ollama",
        )
        canonical_model = embedding_client.provider.canonical_model_name(embedding_model)

        embedding_kg = build_embedding_knowledge_graph(
            embedding_model=canonical_model,
            embedding_metric=resolved_embedding_metric_type,
            embedding_backend=embedding_backend,
            embedding_client=embedding_client,
        )
        embedding_dim = embedding_client.embedding_dim
        if embedding_dim is None:
            raise RuntimeError("Embedding client did not expose an embedding dimension.")
        
        embedding_writer = get_embedding_writer_interface(embedding_kg)
        assert embedding_writer is not None, "Embedding backend does not support writing embeddings, which is required for this benchmark configuration."

        embedding_writer.register_model(
            index_config=index_config_from_index_type(
                index_type=resolved_embedding_index_type,
            ),
        )

        query_embeddings = {
            case.id: embedding_writer.embed_texts(case.text, embedding_role=EmbeddingRole.QUERY)
            for case in cases
        }

    kg = build_knowledge_graph()
    configs = build_grounded_configs()

    errors: Dict[str, str] = {}
    case_reports: List[Dict[str, Any]] = []
    active_kg = embedding_kg if embedding_kg is not None else kg

    for case in cases:
        config_results: List[Dict[str, Any]] = []
        for config in configs:
            #try:
            if config.requires_embedding and embedding_kg is None:
                raise MissingExtensionError(
                    "Embedding config requires omop-emb plus embedding model/api settings."
                )

            grounding_kwargs: Optional[Dict[str, object]] = None
            if embedding_kg is not None and embedding_model is not None:
                grounding_kwargs = {
                    "text_embedding": query_embeddings.get(case.id),
                    "text_embedding_model": embedding_model,
                    "embedding_client": embedding_client,
                    "metric_type": resolved_embedding_metric_type,
                    "index_type": resolved_embedding_index_type,
                }

            row = _evaluate_grounded_case(
                kg=active_kg,
                case=case,
                config=config,
                default_parent_ids=grounding_parent_ids,
                grounding_kwargs=grounding_kwargs,
            )
            config_results.append(row)

            #except Exception as exc:
            #    errors[f"{case.id}:{config.name}"] = str(exc)
            #    config_results.append(
            #        {
            #            "config": config.name,
            #            "error": str(exc),
            #            "predicted_top": {"concept_id": None, "concept_name": None, "total_score": 0.0, "relevance": 0.0, "embedding_score": 0.0},
            #            "target_total_score": 0.0,
            #        }
            #    )

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

    # Build per-config ranking rows for summary statistics.
    per_config: Dict[str, List[Dict[str, Any]]] = {}
    for case_report in case_reports:
        for cfg_result in case_report["config_results"]:
            config_name = str(cfg_result.get("config", ""))
            if "error" in cfg_result:
                continue
            target_idx = cfg_result.get("target_idx_in_grounded")
            expected_id = case_report["expected_concept_id"]
            if expected_id is None or target_idx is None:
                top1, mrr_val, rak = 0.0, 0.0, 0.0
            else:
                top1 = 1.0 if target_idx == 0 else 0.0
                mrr_val = 1.0 / (int(target_idx) + 1)
                rak = 1.0 if int(target_idx) < k else 0.0
            per_config.setdefault(config_name, []).append({
                "case_id": case_report["case_id"],
                "bucket": case_report["bucket"],
                "top1_correct": top1,
                "mrr": mrr_val,
                "recall_at_k": rak,
            })

    summaries = {name: _summarise_config(rows, name) for name, rows in per_config.items()}

    bucket_summaries: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for config_name, rows in per_config.items():
        by_bucket: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            b = str(row["bucket"])
            by_bucket.setdefault(b, []).append(row)
        bucket_summaries[config_name] = {
            b: _summarise_config(b_rows, f"{config_name}:{b}")
            for b, b_rows in by_bucket.items()
        }

    significance: Dict[str, Dict[str, float]] = {}
    if "basic" in per_config and "extended" in per_config:
        significance["basic_vs_extended"] = mcnemar(per_config["basic"], per_config["extended"])
    if "extended" in per_config and "full_text" in per_config:
        significance["extended_vs_full_text"] = mcnemar(per_config["extended"], per_config["full_text"])
    if "full_text" in per_config and "full_text_with_embedding" in per_config:
        significance["full_text_vs_full_text_with_embedding"] = mcnemar(
            per_config["full_text"], per_config["full_text_with_embedding"]
        )

    report = {
        "cases_evaluated": len(cases),
        "cases": case_reports,
        "summaries": summaries,
        "bucket_summaries": bucket_summaries,
        "significance": significance,
        "errors": errors,
        "embedding_model": embedding_model,
        "embedding_backend": embedding_backend,
        "embedding_metric_type": embedding_metric_type,
        "embedding_index_type": embedding_index_type,
        "grounding_parent_ids": grounding_parent_ids,
        "k": k,
    }

    output = json.dumps(report, indent=2)
    out_str = f"Results for {len(cases)} cases across {len(configs)} configs."
    if out_file is not None:
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(output)
        out_str += f" Results saved to {out_file}"

    _print_summary_report(summaries, bucket_summaries, significance, k)
    print(out_str)


if __name__ == "__main__":
    app()