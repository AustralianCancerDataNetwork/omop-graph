from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, cast

from omop_graph.graph.nodes import LabelMatch, LabelMatchGroupView, LabelMatchKind
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


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    text: str
    bucket: str
    domain: str
    vocabulary: str
    expected_concept_id: Optional[int]
    hits: Dict[str, List[int]]


class BenchmarkKnowledgeGraph:
    """A lightweight stand-in for KnowledgeGraph that serves deterministic resolver hits."""

    def __init__(self) -> None:
        self.current_case: Optional[BenchmarkCase] = None

    def set_case(self, case: BenchmarkCase) -> None:
        self.current_case = case

    def concept_lookup(
        self,
        label: str,
        match_kind: LabelMatchKind,
        synonym: bool = False,
        search_constraint=None,
        sort: bool = False,
    ) -> LabelMatchGroupView:
        assert self.current_case is not None, "Case must be set before lookups"
        key = _resolver_key(match_kind=match_kind, synonym=synonym)
        concept_ids = tuple(self.current_case.hits.get(key, []))
        matches = tuple(
            LabelMatch(
                input_label=label,
                matched_label=f"concept-{cid}",
                concept_id=cid,
                match_kind=match_kind,
                is_standard=True,
                is_active=True,
            )
            for cid in concept_ids
        )
        if sort:
            matches = tuple(sorted(matches))
        return LabelMatchGroupView.from_matches(matches)


class MockEmbeddingResolver(CandidateResolver):
    """Embedding resolver for synthetic benchmarking without omop-emb backend dependencies."""

    def __init__(self) -> None:
        super().__init__(match_kind=LabelMatchKind.EMBEDDING, synonym=False)

    def get_matches(self, kg: BenchmarkKnowledgeGraph, text: str, constraints=None, sort: bool = False, **kwargs) -> Tuple[LabelMatch, ...]:
        assert kg.current_case is not None, "Case must be set before lookups"
        concept_ids = tuple(kg.current_case.hits.get("embedding", []))
        matches = tuple(
            LabelMatch(
                input_label=text,
                matched_label=f"concept-{cid}",
                concept_id=cid,
                match_kind=LabelMatchKind.EMBEDDING,
                is_standard=True,
                is_active=True,
            )
            for cid in concept_ids
        )
        if sort:
            matches = tuple(sorted(matches))
        return matches


def _resolver_key(match_kind: LabelMatchKind, synonym: bool) -> str:
    if match_kind == LabelMatchKind.EXACT:
        return "exact_synonym" if synonym else "exact"
    if match_kind == LabelMatchKind.PARTIAL:
        return "partial_synonym" if synonym else "partial"
    if match_kind == LabelMatchKind.FTS:
        return "fts_synonym" if synonym else "fts"
    if match_kind == LabelMatchKind.EMBEDDING:
        return "embedding"
    raise ValueError(f"Unsupported match kind: {match_kind}")


def _load_cases(path: Path) -> List[BenchmarkCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [BenchmarkCase(**row) for row in payload]


def _build_configs() -> Dict[str, Tuple[CandidateResolver, ...]]:
    basic = (
        ExactLabelResolver(),
        ExactSynonymResolver(),
    )
    extended = (
        *basic,
        PartialLabelResolver(),
        PartialSynonymResolver(),
        FullTextResolver(),
        FullTextSynonymResolver(),
    )
    full = (
        *extended,
        MockEmbeddingResolver(),
    )

    return {
        "basic": basic,
        "extended": extended,
        "full_with_embeddings": full,
    }


def _evaluate_case(
    kg: BenchmarkKnowledgeGraph,
    case: BenchmarkCase,
    resolvers: Tuple[CandidateResolver, ...],
    k: int,
) -> Dict[str, float | int | bool | str]:
    kg.set_case(case)
    pipeline = ResolverPipeline(resolvers=resolvers)

    typed_kg = cast(KnowledgeGraph, kg)

    # Candidate pruning estimate: pre-dedup hits vs deduped pipeline output.
    raw_hits = 0
    for resolver in resolvers:
        raw_hits += len(tuple(resolver.resolve(typed_kg, case.text)))

    t0 = time.perf_counter()
    predictions = [hit.concept_id for hit in pipeline.resolve(typed_kg, case.text)]
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
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int((p / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


def _mcnemar(a: Sequence[Dict[str, float | int | bool | str]], b: Sequence[Dict[str, float | int | bool | str]]) -> Dict[str, float]:
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


def run(cases_path: Path, k: int, domain_filter: Optional[set[str]] = None, vocab_filter: Optional[set[str]] = None) -> Dict[str, object]:
    cases = _load_cases(cases_path)
    if domain_filter:
        cases = [c for c in cases if c.domain in domain_filter]
    if vocab_filter:
        cases = [c for c in cases if c.vocabulary in vocab_filter]

    kg = BenchmarkKnowledgeGraph()
    configs = _build_configs()

    per_config: Dict[str, List[Dict[str, float | int | bool | str]]] = {}
    summaries: Dict[str, Dict[str, float | str]] = {}
    bucket_summaries: Dict[str, Dict[str, Dict[str, float | str]]] = {}

    for name, resolvers in configs.items():
        rows = [_evaluate_case(kg=kg, case=case, resolvers=resolvers, k=k) for case in cases]
        per_config[name] = rows
        summaries[name] = _summarise(rows, name)

        buckets: Dict[str, List[Dict[str, float | int | bool | str]]] = {}
        for row in rows:
            bucket = str(row["bucket"])
            buckets.setdefault(bucket, []).append(row)
        bucket_summaries[name] = {bucket: _summarise(bucket_rows, f"{name}:{bucket}") for bucket, bucket_rows in buckets.items()}

    significance = {
        "basic_vs_extended": _mcnemar(per_config["basic"], per_config["extended"]),
        "extended_vs_full_with_embeddings": _mcnemar(per_config["extended"], per_config["full_with_embeddings"]),
    }

    return {
        "cases_evaluated": len(cases),
        "k": k,
        "summaries": summaries,
        "bucket_summaries": bucket_summaries,
        "significance": significance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic resolver benchmark without OMOP CDM dependency.")
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path(__file__).with_name("resolver_cases.json"),
        help="Path to benchmark case JSON file.",
    )
    parser.add_argument("--k", type=int, default=5, help="K for Recall@K.")
    parser.add_argument("--domain", action="append", default=None, help="Optional domain filter (repeatable).")
    parser.add_argument("--vocabulary", action="append", default=None, help="Optional vocabulary filter (repeatable).")
    parser.add_argument("--out", type=Path, default=None, help="Optional output JSON path.")
    args = parser.parse_args()

    report = run(
        cases_path=args.cases,
        k=args.k,
        domain_filter=set(args.domain) if args.domain else None,
        vocab_filter=set(args.vocabulary) if args.vocabulary else None,
    )

    output = json.dumps(report, indent=2)
    print(output)
    if args.out is not None:
        args.out.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
