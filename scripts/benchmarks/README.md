# Synthetic Resolver Benchmark

This benchmark evaluates resolver configurations without requiring OMOP CDM access.

## What It Measures

- Top-1 accuracy
- Mean Reciprocal Rank (MRR)
- Recall@K
- False grounding rate
- Safe-null rate for out-of-scope cases
- Median and P95 latency
- Candidate pruning ratio (raw hits vs deduplicated predictions)
- McNemar-style paired comparison statistics between ablations

## Ablation Configurations

- `basic`: exact + exact synonym
- `extended`: basic + partial + full-text resolvers
- `full_with_embeddings`: extended + synthetic embedding resolver

## Run

From repository root:

```bash
python scripts/benchmarks/benchmark_resolvers.py --k 5
```

Optional filters:

```bash
python scripts/benchmarks/benchmark_resolvers.py --domain Condition --vocabulary SNOMED
```

Write report to disk:

```bash
python scripts/benchmarks/benchmark_resolvers.py --out scripts/benchmarks/report.json
```

## Cases

Cases are defined in `scripts/benchmarks/resolver_cases.json` and include buckets:

- easy
- synonym-heavy
- ambiguous
- noisy
- out-of-scope

Add or edit examples there to expand evaluation coverage.
