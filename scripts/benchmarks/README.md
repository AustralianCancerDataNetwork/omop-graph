# Live Resolver Benchmark

This benchmark evaluates resolver configurations against a live OMOP CDM database.

Set `OMOP_DATABASE_URL` or pass `--database-url` to point the benchmark at your local database.

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

- `basic`: exact label + exact synonym resolvers
- `extended`: basic + partial label + partial synonym resolvers
- `full_text`: extended + full-text label + full-text synonym resolvers
- `full_text_with_embedding`: full_text + embedding resolver

If your local OMOP CDM does not have the full-text columns or indexes available, the `full_text` entry will be reported under `errors` in the JSON output.

The embedding ablation requires all of the following to be configured:

- `OMOP_EMB_BACKEND` or `--embedding-backend`
- `OMOP_EMB_MODEL` or `--embedding-model`
- `OMOP_OLLAMA_API_BASE` or `--embedding-api-base`

If any of those are missing, the benchmark will still run the non-embedding configs and report the embedding run under `errors`.

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

Embedding example:

```bash
python scripts/benchmarks/benchmark_resolvers.py \
	--embedding-backend pgvector \
	--embedding-model nomic-embed-text \
	--embedding-api-base http://ollama:11434/v1
```

## Cases

Cases are defined in `scripts/benchmarks/resolver_cases.json` and are grouped by bucket:

- easy
- synonym-heavy
- ambiguous
- noisy
- out-of-scope

Add or edit examples there to expand evaluation coverage.

## Case Format

Each bucket contains a list of case objects. The loader also accepts the older flat list shape, but the bucketed format is easier to read and edit.

- `id`: stable case identifier used in reports
- `text`: the input string given to the resolver pipeline
- `domain` and `vocabulary`: search constraints applied to the resolver pipeline when they are not `NA`
- `expected_concept_id`: the expected best concept for ranking metrics, or `null` when the case should stay ungrounded
- `expected_concept_name`: a human-readable label for the expected concept, or `null` if there is no expected concept

There is no synthetic hit map anymore. The benchmark resolves the input text against your live OMOP CDM and compares the returned concept IDs to `expected_concept_id`.

The difficulty buckets are only for grouping and summary. They do not change the resolver behavior.
