# Live Resolver Benchmark

This benchmark evaluates resolver configurations against a live OMOP CDM database.

Set `OMOP_CDM_DB_URL` or pass `--database-url` to point the benchmark at your local database.

## What It Measures

- Top-1 accuracy
- Mean Reciprocal Rank (MRR)
- Recall@K
- False grounding rate
- Safe-null rate for out-of-scope cases
- Median and P95 latency
- Candidate pruning ratio (raw hits vs deduplicated predictions)
- McNemar-style paired comparison statistics between ablations

Additional grounded metrics (when `--grounding-parent-id` is provided):

- `ground_top1`, `ground_mrr`, `ground_recall_at_k`
- `ground_false_grounding_rate`, `ground_safe_null_rate`
- `ground_pred_count_mean`
- `ground_target_rank_mean`, `ground_target_total_score_mean`
- `ground_target_relevance_mean`, `ground_target_embedding_score_mean`

### Resolver vs Scoring Metrics

The report now contains two metric families:

- Resolver metrics (`top1_accuracy`, `mrr`, `recall_at_k`, etc.):
	These are based on the resolver pipeline output order (candidate generation +
	deduplication behavior).
- Scoring metrics (`score_*`):
	These are based on the concept scoring stage and are reported in two modes:
	- `score_*_noemb`: scoring without semantic similarity vectors
	- `score_*_emb`: scoring with semantic similarity vectors when available

Use these to separate concerns:

- If resolver metrics improve but scoring metrics do not, retrieval improved but
	score ranking likely did not.
- If `score_*_emb` beats `score_*_noemb`, embedding similarity is adding value
	for ranking quality.
- If both are flat, your current cases may be too easy (or embedding setup may
	not be contributing to disambiguation).

`score_emb_cases` indicates how many cases actually had embedding-based scoring
available in that summary. If this is `0`, `score_*_emb` will be `NA`.

### Practical Interpretation Pattern

For each config (`basic`, `extended`, `full_text`, `full_text_with_embedding`):

1. Check resolver quality first (`top1_accuracy`, `mrr`, `recall_at_k`).
2. Check ranking uplift from semantic similarity:
	 compare `score_mrr_emb` vs `score_mrr_noemb`.
3. Check safety metrics (`false_grounding_rate`, `safe_null_rate`) to ensure
	 gains are not coming from over-grounding noisy/out-of-scope inputs.
4. Use bucket summaries to find where gains happen (ambiguous/noisy typically
	 show the strongest embedding ranking signal).

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

Note: scoring comparison fields (`score_*_emb`) are populated whenever the
benchmark has a working embedding client and model configuration, even for
non-embedding resolver configs. This enables side-by-side scoring comparison in
a single run.

## Run

From repository root:

```bash
python scripts/benchmarks/benchmark_resolvers.py --k 5
```

Run with grounding metrics enabled (repeatable parent IDs):

```bash
python scripts/benchmarks/benchmark_resolvers.py \
	--k 5 \
	--grounding-parent-id 441840
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

Embedding + grounding example:

```bash
python scripts/benchmarks/benchmark_resolvers.py \
	--k 5 \
	--embedding-backend pgvector \
	--embedding-model nomic-embed-text \
	--embedding-api-base http://ollama:11434/v1 \
	--grounding-parent-id 441840
```

## Poster-Oriented Grounded Benchmark

Use `benchmark_poster.py` to run an end-to-end grounded evaluation with
`ground_term` and produce case-level showcase rows suitable for poster tables.

```bash
python scripts/benchmarks/benchmark_poster.py \
	--k 5 \
	--grounding-parent-id 441840 \
	--embedding-backend pgvector \
	--embedding-model nomic-embed-text \
	--embedding-api-base http://ollama:11434/v1 \
	--out scripts/benchmarks/poster_report.json
```

The poster report includes:

- normal summary and bucket summaries per config
- significance comparisons
- `representative_cases`: top case-level improvements from `basic` to
	`full_text_with_embedding` (rank and score deltas)

## Cancer NSW Grounded Benchmark

Use `benchmark_cancer_nsw.py` to run the same grounded benchmark flow against
the dedicated cancer-focused case set.

```bash
python scripts/benchmarks/benchmark_cancer_nsw.py \
	--k 5 \
	--grounding-parent-id 441840 \
	--embedding-backend pgvector \
	--embedding-model nomic-embed-text \
	--embedding-api-base http://ollama:11434/v1
```

This wrapper defaults to:

- `--cases scripts/benchmarks/cancer_nsw_cases.json`
- `--out /home/vscode/benchmark_cancer_nsw.json`

## Cases

Cases are configuration files in `scripts/benchmarks/`:

- `resolver_cases.json` for general poster/resolver evaluation
- `cancer_nsw_cases.json` for cancer-specific benchmarking

Each file is grouped by bucket/category labels.

`resolver_cases.json` currently uses:

- easy
- synonym-heavy
- ambiguous
- noisy
- out-of-scope

`cancer_nsw_cases.json` uses:

- names
- organ

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
