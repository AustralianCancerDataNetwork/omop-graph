# Live Resolver Benchmark

This benchmark evaluates resolver configurations against a live OMOP CDM database.

Database connection is resolved automatically from `~/.config/omop/config.toml` via oa-configurator.
No environment variables or `--database-url` flags are needed.

## What It Measures

- Top-1 accuracy
- Mean Reciprocal Rank (MRR)
- Recall@K
- False grounding rate
- Safe-null rate for out-of-scope cases
- Median and P95 latency
- Candidate pruning ratio (raw hits vs deduplicated predictions)
- McNemar-style paired comparison statistics between ablations

Additional grounded metrics (when `--grounding-parent-ids` is provided):

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

The embedding ablation requires the embedding model to be configured in `~/.config/omop/config.toml`
under the `tools.omop_emb` section (fields: `embedding_model`, `ollama_api_base`, `api_key`).

If the embedding configuration is missing, the benchmark will still run the non-embedding configs
and report the embedding run under `errors`.

Note: scoring comparison fields (`score_*_emb`) are populated whenever the
benchmark has a working embedding client and model configuration, even for
non-embedding resolver configs. This enables side-by-side scoring comparison in
a single run.

## Run

From the repository root, using the VS Code launch configs or directly:

```bash
python scripts/benchmarks/benchmark.py -vv run-benchmark \
    --cases-file=scripts/benchmarks/benchmark_cases/resolver_cases.json \
    --k=5
```

With grounding (hierarchy-anchored evaluation):

```bash
python scripts/benchmarks/benchmark.py -vv run-benchmark \
    --cases-file=scripts/benchmarks/benchmark_cases/resolver_cases.json \
    --k=5 \
    --grounding-parent-ids=441840
```

With embedding:

```bash
python scripts/benchmarks/benchmark.py -vv run-benchmark \
    --cases-file=scripts/benchmarks/benchmark_cases/resolver_cases.json \
    --k=5 \
    --grounding-parent-ids=441840 \
    --embedding-model=medembed-small-v0.1-local:f16 \
    --embedding-api-base-url=http://host.docker.internal:11434/v1 \
    --embedding-metric-type=cosine \
    --embedding-index-type=flat \
    --embedding-api-key=ollama
```

Write report to disk:

```bash
python scripts/benchmarks/benchmark.py -vv run-benchmark \
    --cases-file=scripts/benchmarks/benchmark_cases/resolver_cases.json \
    --k=5 \
    --out-file=/home/vscode/benchmark_report.json
```

## Trace Tool

`trace_example.py` runs cases through every resolver stage and produces a detailed JSON
trace and optional SVG flowchart(s) — useful for debugging individual cases and ablation
analysis (per-resolver ranking, target rank per resolver).

```bash
python scripts/benchmarks/trace_example.py -vv trace \
    --cases-file=scripts/benchmarks/benchmark_cases/resolver_cases.json \
    --case-id=easy_hodgkin_lymphoma \
    --parent-ids=443392 \
    --top-n=5 \
    --out-file=results/trace_example.json
```

Generate an SVG flowchart from a trace:

```bash
python scripts/benchmarks/trace_example.py svg \
    --trace-file=results/trace_example.json \
    --out-path=results
```

`--out-file`/`--out-path` is mutually exclusive: `--out-file` writes one combined JSON for
all cases (legacy behavior); `--out-path` writes one file per case, named
`trace_<case_id>.json`, into the given directory — handy when comparing multiple embedding
models, since each model's cases land in their own folder (see below).

`--embedding-model` overrides the model configured in `config.toml` for a single run, useful
for comparing multiple ingested embedding models without editing config between runs:

```bash
python scripts/benchmarks/trace_example.py -vv trace \
    --cases-file=scripts/benchmarks/benchmark_cases/amia_cases.json \
    --top-n=5 --embedding-model=snowflake-arctic-embed2:568m \
    --out-path=results/amia/snowflake-arctic-embed2:568m
```

`svg --trace-file` accepts either a single JSON file or a directory of `trace_*.json` files
(it merges cases from all of them), and `--case-id` defaults to rendering **every** case
found (pass a specific id to render just one). Output defaults to a `plots/` subdirectory
next to `--trace-file`, overridable with `--out-path`:

```bash
python scripts/benchmarks/trace_example.py svg \
    --trace-file=results/amia/snowflake-arctic-embed2:568m
# -> results/amia/snowflake-arctic-embed2:568m/plots/trace_<case_id>.svg, one per case
```

## Cases

Cases are configuration files in `scripts/benchmarks/benchmark_cases/`:

- `resolver_cases.json` for general poster/resolver evaluation
- `cancer_nsw_cases.json` for cancer-specific benchmarking
- `amia_cases.json` for the AMIA reviewer-response MWE (`trace_example.py` worked example):
  cross-domain cases, each with a per-case `parent_ids` anchor, chosen and verified against
  the live CDM so that each bucket cleanly isolates the resolver layer it's meant to
  demonstrate (see `Case Format` below for `parent_ids`)

Each file is grouped by bucket/category labels.

`resolver_cases.json` and `amia_cases.json` use:

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
- `parent_ids`: optional per-case hierarchy anchor(s) for `ground_term`'s mandatory parent constraint. Falls back to `--parent-ids` on the CLI when omitted; needed whenever a cases file mixes domains (e.g. cancer and cardiology cases can't share one global anchor)
- `expected_concept_id`: the expected best concept for ranking metrics, or `null` when the case should stay ungrounded
- `expected_concept_name`: a human-readable label for the expected concept, or `null` if there is no expected concept

There is no synthetic hit map anymore. The benchmark resolves the input text against your live OMOP CDM and compares the returned concept IDs to `expected_concept_id`.

The difficulty buckets are only for grouping and summary. They do not change the resolver behavior.

`trace_example.py`'s output JSON also records the resolved `embedding_model` name (or `null`
if embeddings weren't available) at the top level, alongside `cases`, so a trace file is
self-describing even outside any folder-naming convention.
