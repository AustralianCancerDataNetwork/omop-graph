# Testing

This page summarizes how tests are configured in `omop-graph` and what is currently tested.

## Test Configuration

The test runner configuration lives in `pytest.toml` at the repository root:

- `testpaths = ["tests"]`
- `addopts = ["-rf", "-rx", "--strict-markers"]`
- CLI logging enabled at `DEBUG`, filtered to `omop_graph`/`orm_loader`/`omop_spires`/`tests` namespaces (an autouse session fixture in `tests/conftest.py`)

## What Is Currently Tested

Graph queries run against a mock CDM on every dialect: `tests/fixtures/mock_cdm.py`'s `mock_cdm_kg` is parametrized with oa-configurator's `DIALECT_PARAMS` over the `mock_cdm_engine_sqlite`, `mock_cdm_engine` (PostgreSQL) and `mock_cdm_engine_duckdb` fixtures. Each param carries its dialect's routing mark (`sqlite`, `postgresql`, `duckdb`), which selects the CI job that runs it. A test without a routing mark needs no database. Current coverage includes:

- Grounding behavior from text inputs to expected OMOP concept IDs
- Rendering behavior for text and Mermaid output under `tests/render`
- Optional full-text behavior guards when sidecar full-text metadata/columns are not present
- KnowledgeGraph full-text lookup returning no matches when SQLite does not support full-text search

The grounding test suite is structured with parametrized cases so each clinical term is a separate pytest case for easier isolation and debugging.

### Dialect-specific suites

Tests tagged `postgresql` or `duckdb` run against a real database via oa-configurator's test infrastructure. This is where schema-drift protection and split-connection behavior are covered: `test_schema_provenance_guard.py`, `test_vocab_split_connection.py`, `test_oaklib_schema_awareness.py`, `test_fulltext_vocab_schema_postgres.py`, `test_predicate_flags.py`, `test_relationship_classification.py` and `test_relationship_classification_duckdb.py`.

## Running Tests

Run all tests. This needs the test databases configured. A test whose database is not configured fails, and the message names it:

```bash
pytest
```

Run the tests of one routing mark, or the tests that need no database:

```bash
pytest -m postgresql
pytest -m "not (sqlite or duckdb or postgresql)"
```

Run one file:

```bash
pytest tests/test_grounding.py
```

Run one parametrized case (example):

```bash
pytest tests/test_grounding.py -k thyroid-cancer
```
