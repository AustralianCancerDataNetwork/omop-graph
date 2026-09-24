# Testing

This page summarizes how tests are configured in `omop-graph` and what is currently tested.

## Test Configuration

The test runner configuration lives in `pytest.toml` at the repository root:

- `testpaths = ["tests"]`
- `addopts = ["-rf", "-rx", "--disable-pytest-warnings"]`
- CLI logging enabled at `DEBUG`, filtered to `omop_graph`/`orm_loader`/`omop_spires`/`tests` namespaces (an autouse session fixture in `tests/conftest.py`)

## What Is Currently Tested

The full suite runs against an **in-memory SQLite mock CDM** (`tests/fixtures/mock_cdm.py`'s `mock_cdm_engine` fixture): no real database, no oa-configurator setup, and no environment variables are needed to run it. Current coverage is centered on functional behavior and integration-style graph operations, including:

- Grounding behavior from text inputs to expected OMOP concept IDs
- Rendering behavior for text and Mermaid output under `tests/render`
- Optional full-text behavior guards when sidecar full-text metadata/columns are not present

The grounding test suite is structured with parametrized cases so each clinical term is a separate pytest case for easier isolation and debugging.

### PostgreSQL-only integration suite

A separate suite, tagged with the `db_dialect` marker and excluded from the
default run (`pytest.toml`'s `-m "not db_dialect"`), runs against a real
PostgreSQL database via oa-configurator's test infrastructure. This is where
schema-drift protection and split-connection behavior are covered:
`test_schema_provenance_guard.py`, `test_vocab_split_connection.py`,
`test_oaklib_schema_awareness.py`, `test_fulltext_vocab_schema_postgres.py`,
`test_predicate_flags.py`, `test_relationship_classification.py`.

## Running Tests

Run all tests (SQLite suite only, the default):

```bash
pytest
```

Include the PostgreSQL-only suite:

```bash
pytest -m db_dialect
```

Run one file:

```bash
pytest tests/test_grounding.py
```

Run one parametrized case (example):

```bash
pytest tests/test_grounding.py -k thyroid-cancer
```
