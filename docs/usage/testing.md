# Testing

!!! note
    Test execution is currently local/developer-oriented. CI and deployment-grade test orchestration are planned, but this setup should currently be treated as local-only.

This page summarizes how tests are configured in `omop-graph`, what is currently tested, and how to set up environment variables for local runs.

## Test Configuration

The test runner configuration lives in `pytest.toml` at the repository root.

Current defaults include:

- `testpaths = ["tests"]`
- `addopts = ["-rf", "-rx", "--disable-pytest-warnings"]`
- CLI logging enabled at `INFO`

In addition, `tests/conftest.py` defines session-wide fixtures for:

- Loading environment variables from a dotenv file (default: repo-level `.env`)
- Filtering log output to project-relevant logger namespaces

### Dotenv Resolution For Tests

The autouse session fixture loads environment variables before tests execute:

1. If `OMOP_GRAPH_TEST_ENV_FILE` is set, that file is loaded
2. Otherwise, `./.env` (repo root) is loaded

This means individual test files generally do not need to call `load_dotenv()` directly.

## What Is Currently Tested

Current coverage is centered on functional behavior and integration-style graph operations, including:

- Grounding behavior from text inputs to expected OMOP concept IDs
- Rendering behavior for text and Mermaid output under `tests/render`
- Optional full-text behavior guards when sidecar full-text metadata/columns are not present

The grounding test suite is now structured with parametrized cases so each clinical term is a separate pytest case for easier isolation and debugging.

## Minimal Local .env Example

Create a local `.env` file in the `omop-graph` repo root:

```dotenv
OMOP_CDM_DB_URL=postgresql://omop:omop@db-omop:5432/omop
OMOP_EMB_API_BASE=http://ollama:11434/v1
OMOP_EMB_BACKEND=pgvector
```

Optional override when you want tests to use a different dotenv path:

```bash
export OMOP_GRAPH_TEST_ENV_FILE=/absolute/path/to/custom-test.env
```

## Running Tests

Run all tests:

```bash
pytest
```

Run one file:

```bash
pytest tests/test_grounding.py
```

Run one parametrized case (example):

```bash
pytest tests/test_grounding.py -k thyroid-cancer
```
