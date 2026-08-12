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
