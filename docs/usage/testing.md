# Testing

This page summarizes how tests are configured in `omop-graph` and what is currently tested.

## Test Configuration

The test runner configuration lives in `pytest.toml` at the repository root:

- `testpaths = ["tests"]`
- `addopts = ["-rf", "-rx", "--disable-pytest-warnings"]`
- CLI logging enabled at `DEBUG`, filtered to `omop_graph`/`orm_loader`/`omop_spires`/`tests` namespaces (an autouse session fixture in `tests/conftest.py`)

## Layout

The tests are grouped by what they cover, in folders under `tests`:

- `oaklib` for the oaklib adapter, one file per interface
- `graph` for the knowledge graph: edges, queries, traversal and enumeration
- `classification` for the relationship classification data and loader
- `reasoning` for grounding and the resolvers
- `db` for connections, schemas and the Postgres fixture
- `extensions` for the optional embedding and full-text support
- `cli` for the command line
- `fixtures` for the shared mock CDM

## What Is Currently Tested

The default suite runs graph queries against an **in-memory SQLite mock CDM** (`tests/fixtures/mock_cdm.py`'s `mock_cdm_engine_sqlite` fixture, used by `mock_cdm_kg`). PostgreSQL-only integration tests use the separate `mock_cdm_engine` fixture and oa-configurator's isolated test database fixture. Current SQLite coverage includes:

- Grounding behavior from text inputs to expected OMOP concept IDs
- Optional full-text behavior guards when sidecar full-text metadata/columns are not present
- KnowledgeGraph full-text lookup returning no matches when SQLite does not support full-text search

The grounding test suite is structured with parametrized cases so each clinical term is a separate pytest case for easier isolation and debugging.

### PostgreSQL-only integration suite

A separate suite, tagged with the `db_dialect` marker and excluded from the
default run (`pytest.toml`'s `-m "not db_dialect"`), runs against a real
PostgreSQL database via oa-configurator's test infrastructure. This is where
schema-drift protection and split-connection behavior are covered, mostly in
`tests/db`: `test_schema_provenance_guard.py`, `test_vocab_split_connection.py` and
`test_fulltext_vocab_schema_postgres.py`.

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
pytest tests/reasoning/test_grounding.py
```

Run one parametrized case (example):

```bash
pytest tests/reasoning/test_grounding.py -k thyroid-cancer
```
