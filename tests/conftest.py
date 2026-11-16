import logging

import pytest

pytest_plugins = ("fixtures.mock_cdm",)


def _isolated_cdm_test_database(request, field_name, *, dialect=None):
    from oa_configurator.testing import isolated_test_database

    from omop_graph.config import OmopGraphConfig

    return isolated_test_database(OmopGraphConfig, field_name, dialect=dialect, request=request)


@pytest.fixture
def pg_db(request):
    """Canonical isolated PostgreSQL test database (Phase 0 of the
    schema_translate_map fix).

    Resolves via OA_Configurator resource 'test_cdm_db_pg' in ~/.config/omop/config.toml.
    Run: omop-config configure omop_graph (answer Y when asked to configure test database).

    Everything a test does through ``pg_db.connection``/``pg_db.session``
    happens inside one transaction that's rolled back on exit. Nothing
    here is ever committed to the shared server, so concurrent test runs
    can't collide and no manual cleanup is needed. omop-graph has no
    existing Postgres test fixture; built from scratch here, matching the
    convention already used by OMOP_Alchemy/orm-loader/omop-emb.
    """
    with _isolated_cdm_test_database(request, "test_cdm_db_pg") as db:
        yield db


@pytest.fixture
def duckdb_db():
    """Disposable local DuckDB CDM; no service or config is needed."""
    # The DuckDB integration test creates only minimal test tables, avoiding
    # shared cyclic ORM metadata; don't apply isolated_test_database's broad
    # db_dialect guard, which would exclude it from the default suite.
    with _isolated_cdm_test_database(
        None,
        "test_cdm_db_duckdb",
        dialect="duckdb",
    ) as db:
        yield db


class WhitelistFilter(logging.Filter):
    def __init__(self, whitelist):
        self.whitelist = whitelist

    def filter(self, record):
        if record.levelno >= logging.WARNING:
            return True
        return any(record.name.startswith(prefix) for prefix in self.whitelist)


@pytest.fixture(autouse=True, scope="session")
def configure_logging_whitelist():
    """Attach the filter to the root logger's handlers, not the logger itself,
    so the check runs right before text actually reaches the screen.
    """
    whitelisted_loggers = ["omop_graph", "orm_loader", "omop_spires", "tests"]
    my_filter = WhitelistFilter(whitelisted_loggers)
    for handler in logging.getLogger().handlers:
        handler.addFilter(my_filter)
