import logging

import pytest

pytest_plugins = ("fixtures.mock_cdm",)


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
    from oa_configurator.testing import isolated_test_database
    from omop_graph.config import OmopGraphConfig

    with isolated_test_database(OmopGraphConfig, "test_cdm_db_pg", request=request) as db:
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
