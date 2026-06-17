import logging
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

pytest_plugins = ("fixtures.mock_cdm",)


TEST_ENV_FILE_ENV_VAR = "OMOP_GRAPH_TEST_ENV_FILE"
DEFAULT_TEST_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


class WhitelistFilter(logging.Filter):
    def __init__(self, whitelist):
        self.whitelist = whitelist

    def filter(self, record):
        # 1. Allow high severity logs
        if record.levelno >= logging.WARNING:
            return True
        # 2. Allow logs from your specific modules
        return any(record.name.startswith(prefix) for prefix in self.whitelist)


@pytest.fixture(autouse=True, scope="session")
def load_test_environment():
    """Load environment variables from the repo .env file before tests run."""
    env_file = Path(os.getenv(TEST_ENV_FILE_ENV_VAR, DEFAULT_TEST_ENV_FILE))
    load_dotenv(dotenv_path=env_file, override=False)


@pytest.fixture(autouse=True, scope="session")
def configure_logging_whitelist():
    """
    Attaches the filter to the HANDLERS, not the logger.
    """
    # Your allowed list
    my_whitelisted_loggers = ["omop_graph", "orm_loader", "omop_spires", "tests"]

    # Instantiate the filter
    my_filter = WhitelistFilter(my_whitelisted_loggers)

    # Get the root logger
    root_logger = logging.getLogger()

    # --- THE FIX ---
    # We iterate over the handlers (Console, File, etc.) and attach the filter there.
    # This forces the check to happen right before the text hits the screen.
    for handler in root_logger.handlers:
        handler.addFilter(my_filter)
