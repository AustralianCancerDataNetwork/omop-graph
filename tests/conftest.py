import logging

import pytest

pytest_plugins = ("fixtures.mock_cdm",)


class WhitelistFilter(logging.Filter):
    def __init__(self, whitelist):
        self.whitelist = whitelist

    def filter(self, record):
        if record.levelno >= logging.WARNING:
            return True
        return any(record.name.startswith(prefix) for prefix in self.whitelist)


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
