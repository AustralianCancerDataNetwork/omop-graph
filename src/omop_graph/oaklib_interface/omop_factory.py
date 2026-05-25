import os
from typing import Optional, Union

from sqlalchemy.engine import URL

from .omop_resource import OMOPOntologyResource
from omop_graph.config import (
    ENV_OMOP_CDM_DB_URL,
    ENV_OMOP_CDM_DB_HOST,
    ENV_OMOP_CDM_DB_NAME,
    ENV_OMOP_CDM_DB_PASSWORD,
    ENV_OMOP_CDM_DB_PORT,
    ENV_OMOP_CDM_DB_USER,
    ENV_OMOP_CDM_DB_DRIVER,
)


def build_engine_string() -> URL:
    """Compose a SQLAlchemy ``URL`` for the OMOP CDM database from environment variables.

    Returns
    -------
    sqlalchemy.URL

    Notes
    -----
    If ``OMOP_CDM_DB_URL`` is set it is used as-is for any backend, allowing
    callers to supply a fully-qualified connection string without setting the
    individual component variables.

    Raises
    ------
    RuntimeError
        If a required environment variable is missing.
    """
    from sqlalchemy import URL
    from sqlalchemy.engine import make_url

    optional_url = os.getenv(ENV_OMOP_CDM_DB_URL)
    if optional_url:
        return make_url(optional_url)

    # Required variables for composing the URL
    driver = _get_required_env_variable(ENV_OMOP_CDM_DB_DRIVER)
    user = _get_required_env_variable(ENV_OMOP_CDM_DB_USER)
    password = _get_required_env_variable(ENV_OMOP_CDM_DB_PASSWORD)
    host = _get_required_env_variable(ENV_OMOP_CDM_DB_HOST)
    database = _get_required_env_variable(ENV_OMOP_CDM_DB_NAME)
    port = int(_get_required_env_variable(ENV_OMOP_CDM_DB_PORT))
    return URL.create(
        drivername=driver,
        username=user,
        password=password,
        host=host,
        port=port,
        database=database,
    )

def _get_required_env_variable(name: str) -> str:
    """Get the value of a required environment variable.

    Parameters
    ----------
    name : str
        Environment variable name.

    Returns
    -------
    str
        Environment variable value.

    Raises
    ------
    RuntimeError
        If the environment variable is not set.
    """
    value = os.getenv(name)
    if value is None:
        raise RuntimeError(f"Required environment variable {name!r} is not set.")
    return value


def omop_resource(
    *,
    url: Optional[Union[str, URL]] = None,
    slug: Optional[str] = "omop",
) -> OMOPOntologyResource:
    """
    Create an OMOP DatabaseOntologyResource.

    This factory function resolves the database connection string by prioritizing
    an explicit URL argument. If no URL is provided, it attempts to read from
    the specified environment variable.

    Parameters
    ----------
    url : str | URL, optional
        The explicit database connection URL (highest priority).
    env_var : str, optional
        The name of the environment variable to check if `url` is None.
        Defaults to 'OMOP_CDM_DB_URL'.
    slug : str, optional
        A slug identifier for the resource. Defaults to 'omop'.

    Returns
    -------
    OMOPOntologyResource
        The configured resource object.

    Raises
    ------
    ValueError
        If neither `url` is provided nor the `env_var` is set.
    """
    resolved = url or build_engine_string()

    if not resolved:
        raise ValueError(
            f"No database URL provided and required environment variables not set"
        )

    return OMOPOntologyResource(
        slug=slug,
        url=resolved,
    )