from __future__ import annotations
import os
from typing import Optional, Union
from sqlalchemy import create_engine, URL, make_url
from sqlalchemy.orm import sessionmaker, Session

from omop_graph.config import (
    ENV_OMOP_CDM_DB_DRIVER,
    ENV_OMOP_CDM_DB_HOST,
    ENV_OMOP_CDM_DB_NAME,
    ENV_OMOP_CDM_DB_PASSWORD,
    ENV_OMOP_CDM_DB_PORT,
    ENV_OMOP_CDM_DB_URL,
    ENV_OMOP_CDM_DB_USER
)


def make_engine(
    url: Optional[Union[URL, str]] = None,
    *,
    echo: bool = False,
    connect_timeout: int = 10,
):
    url = url or build_engine_string()
    if isinstance(url, str):
        url = URL.create(url)

    kwargs = {}
    if not url.drivername.startswith("sqlite"):
        kwargs["connect_args"] = {"connect_timeout": connect_timeout}

    return create_engine(url, echo=echo, **kwargs)

def build_engine_string() -> "URL":
    """Compose a SQLAlchemy ``URL`` for the given backend at runtime.

    Returns
    -------
    sqlalchemy.URL

    Notes
    -----
    If ``OMOP_CDM_DB_URL`` is set it is directly used to create the URL, and all other environment variables are ignored. 
    Otherwise, the following environment variables are read to compose the URL for a relational database backend:
    - ``OMOP_CDM_DB_DRIVER`` (required): the SQLAlchemy driver name (e.g. 'postgresql', 'mysql', 'sqlite').
    - ``OMOP_CDM_DB_USER`` (required): the username for database authentication.
    - ``OMOP_CDM_DB_PASSWORD`` (required): the password for database authentication.
    - ``OMOP_CDM_DB_HOST`` (required): the hostname or IP address of the database server.
    - ``OMOP_CDM_DB_NAME`` (required): the name of the database to connect to.
    - ``OMOP_CDM_DB_PORT`` (optional, default 5432): the port number on which the database server is listening.

    Raises
    ------
    RuntimeError
        If a required environment variable is missing.
    ValueError
        If ``backend`` does not support URL composition from environment
        variables (e.g. ``FAISS``).
    """


    optional_url = os.getenv(ENV_OMOP_CDM_DB_URL)
    if optional_url:
        return make_url(optional_url)

    driver = _get_required_env_variable(ENV_OMOP_CDM_DB_DRIVER)
    user = _get_required_env_variable(ENV_OMOP_CDM_DB_USER)
    password = _get_required_env_variable(ENV_OMOP_CDM_DB_PASSWORD)
    host = _get_required_env_variable(ENV_OMOP_CDM_DB_HOST)
    database = _get_required_env_variable(ENV_OMOP_CDM_DB_NAME)
    port_str = os.getenv(ENV_OMOP_CDM_DB_PORT, "5432")
    port = int(port_str) if port_str else None
    return URL.create(
        drivername=driver,
        username=user,
        password=password,
        host=host,
        port=port,
        database=database,
    )


def make_session(
    url: str,
    *,
    echo: bool = False,
) -> Session:
    engine = make_engine(url, echo=echo)
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


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