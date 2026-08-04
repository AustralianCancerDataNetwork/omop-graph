"""SQLAlchemy engine helper for the OMOP CDM database."""

from __future__ import annotations

from typing import Optional, Union

from sqlalchemy import create_engine, URL, Engine
from sqlalchemy.orm import sessionmaker, Session

from oa_configurator import Resolver
from omop_graph.config import OmopGraphConfig


def make_engine(
    url: Optional[Union[URL, str]] = None,
    *,
    engine_kwargs: Optional[dict] = None,
    execution_options: Optional[dict] = None,
) -> Engine:
    """Return a SQLAlchemy engine.

    When url is omitted, reads connection details from the active oa-configurator
    stack config (schema translate map applied automatically). Pass url explicitly
    to override.

    Parameters
    ----------
    url : URL or str, optional
        SQLAlchemy database URL. If None, resolved from the active oa-configurator config.
    engine_kwargs : dict, optional
        Keyword arguments forwarded to ``sqlalchemy.create_engine`` in both paths.
        Common keys: ``echo``, ``connect_args``, ``pool_size``.
    execution_options : dict, optional
        Options forwarded to ``engine.execution_options()``. In the resolver path these
        are merged with the auto-generated ``schema_translate_map`` (resolver wins on
        that key via ``setdefault``).

    Returns
    -------
    Engine
        A SQLAlchemy engine instance.
    """
    engine_kwargs = engine_kwargs or {}
    if url is None:
        resolver = Resolver.from_active_config()
        db_name = resolver.resolve_package_config(OmopGraphConfig).cdm_db
        database = resolver.resolve_database(db_name)
        return database.create_engine(execution_options=execution_options, **engine_kwargs)

    from sqlalchemy import make_url as _make_url

    if isinstance(url, str):
        url = _make_url(url)
    engine = create_engine(url, **engine_kwargs)
    if execution_options:
        engine = engine.execution_options(**execution_options)
    return engine


def make_session(
    url: str,
    *,
    echo: bool = False,
) -> Session:
    engine = make_engine(url, engine_kwargs={"echo": echo})
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()
