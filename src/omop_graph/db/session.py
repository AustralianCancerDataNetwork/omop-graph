"""SQLAlchemy engine helper for the OMOP CDM database."""

from __future__ import annotations

from typing import Optional, Union

from sqlalchemy import create_engine, URL, Engine
from sqlalchemy.orm import sessionmaker, Session

from oa_configurator import ResolvedCDMDatabase, Resolver
from omop_graph.config import OmopGraphConfig


def resolve_cdm_database() -> ResolvedCDMDatabase:
    """Resolve the active oa-configurator config's CDM database.

    Split out from make_engine() for callers that need the resolved object
    itself (e.g. schema-provenance guarding, Phase 9), not just an engine.
    """
    resolver = Resolver.from_active_config()
    db_name = resolver.resolve_package_config(OmopGraphConfig).cdm_db
    resolved = resolver.resolve_database(db_name)
    if not isinstance(resolved, ResolvedCDMDatabase):
        raise TypeError(
            f"OmopGraphConfig.cdm_db must resolve to a CDM database, got {type(resolved).__name__}"
        )
    return resolved


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
        Options forwarded to ``engine.execution_options()``. In the resolver
        path, a ``schema_translate_map`` here may add keys the resolver
        doesn't define, but may not include ``None``: that key is always set
        from the resolved config, and ``create_engine()`` raises
        ``ValueError`` if it is overridden here.

    Returns
    -------
    Engine
        A SQLAlchemy engine instance.
    """
    engine_kwargs = engine_kwargs or {}
    if url is None:
        database = resolve_cdm_database()
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
