"""Resolution helper for the OMOP CDM database."""

from __future__ import annotations

from typing import Optional
from collections.abc import Generator
from contextlib import contextmanager
import sqlalchemy.orm as so
from sqlalchemy.engine import Engine
from oa_configurator import ResolvedCDMDatabase, Resolver
from omop_alchemy.config import create_cdm_engines
from omop_alchemy.cross_database import CDMSession, cdm_sessionmaker
from sqlalchemy.orm import sessionmaker

from omop_graph.config import OmopGraphConfig


def resolve_cdm_database(name: Optional[str] = None) -> ResolvedCDMDatabase:
    """Resolve a CDM database from the active oa-configurator config.

    Parameters
    ----------
    name : str, optional
        Name of the ``[databases.*]`` entry. Defaults to ``OmopGraphConfig.cdm_db``.

    Returns
    -------
    ResolvedCDMDatabase

    Raises
    ------
    TypeError
        If the entry is not a CDM database.
    """
    resolver = Resolver.from_active_config()
    if name is None:
        name = resolver.resolve_package_config(OmopGraphConfig).cdm_db
    resolved = resolver.resolve_database(name)
    if not isinstance(resolved, ResolvedCDMDatabase):
        raise TypeError(
            f"Database {name!r} must resolve to a CDM database, got {type(resolved).__name__}"
        )
    return resolved


def cdm_session_factory(
    name: Optional[str] = None, *, resolved: ResolvedCDMDatabase | None = None
) -> sessionmaker[CDMSession]:
    """Read-only routed session factory on a CDM database from the active config.

    Builds the engine pair with ``create_cdm_engines()``, checking its schema
    claims without registering them. The engines live as long as the process,
    so this suits scripts that run once.

    Parameters
    ----------
    name : str, optional
        As for :func:`resolve_cdm_database`.
    """
    resolved = resolved or resolve_cdm_database(name)
    return _create_cdm_sessions(resolved, register_claims=False)[0]


def _create_cdm_sessions(
    resolved: ResolvedCDMDatabase, *, register_claims: bool
) -> tuple[sessionmaker[CDMSession], Engine, Engine]:
    """Build routed sessions and their owned engine pair through omop-alchemy."""
    primary, vocab = create_cdm_engines(
        resolved, register_claims=register_claims
    )
    return cdm_sessionmaker(resolved, primary=primary, vocab=vocab), primary, vocab


@contextmanager
def open_cdm_sessions(
    resolved: ResolvedCDMDatabase, *, register_claims: bool = True
) -> Generator[so.sessionmaker[CDMSession], None, None]:
    """Yield a routed session factory on ``create_cdm_engines(resolved)``, disposing both engines on exit.

    Parameters
    ----------
    resolved : ResolvedCDMDatabase
    register_claims : bool, optional
        As for :func:`create_cdm_engines`.

    Yields
    ------
    sqlalchemy.orm.sessionmaker[CDMSession]
    """
    cdm_sessions, primary, vocab = _create_cdm_sessions(
        resolved, register_claims=register_claims
    )
    try:
        yield cdm_sessions
    finally:
        for engine in {primary, vocab}:
            engine.dispose()
