"""SQLAlchemy engine helper for the OMOP CDM database."""

from __future__ import annotations

from typing import Optional, Union

from sqlalchemy import create_engine, URL
from sqlalchemy.orm import sessionmaker, Session

from oa_configurator import Resolver
from omop_alchemy.config import OmopAlchemyConfig


def get_engine():
    """Return a SQLAlchemy engine for the CDM database via oa-configurator."""
    return Resolver.from_active_config().resolve_resource(OmopAlchemyConfig.CDM_DB.semantic_name).create_engine()


def make_engine(
    url: Optional[Union[URL, str]] = None,
    *,
    echo: bool = False,
    connect_timeout: int = 10,
):
    """Return a SQLAlchemy engine.

    When url is omitted, reads connection details from the active oa-configurator
    stack config. Pass url explicitly to override.
    """
    if url is None:
        return get_engine()
    from sqlalchemy import make_url as _make_url
    if isinstance(url, str):
        url = _make_url(url)
    kwargs = {}
    if not url.drivername.startswith("sqlite"):
        kwargs["connect_args"] = {"connect_timeout": connect_timeout}
    return create_engine(url, echo=echo, **kwargs)


def make_session(
    url: str,
    *,
    echo: bool = False,
) -> Session:
    engine = make_engine(url, echo=echo)
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()
