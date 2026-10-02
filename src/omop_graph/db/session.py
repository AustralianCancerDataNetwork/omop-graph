"""Resolution helper for the OMOP CDM database."""

from __future__ import annotations

from typing import Optional

from oa_configurator import ResolvedCDMDatabase, Resolver
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
