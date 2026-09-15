"""Factory for creating OAK OMOP ontology resources."""

from __future__ import annotations

import logging
from typing import Optional, Union

from sqlalchemy.engine import URL

from .omop_resource import OMOPOntologyResource
from oa_configurator import SCHEMA_TRANSLATE_MAP_KEY, ResolvedCDMDatabase, Resolver
from omop_graph.config import OmopGraphConfig

logger = logging.getLogger(__name__)


def omop_resource(
    *,
    url: Optional[Union[str, URL]] = None,
    slug: Optional[str] = "omop",
) -> OMOPOntologyResource:
    """Create an OMOP DatabaseOntologyResource.

    When url is omitted, reads connection details from the active oa-configurator
    stack config. Pass url explicitly to override.

    Parameters
    ----------
    url : str | URL, optional
        Explicit database connection URL. When omitted the active oa-configurator
        config is used. An explicit url bypasses oa-configurator entirely, so no
        schema_translate_map is applied. Logs a warning when given for this reason.
    slug : str, optional
        Slug identifier for the resource. Defaults to 'omop'.

    Returns
    -------
    OMOPOntologyResource
    """
    execution_options = None
    vocab_url = None
    vocab_execution_options = None
    resolved: ResolvedCDMDatabase | None = None
    if url is None:
        resolver = Resolver.from_active_config()
        db_name = resolver.resolve_package_config(OmopGraphConfig).cdm_db
        database = resolver.resolve_database(db_name)
        if not isinstance(database, ResolvedCDMDatabase):
            raise TypeError(
                f"OmopGraphConfig.cdm_db must resolve to a CDM database, got "
                f"{type(database).__name__}"
            )
        resolved = database
        url = resolved.connection.url
        execution_options = {SCHEMA_TRANSLATE_MAP_KEY: resolved.schema_translate_map()}
        if resolved.connection != resolved.vocab_connection:
            vocab_url = resolved.vocab_connection.url
            vocab_execution_options = execution_options
    else:
        logger.warning(
            "omop_resource() was given an explicit url, bypassing oa-configurator "
            "entirely: no schema_translate_map is applied, so a configured "
            "non-default primary/vocab/results schema is silently not respected. "
            "Omit url= to resolve the active oa-configurator config instead."
        )

    return OMOPOntologyResource(
        slug=slug,
        url=url,
        execution_options=execution_options,
        vocab_url=vocab_url,
        vocab_execution_options=vocab_execution_options,
        resolved=resolved,
    )
