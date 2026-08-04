"""Factory for creating OAK OMOP ontology resources."""

from __future__ import annotations

from typing import Optional, Union

from sqlalchemy.engine import URL

from .omop_resource import OMOPOntologyResource
from oa_configurator import Resolver
from omop_graph.config import OmopGraphConfig


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
        config is used.
    slug : str, optional
        Slug identifier for the resource. Defaults to 'omop'.

    Returns
    -------
    OMOPOntologyResource
    """
    if url is None:
        resolver = Resolver.from_active_config()
        db_name = resolver.resolve_package_config(OmopGraphConfig).cdm_db
        database = resolver.resolve_database(db_name)
        url = database.connection.url

    return OMOPOntologyResource(
        slug=slug,
        url=url,
    )
