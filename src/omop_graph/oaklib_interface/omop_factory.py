import os
from typing import Optional, Union

from sqlalchemy.engine import URL

from .omop_resource import OMOPOntologyResource

OMOP_DATABASE_ENV_VAR = "OMOP_DATABASE_URL"


def omop_resource(
    *,
    url: Optional[Union[str, URL]] = None,
    env_var: str = OMOP_DATABASE_ENV_VAR,
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
        Defaults to 'OMOP_DATABASE_URL'.
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
    resolved = url or os.getenv(env_var)

    if not resolved:
        raise ValueError(
            f"No database URL provided and environment variable {env_var} is not set"
        )

    return OMOPOntologyResource(
        slug=slug,
        url=resolved,
    )