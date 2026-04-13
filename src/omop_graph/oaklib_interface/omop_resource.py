from dataclasses import dataclass
from typing import Optional, Union

from oaklib.resource import OntologyResource
from sqlalchemy.engine import URL, make_url


@dataclass
class OMOPOntologyResource(OntologyResource):
    """
    Ontology resource backed by a live SQLAlchemy database.

    This class extends the `OntologyResource` to support database URLs specifically
    for OMOP backends using SQLAlchemy.

    Parameters
    ----------
    url : str | URL, optional
        The database connection URL.
    slug : str, optional
        A unique slug/identifier for this resource.
    scheme : str, optional
        The scheme identifier. Defaults to 'omop_alchemy'.
    local : bool, optional
        Whether the resource is local. Defaults to False.
    in_memory : bool, optional
        Whether the resource is in-memory. Defaults to False.
    readonly : bool, optional
        Whether the resource is read-only. Defaults to True.
    """

    url: Optional[Union[str, URL]] = None
    slug: Optional[str] = None
    scheme: str = "omop_alchemy"
    local: bool = False
    in_memory: bool = False
    readonly: bool = True

    def _parsed_url(self) -> Optional[URL]:
        """
        Parse the connection URL into a SQLAlchemy URL object.

        Returns
        -------
        URL | None
            The parsed URL object, or None if no URL is set.
        """
        if not self.url:
            return None
        return make_url(self.url) if isinstance(self.url, str) else self.url

    @property
    def display_slug(self) -> Optional[str]:
        """
        Get a safe, redacted identifier for logs / UI.

        Returns
        -------
        str | None
            The string representation of the parsed URL.
        """
        u = self._parsed_url()
        return str(u) if u else None

    def valid(self) -> bool:
        """
        Check if the database ontology resource is valid.

        Returns
        -------
        bool
            True if a URL is present, False otherwise.
        """
        return bool(self.url)

    @property
    def local_path(self) -> None:
        """
        Return the local filesystem path.

        Returns
        -------
        None
            Always returns None as database-backed resources have no filesystem path.
        """
        return None

    def __repr__(self) -> str:
        parts = []
        if self.slug:
            parts.append(f"slug={self.slug!r}")
        if self.url:
            parts.append(f"url={str(make_url(self.url))!r}")
        parts.append(f"scheme={self.scheme!r}")
        return f"{type(self).__name__}({', '.join(parts)})"