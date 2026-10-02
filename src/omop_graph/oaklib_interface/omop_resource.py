from dataclasses import dataclass
from typing import Optional

from oaklib.resource import OntologyResource


@dataclass
class OMOPOntologyResource(OntologyResource):
    """
    Ontology resource naming an oa-configurator CDM database.

    Selected by oaklib as ``get_adapter("omop:<database_name>")``.

    Parameters
    ----------
    slug : str, optional
        Name of the ``[databases.*]`` entry to resolve. ``None`` resolves
        ``OmopGraphConfig.cdm_db``.
    scheme : str, optional
        The oaklib scheme. Defaults to 'omop'.
    readonly : bool, optional
        Whether the resource is read-only. Defaults to True.
    """

    slug: Optional[str] = None  # type: ignore[assignment]
    scheme: str = "omop"  # type: ignore[assignment]
    readonly: bool = True  # type: ignore[assignment]

    def valid(self) -> bool:
        """
        Always True: a missing slug resolves the configured default database.

        Returns
        -------
        bool
        """
        return True

    @property
    def local_path(self) -> None:
        """
        Return the local filesystem path.

        Returns
        -------
        None
            Always None as database-backed resources have no filesystem path.
        """
        return None
