"""Configuration for omop-graph via oa-configurator."""

from __future__ import annotations

from typing import Annotated, ClassVar, Final

from pydantic import Field
from oa_configurator import DatabaseConfig, PackageConfigBase, RefTo

TOOL_NAME: Final[str] = "omop_graph"


class OmopGraphConfig(PackageConfigBase):
    """oa-configurator config class for omop-graph.

    omop-graph does not own any database resources. It requires the CDM
    database configured by omop-alchemy, shared purely by naming convention:
    this field defaults to the same name as
    ``omop_alchemy.config.OmopAlchemyConfig.cdm_db``.
    """

    tool_name: ClassVar[str] = TOOL_NAME
    extra_logging_namespaces: ClassVar[tuple[str, ...]] = (
        "orm_loader",
        "omop_alchemy",
        "omop_emb",
    )

    cdm_db: Annotated[str, RefTo(DatabaseConfig)] = "cdm_db"

    max_depth: int = Field(
        default=6,
        description="Maximum graph traversal depth for pathfinding and grounding.",
    )
    max_paths: int = Field(
        default=20,
        description="Maximum number of shortest paths returned per query.",
    )
