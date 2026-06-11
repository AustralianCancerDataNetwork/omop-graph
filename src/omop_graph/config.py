"""Configuration for omop-graph via oa-configurator."""

from __future__ import annotations

from typing import ClassVar, Final

from pydantic import Field
from oa_configurator import PackageConfigBase, ResourceSpec
from omop_alchemy.config import OmopAlchemyConfig

TOOL_NAME: Final[str] = "omop_graph"


class OmopGraphConfig(PackageConfigBase):
    """oa-configurator config class for omop-graph.

    omop-graph does not own any database resources. It requires the CDM
    database configured by omop-alchemy.
    """

    tool_name: ClassVar[str] = TOOL_NAME
    extra_logging_namespaces: ClassVar[tuple[str, ...]] = ("orm_loader", "omop_alchemy", "omop_emb")
    required_resources: ClassVar[tuple[str, ...]] = (OmopAlchemyConfig.CDM_DB.semantic_name,)
    owned_resources: ClassVar[tuple[ResourceSpec, ...]] = ()

    max_depth: int = Field(
        default=6,
        description="Maximum graph traversal depth for pathfinding and grounding.",
    )
    max_paths: int = Field(
        default=20,
        description="Maximum number of shortest paths returned per query.",
    )


