"""Configuration for omop-graph via oa-configurator."""

from __future__ import annotations

from typing import ClassVar, Final

from pydantic import Field
from oa_configurator import PackageConfigBase, ResourceSpec, Resolver, load_stack_config
from oa_configurator import configure_logging as _configure_logging
from omop_alchemy.config import CDM_DB_RESOURCE

TOOL_NAME: Final[str] = "omop_graph"


class OmopGraphConfig(PackageConfigBase):
    """oa-configurator config class for omop-graph.

    omop-graph does not own any database resources. It requires the CDM
    database configured by omop-alchemy.
    """

    tool_name: ClassVar[str] = TOOL_NAME
    required_resources: ClassVar[tuple[str, ...]] = (CDM_DB_RESOURCE,)
    owned_resources: ClassVar[tuple[ResourceSpec, ...]] = ()

    max_depth: int = Field(
        default=6,
        description="Maximum graph traversal depth for pathfinding and grounding.",
    )
    max_paths: int = Field(
        default=20,
        description="Maximum number of shortest paths returned per query.",
    )


def get_resolver() -> Resolver:
    """Return a Resolver loaded from the active stack config."""
    return Resolver(load_stack_config())


def get_config() -> OmopGraphConfig:
    """Return the omop-graph typed config from the active stack config."""
    return OmopGraphConfig.from_stack(load_stack_config())


def configure_logging(verbosity: int = 0) -> None:
    """Configure logging for omop-graph and its dependencies."""
    _configure_logging(verbosity=verbosity, extra_namespaces=["omop_alchemy", "omop_emb", TOOL_NAME])
