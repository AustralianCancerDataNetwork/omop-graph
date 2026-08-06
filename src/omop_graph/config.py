"""Configuration for omop-graph via oa-configurator."""

from __future__ import annotations

from typing import Annotated, ClassVar, Final

from pydantic import Field
from oa_configurator import CDMDatabaseConfig, ModelConfig, PackageConfigBase, RefTo, VectorStoreConfig

TOOL_NAME: Final[str] = "omop_graph"


class OmopGraphConfig(PackageConfigBase):
    """oa-configurator config class for omop-graph.

    omop-graph does not own any database resources. It requires the CDM
    database configured by omop-alchemy, shared purely by naming convention:
    this field defaults to the same name as
    ``omop_alchemy.config.OmopAlchemyConfig.cdm_db``.

    Notes
    -----
    By design, this config is for internal use only and must not be
    imported or resolved by any other package. Embedding support itself is
    entirely caller-supplied: at runtime, omop-graph knows nothing of "its"
    embedding configuration, only the already-built
    ``omop_graph.graph.kg.KnowledgeGraphEmbeddingConfiguration`` a caller
    passes in. ``embedding_model_name``/``vector_store_name`` below exist
    for internal use in a CLI boundary of this module. Not consumed as of today.
    """

    tool_name: ClassVar[str] = TOOL_NAME
    extra_logging_namespaces: ClassVar[tuple[str, ...]] = (
        "orm_loader",
        "omop_alchemy",
        "omop_emb",
    )

    cdm_db: Annotated[str, RefTo(CDMDatabaseConfig)] = "cdm_db"
    embedding_model_name: Annotated[str | None, RefTo(ModelConfig)] = Field(
        default=None,
        description=(
            "Name of a [models.*] entry for embedding-based grounding. Optional: "
            "embedding support is itself optional (gated behind the omop-emb "
            "extension), so this has no default name the way cdm_db does. Unset "
            "means no embedding-based grounding by default."
        ),
    )
    vector_store_name: Annotated[str | None, RefTo(VectorStoreConfig)] = Field(
        default=None,
        description=(
            "Name of a [vector_stores.*] entry for embedding-based grounding. "
            "Optional, same reasoning as embedding_model_name."
        ),
    )

    max_depth: int = Field(
        default=6,
        description="Maximum graph traversal depth for pathfinding and grounding.",
    )
    max_paths: int = Field(
        default=20,
        description="Maximum number of shortest paths returned per query.",
    )
