"""The optional embedding store."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from oa_configurator import ResolvedModel

if TYPE_CHECKING:
    from omop_emb import (
        EmbeddingBackend,
        EmbeddingReaderInterface,
        EmbeddingStoreReader,
        EmbeddingWriterInterface,
        MetricType,
    )

# Local Application Imports
from ...extensions.emb import MissingExtensionError
from ._core import GraphCore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KnowledgeGraphEmbeddingConfiguration:
    """
    Configuration for embedding-based operations in the knowledge graph.

    A complete configuration: whenever embedding support is used at all
    (read or write), a real backend and a real resolved model are both
    required. The caller resolves the vector store and the model,
    builds the backend, and passes both in here. Used to enhance the
    knowledge graph with embedding-based grounding and similarity scoring.

    Parameters
    ----------
    metric_type : MetricType
        The similarity/distance metric to use for embedding comparisons (e.g., cosine, euclidean).
        This is required to ensure that the correct type of index is used in the backend and that
        similarity computations are consistent.
    backend : omop_emb.EmbeddingBackend or omop_emb.EmbeddingStoreReader
        An already-constructed embedding backend. A write-capable
        ``EmbeddingBackend`` (e.g. via ``omop_emb.backends.open_vector_store_writer``)
        is required when ``write=True``; the narrower read-only
        ``EmbeddingStoreReader`` (via ``open_vector_store_reader``) suffices
        when ``write=False``.
    resolved_model : oa_configurator.ResolvedModel
        A model resolved via ``oa_configurator.Resolver.resolve_model()``, carrying real
        provider connection details. ``model_name``/``provider_type`` (see below) are
        read directly off it; used to build the embedding model backend via
        ``omop_llm.build_model_backend_from_resolved`` when ``write=True``.
    write : bool
        If True, the KG holds a write-capable interface that can generate and persist
        embeddings. If False (default), the KG only holds a read-only interface over
        already-computed embeddings.
    compute_missing_embeddings : bool
        If True, the system will compute embeddings on-the-fly for any concept that is not yet present
        in the embedding store. Requires ``write=True`` as it needs a write-capable interface.
        Default: False
    faiss_cache_dir : str, optional
        Directory to cache FAISS index files, for the read-only path only. Passed
        straight through to ``EmbeddingReaderInterface``.
    """

    metric_type: MetricType
    backend: EmbeddingBackend | EmbeddingStoreReader
    resolved_model: ResolvedModel
    write: bool = False
    compute_missing_embeddings: bool = False
    faiss_cache_dir: str | None = None

    def __post_init__(self) -> None:
        if self.compute_missing_embeddings and not self.write:
            raise ValueError(
                "compute_missing_embeddings=True requires write=True "
                "(on-the-fly embedding computation needs a write-capable interface)."
            )

    @property
    def model_name(self) -> str:
        """The model name to use, read directly off ``resolved_model``."""
        return self.resolved_model.model

    @property
    def provider_type(self) -> str:
        """The provider key to use, read directly off ``resolved_model``."""
        return self.resolved_model.provider.provider


class EmbeddingsMixin(GraphCore):
    """The optional embedding store."""

    _emb_config: KnowledgeGraphEmbeddingConfiguration | None
    _emb: EmbeddingWriterInterface | EmbeddingReaderInterface | None

    @property
    def emb(self) -> EmbeddingWriterInterface | EmbeddingReaderInterface:
        """Namespace for all embedding operations.

        Returns an ``EmbeddingWriterInterface`` when ``self._emb_config.write`` is True,
        otherwise an ``EmbeddingReaderInterface`` (read-only).

        The interface/reader is created lazily on first access, from the
        already-constructed backend supplied via
        ``KnowledgeGraphEmbeddingConfiguration.backend``; omop-graph itself
        never resolves omop-emb's config.
        """
        if self._emb is not None:
            return self._emb

        try:
            from omop_emb.interface import (
                EmbeddingReaderInterface,
                EmbeddingWriterInterface,
            )

            if self._emb_config is None:
                raise ValueError(
                    "Embedding configuration is not set. Please provide an EmbeddingConfiguration when initializing the KnowledgeGraph to use embedding features."
                )

            if self._emb_config.write:
                from omop_emb import EmbeddingBackend

                if not isinstance(self._emb_config.backend, EmbeddingBackend):
                    raise ValueError(
                        "write=True requires a write-capable EmbeddingBackend, "
                        f"got {type(self._emb_config.backend).__name__}."
                    )
                # Write-capable interface: the KG builds its own embedding model backend.
                self._emb = EmbeddingWriterInterface(
                    backend=self._emb_config.backend,
                    metric_type=self._emb_config.metric_type,
                    resolved_model=self._emb_config.resolved_model,
                    cdm_session_factory=self.session_factory,
                )
            else:
                # Read-only interface: only ever needs model identity, never live credentials.
                self._emb = EmbeddingReaderInterface(
                    model=self._emb_config.model_name,
                    backend=self._emb_config.backend,
                    metric_type=self._emb_config.metric_type,
                    cdm_session_factory=self.session_factory,
                    provider_type=self._emb_config.provider_type,
                    faiss_cache_dir=self._emb_config.faiss_cache_dir,
                )
            logger.debug(
                "Constructed %s embedding interface for model %r (write=%s).",
                "write-capable" if self._emb_config.write else "read-only",
                self._emb_config.model_name,
                self._emb_config.write,
            )
            return self._emb

        except ModuleNotFoundError as e:
            if e.name and e.name.startswith("omop_emb"):
                logger.info(
                    "Embedding functionality is not available because the optional 'omop-emb' package is not installed."
                )
                raise MissingExtensionError() from e
            raise
        except ImportError:
            logger.info(
                "Embedding functionality failed to initialize due to an import error in the optional embedding stack."
            )
            raise

    @property
    def embedding_configuration(self) -> KnowledgeGraphEmbeddingConfiguration | None:
        """Returns the current embedding configuration, if set."""
        return self._emb_config

    @property
    def compute_missing_embeddings(self) -> bool:
        """Indicates whether on-the-fly computation of missing concept embeddings is enabled."""
        return self._emb_config.compute_missing_embeddings if self._emb_config else False
