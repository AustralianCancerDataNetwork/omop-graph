"""
The optional embedding store of the knowledge graph.

This module holds the whole boundary to the optional ``omop-emb`` package: the flag that
says whether it is installed, the errors, the guards that raise them, the configuration
and the capability mixin. It imports nothing from ``omop_graph.graph.kg``, and ``omop-emb``
types appear in annotations only, so importing it never imports ``omop-emb``.
"""

from __future__ import annotations

import functools
import importlib.util
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Concatenate, cast

import numpy as np
from omop_alchemy.cdm.query import ConceptFilter
from sqlalchemy import Row

from ..model.scope import ALL_CONCEPTS, _require_scope
from ._core import GraphCore

if TYPE_CHECKING:
    from oa_configurator import ResolvedModel
    from omop_emb import (
        EmbeddingBackend,
        EmbeddingModelRecord,
        EmbeddingReaderInterface,
        EmbeddingStoreReader,
        EmbeddingWriterInterface,
        MetricType,
        NearestConceptMatch,
    )
    from omop_emb.utils.embedding_utils import EmbeddingConceptFilter

logger = logging.getLogger(__name__)


HAS_OMOP_EMB = importlib.util.find_spec("omop_emb") is not None
"""Whether the optional ``omop-emb`` package is installed. It is imported on first use."""


def _import_omop_emb() -> None:
    """Import ``omop-emb``, raising :class:`MissingExtensionError` when it does not import."""
    try:
        importlib.import_module("omop_emb")
    except ImportError as error:
        raise MissingExtensionError() from error


class EmbeddingError(Exception):
    """Base of the errors raised when an embedding capability cannot be provided."""


class MissingExtensionError(EmbeddingError, ImportError):
    """Raised when the optional ``omop-emb`` package is required but not installed."""

    def __init__(self, feature: str = "Embedding functionality"):
        super().__init__(
            f"{feature} requires the 'omop-emb' package. "
            "Install it via: pip install omop-graph[emb]"
        )


class EmbeddingsNotConfiguredError(EmbeddingError):
    """Raised when an embedding capability is used on a graph without an embedding configuration."""

    def __init__(self) -> None:
        super().__init__(
            "The knowledge graph has no embedding configuration. Pass a "
            "KnowledgeGraphEmbeddingConfiguration as emb_config to use embeddings."
        )


class EmbeddingWriterRequiredError(EmbeddingError):
    """Raised when a capability needs a write-capable embedding configuration."""

    def __init__(self, capability: str = "This operation") -> None:
        super().__init__(
            f"{capability} needs a write-capable embedding configuration. "
            "Configure the knowledge graph with write=True."
        )


class EmbeddingBackendNotWritableError(EmbeddingError):
    """Raised when ``write=True`` is configured with a backend that cannot write embeddings."""

    def __init__(self, backend: object) -> None:
        super().__init__(
            f"write=True needs a backend that can write embeddings, and {type(backend).__name__} "
            "cannot. Open the vector store with omop_emb.backends.open_vector_store_writer."
        )


class EmbeddingModelNotRegisteredError(EmbeddingError):
    """Raised when the configured embedding model has no registration in the vector store."""

    def __init__(self, model_name: str) -> None:
        super().__init__(
            f"The embedding model {model_name!r} is not registered in the vector store."
        )


class MissingEmbeddingError(EmbeddingError):
    """Raised when concepts have no stored embedding for the configured model."""

    def __init__(self, concept_ids: Iterable[int]) -> None:
        self.concept_ids = tuple(concept_ids)
        super().__init__(f"No stored embedding for concepts {list(self.concept_ids)}.")


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


def requires_embeddings[S: EmbeddingsMixin, **P, R](
    method: Callable[Concatenate[S, P], R],
) -> Callable[Concatenate[S, P], R]:
    """
    Guard a method that needs the embedding store, checked when it is called.

    Parameters
    ----------
    method : Callable
        The method to guard.

    Returns
    -------
    Callable
        The method, which checks the embedding store first.

    Raises
    ------
    MissingExtensionError
        If ``omop-emb`` is not installed.
    EmbeddingsNotConfiguredError
        If the graph has no embedding configuration.
    """

    @functools.wraps(method)
    def guarded(self: S, *args: P.args, **kwargs: P.kwargs) -> R:
        self._check_embeddings()
        return method(self, *args, **kwargs)

    return guarded


def requires_embedding_writer[S: EmbeddingsMixin, **P, R](
    method: Callable[Concatenate[S, P], R],
) -> Callable[Concatenate[S, P], R]:
    """
    Guard a method that needs a write-capable embedding store, checked when it is called.

    Parameters
    ----------
    method : Callable
        The method to guard.

    Returns
    -------
    Callable
        The method, which checks that the embedding store can write first.

    Raises
    ------
    MissingExtensionError
        If ``omop-emb`` is not installed.
    EmbeddingsNotConfiguredError
        If the graph has no embedding configuration.
    EmbeddingWriterRequiredError
        If the configuration is read-only.
    """

    @functools.wraps(method)
    def guarded(self: S, *args: P.args, **kwargs: P.kwargs) -> R:
        self._check_embedding_writer()
        return method(self, *args, **kwargs)

    return guarded


def embedding_filter(concept_filter: ConceptFilter) -> EmbeddingConceptFilter:
    """
    Translate a concept scope into the filter that omop-emb applies during retrieval.

    omop-emb can require standard concepts but has no flag for classification concepts, so
    ``include_classification`` is not passed on. A scope that requires standard concepts
    retrieves standard concepts only, which are the valid mapping targets.

    Parameters
    ----------
    concept_filter : ConceptFilter
        The scope to translate.

    Returns
    -------
    EmbeddingConceptFilter
        The filter for omop-emb.
    """
    from omop_emb.utils.embedding_utils import EmbeddingConceptFilter

    return EmbeddingConceptFilter(
        concept_ids=concept_filter.concept_ids,
        domains=concept_filter.domains,
        vocabularies=concept_filter.vocabularies,
        require_standard=concept_filter.require_standard,
        require_active=concept_filter.require_active,
    )


class EmbeddingsMixin(GraphCore):
    """
    Nearest concepts and similarity over the stored concept embeddings.

    Every method that needs the store raises a typed :class:`EmbeddingError` when it is
    not available, and never returns None. Callers that degrade gracefully ask
    :attr:`has_embeddings` and :attr:`can_embed_text` first.
    """

    _emb_config: KnowledgeGraphEmbeddingConfiguration | None = None
    _emb: EmbeddingWriterInterface | EmbeddingReaderInterface | None = None
    _model_registered: bool = False

    @property
    def embedding_configuration(self) -> KnowledgeGraphEmbeddingConfiguration | None:
        """Returns the current embedding configuration, if set."""
        return self._emb_config

    @property
    def compute_missing_embeddings(self) -> bool:
        """Indicates whether on-the-fly computation of missing concept embeddings is enabled."""
        return self._emb_config.compute_missing_embeddings if self._emb_config else False

    @property
    def has_embeddings(self) -> bool:
        """Whether ``omop-emb`` is installed and the graph has an embedding configuration."""
        return HAS_OMOP_EMB and self._emb_config is not None

    @property
    def can_embed_text(self) -> bool:
        """Whether the graph can embed text, which needs a write-capable configuration."""
        return (
            self.has_embeddings
            and cast("KnowledgeGraphEmbeddingConfiguration", self._emb_config).write
        )

    def _check_embeddings(self) -> None:
        """Raise the error that names why the embedding store is unavailable."""
        if not HAS_OMOP_EMB:
            raise MissingExtensionError()
        _import_omop_emb()
        if self._emb_config is None:
            raise EmbeddingsNotConfiguredError()

    def _check_embedding_writer(self, capability: str = "This operation") -> None:
        """Raise the error that names why a write-capable embedding store is unavailable."""
        self._check_embeddings()
        assert self._emb_config is not None
        if not self._emb_config.write:
            raise EmbeddingWriterRequiredError(capability)

    @property
    @requires_embeddings
    def emb(self) -> EmbeddingWriterInterface | EmbeddingReaderInterface:
        """
        The omop-emb interface of the configured model.

        A writer when the configuration has ``write=True``, otherwise a read-only reader.
        It is built on first access from the backend of the configuration.

        Raises
        ------
        MissingExtensionError
            If ``omop-emb`` is not installed.
        EmbeddingsNotConfiguredError
            If the graph has no embedding configuration.
        EmbeddingBackendNotWritableError
            If ``write=True`` but the backend cannot write embeddings.
        """
        if self._emb is not None:
            return self._emb
        from omop_emb import EmbeddingBackend
        from omop_emb.interface import EmbeddingReaderInterface, EmbeddingWriterInterface

        config = self._emb_config
        assert config is not None
        if config.write:
            if not isinstance(config.backend, EmbeddingBackend):
                raise EmbeddingBackendNotWritableError(config.backend)
            self._emb = EmbeddingWriterInterface(
                backend=config.backend,
                metric_type=config.metric_type,
                resolved_model=config.resolved_model,
                cdm_session_factory=self.session_factory,
            )
        else:
            self._emb = EmbeddingReaderInterface(
                model=config.model_name,
                backend=config.backend,
                metric_type=config.metric_type,
                cdm_session_factory=self.session_factory,
                provider_type=config.provider_type,
                faiss_cache_dir=config.faiss_cache_dir,
            )
        logger.debug(
            "Constructed %s embedding interface for model %r.",
            "write-capable" if config.write else "read-only",
            config.model_name,
        )
        return self._emb

    def _registered(self) -> EmbeddingReaderInterface:
        """The interface, once its model is known to be registered in the store."""
        interface = self.emb
        if not self._model_registered:
            if not interface.is_model_registered():
                raise EmbeddingModelNotRegisteredError(interface.canonical_model_name)
            self._model_registered = True
        return interface

    @requires_embeddings
    def embedding_models(self) -> tuple[EmbeddingModelRecord, ...]:
        """Return the embedding models registered in the vector store."""
        return self.emb.backend.get_registered_models()

    @requires_embedding_writer
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a query text with the configured model.

        Parameters
        ----------
        text : str
            The text to embed.

        Returns
        -------
        np.ndarray
            The vector, of shape ``(1, D)``.
        """
        from omop_emb import EmbeddingRole

        writer = cast("EmbeddingWriterInterface", self.emb)
        return writer.embed_texts((text,), role=EmbeddingRole.QUERY)

    @requires_embeddings
    def nearest_concepts(
        self,
        query: np.ndarray | int | str,
        *,
        concept_filter: ConceptFilter = ALL_CONCEPTS,
        k: int | None = None,
    ) -> tuple[NearestConceptMatch, ...]:
        """
        Find the concepts whose stored embeddings are nearest to a query.

        Parameters
        ----------
        query : np.ndarray | int | str
            A vector of shape ``(D,)`` or ``(1, D)``, the id of a concept with a stored
            embedding, or a text, which is embedded first and so needs a write-capable
            configuration.
        concept_filter : ConceptFilter
            The concepts that can match, a scope that must not set ``limit``. Use k to
            truncate.
        k : int, optional
            The most matches to return, the default of the store when omitted.

        Returns
        -------
        tuple[NearestConceptMatch, ...]
            The matches, nearest first. Empty when nothing matches.

        Raises
        ------
        EmbeddingModelNotRegisteredError
            If the configured model is not registered in the store.
        EmbeddingWriterRequiredError
            If query is a text and the configuration is read-only.
        ValueError
            If a vector query holds more than one vector.
        """
        _require_scope(concept_filter)
        if isinstance(query, str):
            self._check_embedding_writer("Searching by text")
            query = self.embed_text(query)
        reader = self._registered()
        if isinstance(query, int | np.integer):
            return reader.get_similar_concepts(
                int(query), k, concept_filter=embedding_filter(concept_filter)
            )[0]
        vector = np.atleast_2d(query)
        if vector.shape[0] != 1:
            raise ValueError(f"nearest_concepts takes one query vector, got {vector.shape[0]}.")
        return reader.get_nearest_concepts(
            vector, concept_filter=embedding_filter(concept_filter), k=k
        )[0]

    @requires_embeddings
    def concept_embeddings(self, concept_ids: Iterable[int]) -> dict[int, np.ndarray]:
        """
        Return the stored embedding of each concept that has one.

        Concepts without a stored embedding are left out.

        Parameters
        ----------
        concept_ids : Iterable[int]
            The concepts to read.

        Returns
        -------
        dict[int, np.ndarray]
            The stored vector of each concept that has one.
        """
        stored = self._stored_concept_ids(concept_ids)
        return dict(self._registered().get_embeddings_by_concept_ids(stored)) if stored else {}

    def _stored_concept_ids(self, concept_ids: Iterable[int]) -> tuple[int, ...]:
        """
        Return the concepts among concept_ids that have a stored embedding, in order.

        The store raises for a requested concept without a vector, so the vectors are only
        requested for the concepts this lookup returns.
        """
        from omop_emb.utils.embedding_utils import EmbeddingConceptFilter

        ids = tuple(dict.fromkeys(concept_ids))
        if not ids:
            return ()
        interface = self._registered()
        stored: set[int] = set()
        for batch in self.batches(ids):
            stored |= interface.backend.get_stored_concept_ids(
                model_name=interface.canonical_model_name,
                concept_filter=EmbeddingConceptFilter(concept_ids=batch),
            )
        return tuple(concept_id for concept_id in ids if concept_id in stored)

    @requires_embeddings
    def concepts_without_embedding(self, concept_ids: Iterable[int]) -> tuple[int, ...]:
        """
        Return the concepts among concept_ids that exist and have no stored embedding.

        Parameters
        ----------
        concept_ids : Iterable[int]
            The concepts to check.

        Returns
        -------
        tuple[int, ...]
            The concepts that exist and have no stored embedding.
        """
        return tuple(self._rows_without_embedding(concept_ids))

    def _rows_without_embedding(self, concept_ids: Iterable[int]) -> dict[int, Row]:
        """
        Read the CDM rows of the concepts among concept_ids that have no stored embedding.

        The stored ids are asked per batch of ids, see :meth:`_stored_concept_ids`, so the
        store never lists every id it holds.
        """
        from omop_emb.utils.cdm import fetch_cdm_concepts_for_filter

        ids = tuple(dict.fromkeys(concept_ids))
        stored = set(self._stored_concept_ids(ids))
        rows: dict[int, Row] = {}
        for batch in self.batches(concept_id for concept_id in ids if concept_id not in stored):
            rows.update(
                fetch_cdm_concepts_for_filter(
                    ConceptFilter(concept_ids=batch), cdm_session_factory=self.session_factory
                )
            )
        return rows

    @requires_embedding_writer
    def embed_concepts(self, concept_ids: Iterable[int]) -> tuple[int, ...]:
        """
        Compute and store the embeddings of the concepts that have none.

        Parameters
        ----------
        concept_ids : Iterable[int]
            The concepts to embed when they have no embedding.

        Returns
        -------
        tuple[int, ...]
            The concepts that were embedded.
        """
        missing = self._rows_without_embedding(concept_ids)
        if not missing:
            return ()
        writer = cast("EmbeddingWriterInterface", self.emb)
        missing_ids = tuple(missing)
        writer.embed_and_upsert_concepts(
            concept_ids=missing_ids,
            concept_texts=tuple(row.concept_name for row in missing.values()),
            concept_meta=missing,
        )
        return missing_ids

    @requires_embeddings
    def embedding_similarity(self, concept_a: int, concept_b: int) -> float:
        """
        Return the cosine similarity of the stored embeddings of two concepts.

        Parameters
        ----------
        concept_a : int
            The first concept.
        concept_b : int
            The second concept.

        Returns
        -------
        float
            The cosine similarity of the two stored vectors.

        Raises
        ------
        MissingEmbeddingError
            If either concept has no stored embedding.
        """
        vectors = self.concept_embeddings((concept_a, concept_b))
        absent = [concept_id for concept_id in (concept_a, concept_b) if concept_id not in vectors]
        if absent:
            raise MissingEmbeddingError(absent)
        a, b = vectors[concept_a], vectors[concept_b]
        norm = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b)) / norm if norm else 0.0
