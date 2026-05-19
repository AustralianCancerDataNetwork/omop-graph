from __future__ import annotations

import logging
import importlib.util
from typing import TYPE_CHECKING, Optional, Sequence, TypeAlias, Tuple
import numpy as np

HAS_OMOP_EMB = importlib.util.find_spec("omop_emb") is not None

if TYPE_CHECKING:
    # Optional embedding-specific ones
    from omop_emb import BackendType, MetricType, IndexType, ProviderType
    from omop_emb import EmbeddingWriterInterface, EmbeddingReaderInterface
    from omop_emb.embeddings import EmbeddingRole
    from omop_emb.utils.embedding_utils import NearestConceptMatch
    from omop_emb.utils.embedding_utils import EmbeddingConceptFilter


    EmbeddingBackendType: TypeAlias = BackendType
    EmbeddingMetricType: TypeAlias = MetricType
    EmbeddingIndexType: TypeAlias = IndexType
    EmbeddingProviderType: TypeAlias = ProviderType
    EmbeddingRoleType: TypeAlias = EmbeddingRole

    # Circular imports for static type hints
    from omop_graph.graph.kg import KnowledgeGraph
    from omop_graph.graph.paths import StandardConcept

else:
    EmbeddingBackendType = str
    EmbeddingMetricType = str
    EmbeddingIndexType = str
    EmbeddingProviderType = str
    EmbeddingRoleType = str

SUPPORTED_BACKENDS: Tuple[str, ...] = ()
SUPPORTED_METRICS: Tuple[str, ...] = ()
if HAS_OMOP_EMB:
    try:
        from omop_emb import BackendType, MetricType
        from omop_emb.embeddings import EmbeddingRole
        from omop_emb import EmbeddingReaderInterface, EmbeddingWriterInterface
        # Extract the string values from the StrEnums
        SUPPORTED_BACKENDS = tuple(v.value for v in BackendType)
        SUPPORTED_METRICS = tuple(v.value for v in MetricType)
    except ModuleNotFoundError as exc:
        # Only swallow missing optional dependency imports.
        if exc.name and exc.name.startswith("omop_emb"):
            pass
        else:
            raise

logger = logging.getLogger(__name__)


class MissingExtensionError(ImportError):
    """Raised when an optional omop extension is required but not installed."""
    def __init__(self, feature: str = "Embedding functionality"):
        super().__init__(
            f"{feature} requires the 'omop-emb' package. "
            "Install it via: pip install omop-graph[emb]"
        )

def _get_embedding_interface(kg: KnowledgeGraph) -> Optional[EmbeddingReaderInterface | EmbeddingWriterInterface]:
    """
    Internal utility to retrieve the embedding interface from the KG, without distinguishing between reader/writer.
    This is only used for internal logic where we don't need to differentiate between reader and writer capabilities.
    For external use, prefer get_embedding_reader_interface and get_embedding_writer_interface for clearer intent and error handling.

    Returns None if the extension is not available or if any errors occur during retrieval. The following errors can be raised by the kg.emb property:
    - MissingExtensionError: if the omop_emb package is not installed.
    - ValueError: if the package is installed but the KG was not initialized with an embedding configuration.

    Both errors indicate that the embedding interface is not available, so we catch them and return None to simplify handling for callers.
    """
    try:
        return kg.emb
    except ValueError:
        logger.debug("Embedding interface not available: no EmbeddingConfiguration provided.")
        return None
    except MissingExtensionError as exc:
        logger.warning(f"Embedding interface not available: {exc}")
        return None
    
def get_embedding_reader_interface(kg: KnowledgeGraph) -> Optional["EmbeddingReaderInterface"]:
    """
    Utility to safely retrieve the embedding reader interface from the KG.
    Returns None if the extension is not available or if the interface is not a reader.
    """
    interface = _get_embedding_interface(kg)
    if interface is not None and HAS_OMOP_EMB and not isinstance(interface, EmbeddingReaderInterface):
        raise TypeError(f"Expected embedding interface to be a reader, but got {type(interface)}.")
    return interface


def get_embedding_writer_interface(kg: KnowledgeGraph) -> Optional["EmbeddingWriterInterface"]:
    """
    Utility to safely retrieve the embedding writer interface from the KG.
    Returns None if the extension is not available or if the interface is not a writer.
    """
    interface = _get_embedding_interface(kg)
    if interface is not None and HAS_OMOP_EMB and not isinstance(interface, EmbeddingWriterInterface):
        raise TypeError(f"Expected embedding interface to be a writer, but got {type(interface)}. Instantiate the KG with an embedding client to get a writer interface.")
    return interface  # type: ignore[return-value]

def semantic_similarity(
    kg: KnowledgeGraph,
    standard_concepts: Sequence[StandardConcept],
    query_embedding: np.ndarray,
) -> Optional[Tuple[Tuple[NearestConceptMatch, ...], ...]]:
    """
    Calculates similarity between a query embedding and stored concept embeddings.

    Parameters
    ----------
    kg : KnowledgeGraph
        The knowledge graph instance, used to access the embedding interface.
    standard_concepts : Sequence[StandardConcept]
        A sequence of standard concepts to score against the query embedding.
    query_embedding : np.ndarray
        The query vector to compare against concept embeddings. Expected shape is (1, D).

    Returns
    -------
    Optional[Tuple[Tuple[NearestConceptMatch, ...], ...]]
        A tuple of tuple of NearestConceptMatch objects containing similarity scores for each concept. The tuples are of shape (q, k) where q is the number of query vectors (usually 1 for a single text embedding) and k is the number of nearest neighbors returned by the embedding interface.
    """
    if not HAS_OMOP_EMB:
        logger.info("Embedding functionality is not available. Ensure 'omop-emb' is installed to use this feature.")
        return None

    embedding_reader = get_embedding_reader_interface(kg)
    if embedding_reader is None:
        logger.info("Embedding reader interface not found in KG. Skipping similarity calculation.")
        return None
    
    from omop_emb.utils.embedding_utils import EmbeddingConceptFilter
    
    concept_ids = tuple(dict.fromkeys(sc.concept_id for sc in standard_concepts))
    concept_filter = EmbeddingConceptFilter(concept_ids=concept_ids, limit=len(concept_ids))

    missing_sc_embeddings = embedding_reader.get_concepts_without_embedding(
        omop_cdm_engine=kg.cdm_engine,
        concept_filter=concept_filter,
    )

    if missing_sc_embeddings:
        if kg.compute_missing_embeddings:
            logger.debug(f"Concepts missing embeddings: {missing_sc_embeddings}. Computing missing embeddings on-the-fly.")
            embedding_writer = get_embedding_writer_interface(kg)
            if embedding_writer is not None:

                missing_concept_ids = tuple(missing_sc_embeddings.keys())
                missing_concept_texts = tuple(missing_sc_embeddings.values())

                embedding_writer.embed_and_upsert_concepts(
                    omop_cdm_engine=kg.cdm_engine,
                    concept_ids=missing_concept_ids,
                    concept_texts=missing_concept_texts,
                )
                logger.debug(f"Computed and stored embeddings for missing concepts: {missing_concept_ids}")
            else:
                logger.info(
                    f"Cannot compute missing embeddings due to missing embedding_writer.\n"
                    "Ensure the KG was initialised with a write-capable client to enable on-the-fly embedding computation.\n"
                    f"Expect missing embedding scores for concepts: {missing_sc_embeddings}"
                )
        else:
            logger.info(
                f"Concepts missing embeddings: {missing_sc_embeddings}.\n"
                "compute_missing_embeddings is disabled; these concepts will be skipped in similarity scoring.\n"
                "Expect missing embedding scores for these concepts in the results."
            )

    nearest_concept_matches = get_neareast_concepts(
        kg=kg,
        query_embedding=query_embedding,
        concept_filter=concept_filter,
    )
        
    return nearest_concept_matches

def get_neareast_concepts(
    kg: KnowledgeGraph,
    query_embedding: np.ndarray,
    concept_filter: Optional[EmbeddingConceptFilter],
) -> Optional[Tuple[Tuple[NearestConceptMatch, ...], ...]]:
    """
    RAG retrieval for concept similarity scores. The query_embedding is compared against
    stored embeddings using the metric and model already configured on the KG's embedding
    reader interface.

    Parameters
    ----------
    kg : KnowledgeGraph
        The knowledge graph instance, used to access the embedding interface.
    query_embedding : np.ndarray
        The query vector to search with. Expected shape is (q, D).
    concept_filter : Optional[EmbeddingConceptFilter]
        Pre-filter applied during KNN (concept IDs, domain, vocabulary, standard).
        Also caps k to ``concept_filter.limit`` when set.

    Returns
    -------
    Tuple[Tuple[NearestConceptMatch, ...], ...], optional
        Shape ``(q, ≤k)``. Returns None when the interface is unavailable or
        no embedding is provided.
    """
    if not HAS_OMOP_EMB:
        return None

    embedding_reader = get_embedding_reader_interface(kg)
    if not embedding_reader:
        logger.info("Embedding interface not available in KG.")
        return None

    if not embedding_reader.is_model_registered():
        logger.info("Model '%s' not registered.", embedding_reader.canonical_model_name)
        return None

    nearest_concepts = embedding_reader.get_nearest_concepts(
        query_embedding=query_embedding,
        concept_filter=concept_filter,
    )
    if not nearest_concepts:
        logger.info("No nearest concepts found for the given query embedding and filter.")
        return None
    return nearest_concepts