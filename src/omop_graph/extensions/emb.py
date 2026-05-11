from __future__ import annotations

import logging
import importlib.util
from typing import TYPE_CHECKING, Optional, Sequence, Mapping, TypeAlias, Tuple
import numpy as np
from sqlalchemy.orm import Session
from omop_graph.graph.constraints import SearchConstraintConcept

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
_PARSE_INDEX_TYPE = None
_PARSE_METRIC_TYPE = None

if HAS_OMOP_EMB:
    try:
        from omop_emb import BackendType, MetricType
        from omop_emb.embeddings import EmbeddingRole
        from omop_emb.config import parse_index_type, parse_metric_type
        from omop_emb import EmbeddingReaderInterface, EmbeddingWriterInterface
        # Extract the string values from the StrEnums
        SUPPORTED_BACKENDS = tuple(v.value for v in BackendType)
        SUPPORTED_METRICS = tuple(v.value for v in MetricType)
        _PARSE_INDEX_TYPE = parse_index_type
        _PARSE_METRIC_TYPE = parse_metric_type
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
    except (MissingExtensionError, ValueError) as exc:
        logger.error(f"Embedding interface not available: {exc}")
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
    text_embedding: Optional[np.ndarray],
    text_embedding_model: Optional[str],
    metric_type: Optional[EmbeddingMetricType],
    index_type: Optional[EmbeddingIndexType],
) -> Optional[Tuple[Tuple[NearestConceptMatch, ...], ...]]:
    """
    Calculates similarity between text embeddings and concept embeddings.

    Parameters
    ----------
    kg : KnowledgeGraph
        The knowledge graph instance, used to access the embedding interface.
    standard_concepts : Sequence[StandardConcept]
        A sequence of standard concepts for which to calculate similarity scores against using the text_embedding.
    text_embedding : Optional[np.ndarray]
        The embedding vector to compare against concept embeddings. Expected shape is (q, dimension) where q is the number of query vectors and dimension is the size of the embedding space for the model. Note: q=1 for a single text embedding.
    text_embedding_model : Optional[str]
        The name of the text embedding model used to generate the text_embedding. This should correspond to
        a model registered in the embedding interface. If None, similarity calculation will not be attempted.
    metric_type : Optional[EmbeddingMetricType]
        The similarity or distance metric to use for calculating similarity scores. This must be compatible with the index type used by the database. If None, similarity calculation will not be attempted.
    index_type : Optional[EmbeddingIndexType]
        The type of vector index used to store the embeddings. This is required to ensure that the correct retrieval method is used from the embedding interface. If None, similarity calculation will not be attempted.

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
    
    if index_type is None:
        logger.info("Index type is required for similarity calculation but not provided. Skipping similarity calculation.")
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
            if (
                embedding_writer is not None and
                text_embedding_model is not None and
                text_embedding is not None
            ):

                missing_concept_ids = tuple(missing_sc_embeddings.keys())
                missing_concept_texts = tuple(missing_sc_embeddings.values())

                embedding_writer.embed_and_upsert_concepts(
                    omop_cdm_engine=kg.cdm_engine,
                    concept_ids=missing_concept_ids,
                    concept_texts=missing_concept_texts,
                )
                logger.debug(f"Computed and stored embeddings for missing concepts: {missing_concept_ids}")
            else:
                param_dict = {
                    "text_embedding_model": text_embedding_model,
                    "embedding_writer": embedding_writer,
                    "text_embedding": text_embedding,
                    "index_type": index_type
                }
                none_params = [k for k, v in param_dict.items() if v is None]
                logger.info(
                    f"Cannot compute missing embeddings due to missing parameters: {none_params}\n"
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
        text_embedding_model=text_embedding_model,
        text_embedding=text_embedding,
        concept_filter=concept_filter,
        metric_type=metric_type,
        index_type=index_type,
    )
        
    return nearest_concept_matches

def get_neareast_concepts(
    kg: KnowledgeGraph,
    text_embedding_model: Optional[str],
    text_embedding: Optional[np.ndarray],
    concept_filter: Optional[EmbeddingConceptFilter],
    metric_type: Optional[EmbeddingMetricType],
    index_type: Optional[EmbeddingIndexType],
) -> Optional[Tuple[Tuple[NearestConceptMatch, ...], ...]]:
    """
    RAG retrieval for concept similarity scores. The text_embedding is used to retrieve the nearest concepts from the database
    using stored embeddings and the specified similarity metric.

    Parameters
    ----------
    kg : KnowledgeGraph
        The knowledge graph instance, used to access the embedding interface.
    text_embedding_model : Optional[str]
        The name of the text embedding model to use for retrieval. This should correspond to a model registered in the embedding interface. If None, retrieval will not be attempted.
    text_embedding : Optional[np.ndarray]
        The embedding vector to search with. Expected shape is (q, dimension) where q is the number of query vectors and dimension is the size of the embedding space for the model. If None, retrieval will not be attempted.
    concept_filter : Optional[EmbeddingConceptFilter], optional
        A filter to specify which concepts to consider as potential nearest neighbors. Also limits the number of neighbors returned (K). If None, internal defaults are used to limit the number of neighbours.
    index_type : IndexType
        The type of vector index used to store the embeddings.
    metric_type : MetricType
        The similarity or distance metric to use for nearest neighbor search. This must be compatible with the index type used by the database.

    Returns
    -------
    Tuple[Mapping[int, float], ...], optional
        A tuple of dictionaries containing nearest concept matches for each query vector. The outer tuple is of length q (number of query vectors), and each inner dictionary maps concept IDs to their similarity scores with the query embedding (having k entries corresponding to the k nearest neighbors).
        If retrieval fails or if any required parameters are missing, returns None.
    """
    if not HAS_OMOP_EMB:
        return None
    
    embedding_reader = get_embedding_reader_interface(kg)
    if not embedding_reader:
        logger.info("Embedding interface not available in KG.")
        return None
    
    if not text_embedding_model:
        logger.info("No text embedding model specified.")
        return None

    if not index_type:
        logger.info("No index type specified for retrieval.")
        return None

    if metric_type is None:
        logger.info("No metric type specified for retrieval.")
        return None

    if _PARSE_INDEX_TYPE is None or _PARSE_METRIC_TYPE is None:
        logger.info("Embedding type parsers are unavailable; cannot validate metric/index inputs.")
        return None

    try:
        resolved_index_type = _PARSE_INDEX_TYPE(index_type)
        resolved_metric_type = _PARSE_METRIC_TYPE(metric_type)
    except ValueError as exc:
        logger.info(f"Invalid embedding retrieval parameters: {exc}")
        return None

    if not embedding_reader.is_model_registered():
        logger.info(f"Model '{text_embedding_model}' not registered.")
        return None

    if text_embedding is None:
        return None

    nearest_concepts = embedding_reader.get_nearest_concepts(
        query_embedding=text_embedding,
        concept_filter=concept_filter,
    )
    if not nearest_concepts:
        logger.info("No nearest concepts found for the given query embedding and filter.")
        return None
    return nearest_concepts