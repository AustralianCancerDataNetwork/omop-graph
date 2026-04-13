from __future__ import annotations

import logging
import importlib.util
from typing import TYPE_CHECKING, Optional, Sequence, Mapping, TypeAlias, Tuple
import numpy as np
from sqlalchemy.orm import Session
from omop_graph.graph.constraints import SearchConstraintConcept

HAS_OMOP_EMB = importlib.util.find_spec("omop_emb") is not None

if TYPE_CHECKING:
    from omop_emb import BackendType, MetricType, IndexType
    EmbeddingBackendType: TypeAlias = BackendType
    EmbeddingMetricType: TypeAlias = MetricType
    EmbeddingIndexType: TypeAlias = IndexType
    from omop_graph.graph.kg import KnowledgeGraph
    from omop_graph.graph.paths import StandardConcept
    from omop_emb import EmbeddingInterface
    from omop_llm import LLMClient
else:
    EmbeddingBackendType = str
    EmbeddingMetricType = str
    EmbeddingIndexType = str

SUPPORTED_BACKENDS: Tuple[str, ...] = ()
SUPPORTED_METRICS: Tuple[str, ...] = ()
_PARSE_INDEX_TYPE = None
_PARSE_METRIC_TYPE = None

if HAS_OMOP_EMB:
    try:
        from omop_emb import BackendType, MetricType
        from omop_emb.config import parse_index_type, parse_metric_type
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

def get_embedding_interface(kg: KnowledgeGraph) -> Optional["EmbeddingInterface"]:
    """
    Utility to safely retrieve the embedding interface from the KG. 
    Returns None if the extension is not available.
    """
    try:
        return kg.emb
    except (MissingExtensionError, ImportError):
        return None

def semantic_similarity(
    kg: KnowledgeGraph,
    standard_concepts: Sequence[StandardConcept],
    text_embedding: Optional[np.ndarray],
    text_embedding_model: Optional[str],
    embedding_client: Optional[LLMClient],
    metric_type: Optional[EmbeddingMetricType],
    index_type: Optional[EmbeddingIndexType],
) -> Optional[Tuple[Mapping[int, float], ...]]:
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
    embedding_client : Optional[LLMClient]
        An optional LLM client used to fetch missing embeddings if they are not present in the database. This is only used as a fallback mechanism if the initial retrieval of similarity scores fails due to missing embeddings. If None, no fallback will be attempted and the function will return None if embeddings are missing.
    metric_type : Optional[EmbeddingMetricType]
        The similarity or distance metric to use for calculating similarity scores. This must be compatible with the index type used by the database. If None, similarity calculation will not be attempted.
    index_type : Optional[EmbeddingIndexType]
        The type of vector index used to store the embeddings. This is required to ensure that the correct retrieval method is used from the embedding interface. If None, similarity calculation will not be attempted.

    Returns
    -------
    Optional[Tuple[Mapping[int, float], ...]]
        A tuple of dictionaries mapping concept IDs to similarity scores for each query embedding.
        The outer tuple is of length q (number of query embeddings, shape[0] of text_embedding), and each inner dictionary contains up to k (the number of unique concepts) entries mapping concept IDs to their similarity scores with the query embedding.

    """
    if not HAS_OMOP_EMB:
        logger.info("Embedding functionality is not available. Ensure 'omop-emb' is installed to use this feature.")
        return None

    embedding_interface = get_embedding_interface(kg)
    if embedding_interface is None:
        logger.info("Embedding interface not found in KG. Ensure the embedding extension is properly configured.")
        return None
    
    concept_ids = tuple(dict.fromkeys(sc.concept_id for sc in standard_concepts))
    concept_filter = SearchConstraintConcept(concept_ids=concept_ids)

    with kg.session_factory() as session:
        similarity_scores_tuple_of_dicts = get_neareast_concepts(
            session=session,
            kg=kg,
            text_embedding_model=text_embedding_model,
            text_embedding=text_embedding,
            concept_filter=concept_filter,
            metric_type=metric_type,
            index_type=index_type,
            k=len(concept_ids)
        )

        if not similarity_scores_tuple_of_dicts:
            # Fallback logic if database retrieval fails
            if all(v is not None for v in [text_embedding_model, embedding_client, text_embedding, index_type]):
                logger.debug("Falling back to embedding client for similarity scores.")

                # Runtime narrowing for static and runtime safety.
                model_name = text_embedding_model
                resolved_index_type = index_type
                if model_name is None or resolved_index_type is None:
                    return None

                # Validate types at runtime since static checks won't catch this without the lib
                if not isinstance(text_embedding, np.ndarray):
                    raise TypeError("text_embedding must be a numpy array.")
                
                # Fetch missing embeddings and update DB
                missing_sc_embeddings = embedding_interface.get_concepts_without_embedding(
                    session=session,
                    concept_filter=concept_filter,  # type: ignore
                    model_name=model_name,
                    index_type=resolved_index_type,
                )

                if missing_sc_embeddings:
                    missing_concept_ids = tuple(missing_sc_embeddings.keys())
                    standard_concept_embeddings = embedding_interface.embed_texts(
                        texts=tuple(missing_sc_embeddings.values()),
                        embedding_client=embedding_client,
                    )

                    embedding_interface.add_to_db(
                        embeddings=standard_concept_embeddings,
                        concept_ids=missing_concept_ids,
                        session=session,
                        model=model_name,
                        index_type=resolved_index_type,
                    )

                # Re-attempt retrieval after update
                similarity_scores_tuple_of_dicts = get_neareast_concepts(
                    session=session,
                    kg=kg,
                    text_embedding_model=text_embedding_model,
                    text_embedding=text_embedding,
                    concept_filter=concept_filter,
                    metric_type=metric_type,
                    index_type=index_type
                )
            else:
                param_dict = {
                    "text_embedding_model": text_embedding_model,
                    "embedding_client": embedding_client,
                    "text_embedding": text_embedding,
                    "index_type": index_type
                }
                none_params = [k for k, v in param_dict.items() if v is None]
                logger.info(f"Fallback embedding calculation not possible for standard_concepts due to missing parameters: {none_params}")

        return similarity_scores_tuple_of_dicts

def get_neareast_concepts(
    session: Session,
    kg: KnowledgeGraph,
    text_embedding_model: Optional[str],
    text_embedding: Optional[np.ndarray],
    concept_filter: Optional[SearchConstraintConcept],
    metric_type: Optional[EmbeddingMetricType],
    index_type: Optional[EmbeddingIndexType],
    k: int = 10
) -> Optional[Tuple[Mapping[int, float], ...]]:
    """
    RAG retrieval for concept similarity scores. The text_embedding is used to retrieve the nearest concepts from the database
    using stored embeddings and the specified similarity metric.

    Parameters
    ----------
    session : Session
        SQLAlchemy session for any required relational access.
    kg : KnowledgeGraph
        The knowledge graph instance, used to access the embedding interface.
    text_embedding_model : Optional[str]
        The name of the text embedding model to use for retrieval. This should correspond to a model registered in the embedding interface. If None, retrieval will not be attempted.
    text_embedding : Optional[np.ndarray]
        The embedding vector to search with. Expected shape is (q, dimension) where q is the number of query vectors and dimension is the size of the embedding space for the model. If None, retrieval will not be attempted.
    concept_filter : Optional[EmbeddingConceptFilter], optional
        A filter to specify which concepts to consider as potential nearest neighbors.
    index_type : IndexType
        The type of vector index used to store the embeddings.
    metric_type : MetricType
        The similarity or distance metric to use for nearest neighbor search. This must be compatible with the index type used by the database.
    k : int, optional
        K nearest neighbors to return for each query vector. Default is 10.

    Returns
    -------
    Tuple[Mapping[int, float], ...], optional
        A tuple of dictionaries containing nearest concept matches for each query vector. The outer tuple is of length q (number of query vectors), and each inner dictionary maps concept IDs to their similarity scores with the query embedding (having k entries corresponding to the k nearest neighbors).
        If retrieval fails or if any required parameters are missing, returns None.
    """
    if not HAS_OMOP_EMB:
        return None
    
    embedding_interface = get_embedding_interface(kg)
    if not embedding_interface:
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

    if not embedding_interface.is_model_registered(model_name=text_embedding_model, index_type=resolved_index_type):
        logger.info(f"Model '{text_embedding_model}' not registered.")
        return None

    if text_embedding is None:
        return None

    similarity_scores_tuple = embedding_interface.get_nearest_concepts(
        session=session,
        model_name=text_embedding_model,
        index_type=resolved_index_type,
        query_embedding=text_embedding,
        concept_filter=concept_filter,  # type: ignore
        metric_type=resolved_metric_type,
        k=k
    )

    if similarity_scores_tuple is None:
        logger.info("No similarity scores retrieved from embedding interface.")
        return None
    
    assert len(similarity_scores_tuple) == text_embedding.shape[0], (
        f"Expected similarity scores for {text_embedding.shape[0]} query embeddings, "
        f"but got {len(similarity_scores_tuple)}."
    )
    assert all(isinstance(d, dict) for d in similarity_scores_tuple), (
        "Expected each item in similarity_scores_tuple to be a dictionary mapping concept IDs to scores."
    )
    assert all(len(d) <= k for d in similarity_scores_tuple), (
        f"Expected at most {k} nearest neighbors per query embedding, but found a dictionary with {max(len(d) for d in similarity_scores_tuple)} entries."
    )
    return similarity_scores_tuple