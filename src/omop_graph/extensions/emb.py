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
    except ImportError:
        pass

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
    unique_standard_concepts: Sequence[StandardConcept],
    text_embedding: Optional[np.ndarray],
    text_embedding_model: Optional[str],
    embedding_client: Optional[LLMClient],
    metric_type: Optional[EmbeddingMetricType],
    index_type: Optional[EmbeddingIndexType],
) -> Optional[np.ndarray]:
    """
    Calculates similarity between text embeddings and concept embeddings.
    """
    if not HAS_OMOP_EMB:
        return None

    embedding_interface = get_embedding_interface(kg)
    if embedding_interface is None:
        return None
    
    concept_filter = SearchConstraintConcept(
        concept_ids=tuple(sc.concept_id for sc in unique_standard_concepts)
    )

    with kg.session_factory() as session:
        similarity_scores_dict = get_neareast_concepts(
            session=session,
            kg=kg,
            text_embedding_model=text_embedding_model,
            text_embedding=text_embedding,
            concept_filter=concept_filter,
            metric_type=metric_type,
            index_type=index_type
        )

        if not similarity_scores_dict:
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
                    concept_filter=concept_filter, # type: ignore
                    model_name=text_embedding_model
                )

                standard_concept_embeddings = embedding_interface.embed_texts(
                    texts=tuple(missing_sc_embeddings.values()),
                    embedding_client=embedding_client,
                )
                
                embedding_interface.add_to_db(
                    embeddings=standard_concept_embeddings,
                    concept_ids=tuple([sc.concept_id for sc in unique_standard_concepts]),
                    session=session,
                    model=model_name,
                    index_type=resolved_index_type,
                )

                # Re-attempt retrieval after update
                similarity_scores_dict = get_neareast_concepts(
                    session=session,
                    kg=kg,
                    text_embedding_model=text_embedding_model,
                    text_embedding=text_embedding,
                    concept_filter=concept_filter,
                    metric_type=metric_type,
                    index_type=index_type
                )

        if similarity_scores_dict:
            return np.array(list(similarity_scores_dict.values()))
        
        return None

def get_neareast_concepts(
    session: Session,
    kg: KnowledgeGraph,
    text_embedding_model: Optional[str],
    text_embedding: Optional[np.ndarray],
    concept_filter: Optional[SearchConstraintConcept],
    metric_type: Optional[EmbeddingMetricType],
    index_type: Optional[EmbeddingIndexType],
    k: int = 10
) -> Optional[Mapping[int, float]]:
    """
    RAG retrieval for concept similarity scores.
    Ensures all types from omop_emb are used via strings or local checks.
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

    return similarity_scores_tuple[0] if similarity_scores_tuple else None