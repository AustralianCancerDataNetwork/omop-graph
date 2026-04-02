# Utils for the optional omop-emb package
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence, Mapping, Literal

import logging
import numpy as np
from sqlalchemy.orm import Session

from omop_graph.graph.constraints import SearchConstraintConcept

if TYPE_CHECKING:
    # Circular import guard for type hints
    from omop_graph.graph.kg import KnowledgeGraph
    from omop_graph.graph.paths import StandardConcept
    # Optional Import
    from omop_emb import MetricType, EmbeddingInterface
    from omop_llm import LLMClient
    

logger = logging.getLogger(__name__)

EmbeddingBackendName = Literal["pgvector", "faiss"]
EmbeddingMetricType = Literal["cosine", "l2"]

class MissingExtensionError(ImportError):
    """Raised when an optional omop extension is required but not installed."""
    pass


def get_embedding_interface(kg: KnowledgeGraph) -> Optional[EmbeddingInterface]:
    """Utility function to get the embedding interface from the KG, if the omop-emb extension is installed. Returns None if the extension is not available."""
    try:
        return kg.emb
    except MissingExtensionError:
        return None
    

def semantic_similarity(
    kg: KnowledgeGraph,
    unique_standard_concepts: Sequence[StandardConcept],
    text_embedding: Optional[np.ndarray],
    text_embedding_model: Optional[str],
    embedding_client: Optional[LLMClient],
    metric_type: Optional[EmbeddingMetricType],
) -> Optional[np.ndarray]:
    """ Retrieve semantic similarity scores for the unique standard concepts using the provided text embedding and model.
    The semantic similarity scores are calculated as the similarity between the text embedding and the embeddings of the unique standard concepts. 
    The function first attempts to retrieve similarity scores from the KG using RAG retrieval. 
    If retrieval from the KG is not possible (e.g., due to missing parameters or no scores found),
    it falls back to using the embedding client to compute similarity scores by embedding the unique standard concepts and calculating similarity with the text embedding.
    
    Parameters
    ----------
    kg : KnowledgeGraph
        The Knowledge Graph instance.
    unique_standard_concepts : Sequence[StandardConcept]
        The unique standard concepts identified for the candidate.
    text_embedding : np.ndarray, optional
        The embedding vector for the input text. Expected shape is (1, dimension) as we only have one text input (query).
        Used for RAG retrieval of similarity scores from the database.
    text_embedding_model : str, optional
        The name of the embedding model to use for RAG retrieval and for storing new embeddings if fallback to embedding client is needed.
    embedding_client : LLMClient, optional
        The client to use for embedding the unique standard concepts and calculating similarity scores if retrieval from the KG is not possible. 
    metric_type : EmbeddingMetricType, optional
        The type of similarity metric to use.
    """

    embedding_interface = get_embedding_interface(kg)
    if embedding_interface is None:
        return None
    
    concept_filter = SearchConstraintConcept(concept_ids=tuple(sc.concept_id for sc in unique_standard_concepts))

    similarity_scores = None
    with kg.session_factory() as session:
        similarity_scores_dict = get_neareast_concepts(
            session=session,
            kg=kg,
            text_embedding_model=text_embedding_model,
            text_embedding=text_embedding,
            concept_filter=concept_filter,
            metric_type=metric_type
        )

        if not similarity_scores_dict:
            if (
                text_embedding_model is not None and
                embedding_client is not None and 
                text_embedding is not None
            ):
                logger.debug("Falling back to embedding client for similarity scores.")

                assert isinstance(text_embedding, np.ndarray), "Text embedding must be a numpy array for fallback similarity scoring."
                assert text_embedding.shape[0] == 1 and text_embedding.ndim == 2, "Text embedding must be a 2D vector with first dim = 1."            

                embedding_interface = get_embedding_interface(kg)
                if embedding_interface is None:
                    return None
                
                missing_sc_embeddings = embedding_interface.get_concepts_without_embedding(
                    session=session,
                    concept_filter=concept_filter, # type: ignore (is the same and needs to be moved to OMOP_Alchemy eventually)
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
                    model=text_embedding_model
                )

                similarity_scores_dict = get_neareast_concepts(
                    session=session,
                    kg=kg,
                    text_embedding_model=text_embedding_model,
                    text_embedding=text_embedding,
                    concept_filter=concept_filter,
                    metric_type=metric_type
                )

                if not similarity_scores_dict:
                    logger.warning("Failed to retrieve similarity scores even after fallback embedding client computation.")
        else:
            similarity_scores = np.array(list(similarity_scores_dict.values()))

        return similarity_scores


def get_neareast_concepts(
    session: Session,
    kg: KnowledgeGraph,
    text_embedding_model: Optional[str],
    text_embedding: Optional[np.ndarray],
    concept_filter: Optional[SearchConstraintConcept],
    metric_type: Optional[EmbeddingMetricType],
    k: int = 10
) -> Optional[Mapping[int, float]]:
    """Tries to retrieve similarity scores for the unique standard concepts from the KG using RAG retrieval based on the provided text embedding and model. 
    
    Parameters
    ----------
    session : Session
        The database session to use for retrieval.
    kg : KnowledgeGraph
        The Knowledge Graph instance.
    text_embedding_model : Optional[str]
        The name of the embedding model to use for retrieval. Must be provided to attempt retrieval.
    text_embedding : Optional[np.ndarray]
        The embedding vector for the input text to use for retrieval. Must be provided to attempt retrieval
    concept_filter : Optional[SearchConstraintConcept]
        A filter specifying which concepts to consider for similarity scoring.
    metric_type : Optional[EmbeddingMetricType]
        The type of similarity metric to use for retrieval. Must be provided to attempt retrieval.
    
    Returns
    -------
    Optional[Mapping[int, float]]
        A dictionary mapping concept_ids to their similarity score with the input text
        retrieved from the KG. Returns None if retrieval was not attempted due to missing parameters or if no similarity scores could be retrieved.
    """
    embedding_interface = get_embedding_interface(kg)
    if embedding_interface is None:
        return None
    
    if text_embedding_model is None:
        logger.info("No text embedding model provided, skipping embedding-based similarity scoring.")
        return None
    if not embedding_interface.is_model_registered(session=session, model_name=text_embedding_model):
        logger.info(f"Text embedding model '{text_embedding_model}' is not registered in the KG, skipping embedding-based similarity scoring.")
        return None
    if text_embedding is None:
        logger.info("No text embedding provided, skipping embedding-based similarity scoring.")
        return None
    if metric_type is None:
        logger.info("No metric type provided, skipping embedding-based similarity scoring.")
        return None

    assert isinstance(text_embedding, np.ndarray), "Text embedding must be a numpy array for RAG retrieval."
    assert text_embedding.shape[0] == 1 and text_embedding.ndim == 2, "Text embedding must be a 2D vector with first dim = 1, i.e. having a query dimension of 1 for RAG retrieval."

    similarity_scores_tuple = embedding_interface.get_nearest_concepts(
        session=session,
        model_name=text_embedding_model,
        query_embedding=text_embedding.tolist()[0],
        concept_filter=concept_filter,  # type: ignore (is the same and needs to be moved to OMOP_Alchemy eventually)
        metric_type=metric_type,
        k=k
    )

    assert len(similarity_scores_tuple) == 1, "Expected a single set of similarity scores for the query embedding given the text embedding shape was (1, embedding_dim)."
    return similarity_scores_tuple[0] 