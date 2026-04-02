"""
Semantic Grounding Orchestration.

This module provides the high-level `ground_term` function, which orchestrates 
the full grounding pipeline:
1.  **Candidate Resolution**: Finding raw concepts that match the input text.
2.  **Hierarchy Validation**: Ensuring candidates have a valid relationship path 
    to required parent concepts.
3.  **Standardization**: Mapping non-standard candidates to standard OMOP concepts.
4.  **Semantic Ranking**: Using embeddings and graph-based scoring to select the 
    best mapping.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple, Sequence, Mapping

import numpy as np
from sqlalchemy.orm import Session

if TYPE_CHECKING:
    from omop_emb import EmbeddingConceptFilter, MetricType, EmbeddingInterface

from omop_graph.extensions.omop_alchemy import ClassIDEnum
from omop_graph.graph.constraints import SearchConstraintConcept
from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.graph.paths import (
    StandardConcept,
    find_standard_paths,
    get_unique_standard_concepts,
)
from omop_graph.graph.scoring import StandardConceptWithScore, score_standard_concepts
from omop_graph.reasoning.resolvers import (
    CandidateHit,
    ResolverPipeline,
)
from omop_graph.extensions.emb import MissingExtensionError
from omop_llm import LLMClient

logger = logging.getLogger(__name__)



@dataclass(frozen=True)
class GroundingConstraints:
    """
    Configuration for restricting the grounding search space.

    Parameters
    ----------
    parent_ids : tuple[int, ...], optional
        OMOP Concept IDs that act as required ancestors for any valid result.
    search_constraint : SearchConstraintConcept, optional
        Domain and Vocabulary restrictions for the initial resolution phase.
    max_depth : int, optional
        Maximum allowed distance in the hierarchy between a candidate and a parent.
    predicate_kinds : frozenset[ClassIDEnum], optional
        The types of relationships allowed during pathfinding.
    """

    parent_ids: Optional[Tuple[int, ...]]
    search_constraint: Optional[SearchConstraintConcept]
    max_depth: int = 6
    predicate_kinds: frozenset[ClassIDEnum] = frozenset({ClassIDEnum.IDENTITY,})


def ground_term(
    resolver_pipeline: ResolverPipeline,
    kg: KnowledgeGraph,
    text: str,
    text_embedding: Optional[np.ndarray],
    text_embedding_model: Optional[str],
    embedding_client: Optional["LLMClient"],
    constraints: GroundingConstraints,
    max_candidates: Optional[int] = None,
    metric_type: Optional[MetricType] = None,
) -> List[StandardConceptWithScore]:
    """
    Ground a text string to a ranked list of standard OMOP concepts.

    Parameters
    ----------
    resolver_pipeline : ResolverPipeline
        The pipeline of search strategies to find initial candidates.
    kg : KnowledgeGraph
        The OMOP Knowledge Graph instance.
    text : str
        The input text to ground.
    text_embedding : np.ndarray
        The embedding vector for the input text.
    text_embedding_model : str, optional
        The name of the embedding model used to generate `text_embedding`. Used for RAG retrieval from the database.
    embedding_client : LLMClient, optional
        DEBUG: A client to obtain embeddings and similarity scores if not available in the KG. Not for production use.
    constraints : GroundingConstraints
        Contextual constraints (parents, domains, etc.) to apply.
    max_candidates : int, optional
        Limit for the number of candidates returned. If None, returns all candidates.
    metric_type : MetricType, optional
        The similarity or distance metric to use for optional embedding-based scoring. This should be compatible with the index type used by the database for RAG retrieval. Must be provided if `text_embedding` and `text_embedding_model` are provided for RAG retrieval.

    Returns
    -------
    list[StandardConceptWithScore]
        A list of standard concepts sorted by their total score (descending).

    Raises
    ------
    NotImplementedError
        If no `parent_ids` are provided in constraints.
    """
    standard_concepts: List[StandardConcept] = []

    search_constraints = constraints.search_constraint
    if search_constraints is not None:
        search_constraints.check(kg)

    resolved = list(resolver_pipeline.resolve(kg, text, constraints=search_constraints))

    # Anchoring
    for hit in resolved:
        if constraints.parent_ids is not None:
            candidate_standard_concepts = find_standard_concepts(
                kg=kg,
                candidate=hit,
                parent_ids=constraints.parent_ids,
                max_depth=constraints.max_depth,
                max_paths=None,
                predicate_kinds=constraints.predicate_kinds,
            )

            if not candidate_standard_concepts:
                concept_name = kg.concept_view(hit.concept_id).concept_name
                logger.debug(
                    f"Failed hierarchy constraint: {hit.concept_id} ({concept_name}) "
                    f"has no path to parents {constraints.parent_ids}"
                )
                continue
            
            standard_concepts.extend(candidate_standard_concepts)
        else:
            # Note: We currently require parent_ids for clinical safety/context
            raise NotImplementedError("Grounding without parent_ids is not supported.")

    unique_standard_concepts = get_unique_standard_concepts(standard_concepts)
    if not unique_standard_concepts:
        logger.info(f"No standard concepts found for '{text}' after hierarchy validation.")
        return []

    similarity_scores = _optional_embedding_scoring(
        kg=kg,
        unique_standard_concepts=unique_standard_concepts,
        text_embedding=text_embedding,
        text_embedding_model=text_embedding_model,
        embedding_client=embedding_client,
        metric_type=metric_type,
    )

    # Scoring
    ranked_standard_concepts = score_standard_concepts(
        text=text, 
        standard_concepts=tuple(unique_standard_concepts),
        kg=kg,
        similarity_scores=similarity_scores
    )

    ranked_standard_concepts.sort(key=lambda sc: sc.total_score, reverse=True)
    return ranked_standard_concepts[:max_candidates] if max_candidates is not None else ranked_standard_concepts


def _optional_embedding_scoring(
    kg: KnowledgeGraph,
    unique_standard_concepts: Sequence[StandardConcept],
    text_embedding: Optional[np.ndarray],
    text_embedding_model: Optional[str],
    embedding_client: Optional["LLMClient"],
    metric_type: Optional[MetricType],
) -> Optional[np.ndarray]:
    """ Retrieve similarity scores for the unique standard concepts using the provided text embedding and model.
    Fallback to using the embedding client to compute similarity scores if retrieval from the KG is not possible. Returns an array of similarity scores corresponding to the unique standard concepts, or None if no similarity scores could be obtained.
    
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
        The name of the embedding model to use.
    embedding_client : LLMClient, optional
        The client for retrieving embeddings.
    metric_type : MetricType, optional
        The type of similarity metric to use.
    """
    
    try:
        embedding_interface = kg.emb
    except MissingExtensionError:
        return None
    
    similarity_scores = None
    with kg.session_factory() as session:
        similarity_scores_dict = _retrieve_embeddings_for_concepts(
            session=session,
            text_embedding_model=text_embedding_model,
            text_embedding=text_embedding,
            unique_standard_concepts=unique_standard_concepts,
            embedding_interface=embedding_interface,
            metric_type=metric_type
        )

        if not similarity_scores_dict:
            if (
                text_embedding_model is not None and
                embedding_client is not None and 
                text_embedding is not None
            ):
                logger.debug("Falling back to embedding client for similarity scores (DEBUG ONLY).")

                sc_names = tuple([sc.concept_name for sc in unique_standard_concepts])
                standard_concept_embeddings = embedding_client.embeddings(sc_names)
                assert isinstance(text_embedding, np.ndarray), "Text embedding must be a numpy array for fallback similarity scoring."
                assert text_embedding.shape[0] == 1 and text_embedding.ndim == 2, "Text embedding must be a 2D vector with first dim = 1."            
                
                embedding_interface.add_to_db(
                    embeddings=standard_concept_embeddings,
                    concept_ids=tuple([sc.concept_id for sc in unique_standard_concepts]),
                    session=session,
                    model=text_embedding_model
                )

                similarity_scores_dict = _retrieve_embeddings_for_concepts(
                    session=session,
                    text_embedding_model=text_embedding_model,
                    text_embedding=text_embedding,
                    unique_standard_concepts=unique_standard_concepts,
                    embedding_interface=embedding_interface,
                    metric_type=metric_type
                )

                if not similarity_scores_dict:
                    logger.warning("Failed to retrieve similarity scores even after fallback embedding client computation.")
        else:
            similarity_scores = np.array(list(similarity_scores_dict.values()))

        return similarity_scores

def _retrieve_embeddings_for_concepts(
    session: Session,
    text_embedding_model: Optional[str],
    text_embedding: Optional[np.ndarray],
    unique_standard_concepts: Sequence[StandardConcept],
    embedding_interface: EmbeddingInterface,
    metric_type: Optional[MetricType]
) -> Optional[Mapping[int, float]]:
    """Tries to retrieve similarity scores for the unique standard concepts from the KG using RAG retrieval based on the provided text embedding and model. 
    
    Parameters
    ----------
    session : Session
        The database session to use for retrieval.
    text_embedding_model : Optional[str]
        The name of the embedding model to use for retrieval. Must be provided to attempt retrieval.
    text_embedding : Optional[np.ndarray]
        The embedding vector for the input text to use for retrieval. Must be provided to attempt retrieval
    unique_standard_concepts : Sequence[StandardConcept]
        The unique standard concepts identified for the candidate, for which we want to retrieve similarity scores.
    embedding_interface : EmbeddingInterface
        The embedding interface to use for retrieval.
    metric_type : Optional[MetricType]
        The type of similarity metric to use for retrieval. Must be provided to attempt retrieval.
    
    Returns
    -------
    Optional[Mapping[int, float]]
        A dictionary mapping concept_ids of the unique standard concepts to their similarity score with the input text
        retrieved from the KG. Returns None if retrieval was not attempted due to missing parameters or if no similarity scores could be retrieved.
    """
    
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

    concept_filter = EmbeddingConceptFilter(
        concept_ids=tuple(sc.concept_id for sc in unique_standard_concepts)
    )
    similarity_scores_tuple = embedding_interface.get_nearest_concepts(
        session=session,
        model_name=text_embedding_model,
        query_embedding=text_embedding.tolist()[0],
        concept_filter=concept_filter,
        metric_type=metric_type
    )

    assert len(similarity_scores_tuple) == 1, "Expected a single set of similarity scores for the query embedding given the text embedding shape was (1, embedding_dim)."
    return similarity_scores_tuple[0] 

def find_standard_concepts(
    kg: KnowledgeGraph,
    candidate: CandidateHit,
    parent_ids: Tuple[int, ...],
    max_depth: int,
    max_paths: Optional[int] = 3,
    predicate_kinds: frozenset[ClassIDEnum] = frozenset({ClassIDEnum.IDENTITY}),
    lowest_cost: Optional[float] = None,
) -> List[StandardConcept]:
    """
    Identify standard concepts related to a candidate that satisfy parent constraints.

    Parameters
    ----------
    kg : KnowledgeGraph
        The Knowledge Graph instance.
    candidate : CandidateHit
        The initial match found by a resolver.
    parent_ids : tuple[int, ...]
        Acceptable ancestor IDs.
    max_depth : int
        Maximum separation allowed.
    max_paths : int, optional
        Limit on unique standard concepts per parent lookup.
    predicate_kinds : frozenset, optional
        Edge types to traverse.
    lowest_cost : float, optional
        Minimum cost threshold for pathfinding.

    Returns
    -------
    list[StandardConcept]
        Standard concepts associated with the candidate that hit the targets.
    """
    paths = []

    for parent in parent_ids:
        found = find_standard_paths(
            kg=kg,
            candidate=candidate,
            target=parent,
            predicate_kinds=predicate_kinds,
            max_depth=max_depth,
            max_concepts=max_paths,
            lowest_cost=lowest_cost,
        )
        paths.extend(found)

    return paths