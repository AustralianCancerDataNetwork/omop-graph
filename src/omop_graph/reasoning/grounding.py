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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np

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

    # Optional: Semantic Scoring (Embeddings)
    try:
        with kg.session_factory() as session:
            similarity_scores_dict = {}
            if (
                text_embedding_model is not None and 
                kg.emb.is_model_registered(session=session, model_name=text_embedding_model) and
                text_embedding is not None
            ):
                assert isinstance(text_embedding, np.ndarray), "Text embedding must be a numpy array for RAG retrieval."
                assert text_embedding.shape[0] == 1 and text_embedding.ndim == 2, "Text embedding must be a 2D vector with first dim = 1."
                similarity_scores_dict = kg.emb.get_similarities(
                    session=session,
                    embedding_model_name=text_embedding_model,
                    text_embedding=text_embedding.tolist()[0],
                    concept_ids=tuple(sc.concept_id for sc in unique_standard_concepts)
                )

            if not similarity_scores_dict:
                # Either the filtering resulted in no concepts or the embedding model is not registered. 
                if text_embedding_model is not None and kg.emb.is_model_registered(
                    session=session,
                    model_name=text_embedding_model
                ):
                    logger.warning(
                        f"Filtering resulted in no concepts with available embeddings for model '{text_embedding_model}'.")
                
                # Fallback (but just debug)
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
                    similarity_scores = embedding_client.cosine_similarity(
                        text_embedding, standard_concept_embeddings
                    )[0]

                    # Store them in the DB for faster acess in the future
                    kg.emb.add_to_db(
                        embeddings=standard_concept_embeddings,
                        concept_ids=tuple([sc.concept_id for sc in unique_standard_concepts]),
                        session=session,
                        model=text_embedding_model
                    )

                else:
                    if text_embedding_model is not None:
                        logger.warning((
                            f"Embedding model '{text_embedding_model}' is not registered in the KG. No similarity scores will be available. "
                            f"Fallback to embedding client is not possible (embedding_client: {embedding_client is not None}, text_embedding: {text_embedding is not None}).")
                        )
                    similarity_scores = None
            else:
                similarity_scores = np.array(list(similarity_scores_dict.values()))
    except MissingExtensionError:
        logger.info("Embedding-based grounding not available. Install this package with [emb] option to enable this functionality.")
        similarity_scores = None

    # Scoring
    ranked_standard_concepts = score_standard_concepts(
        text=text, 
        standard_concepts=tuple(unique_standard_concepts),
        kg=kg,
        similarity_scores=similarity_scores
    )

    ranked_standard_concepts.sort(key=lambda sc: sc.total_score, reverse=True)
    return ranked_standard_concepts[:max_candidates] if max_candidates is not None else ranked_standard_concepts


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