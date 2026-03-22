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
    find_standard_concepts_unconstrained,
    get_unique_standard_concepts,
)
from omop_graph.graph.scoring import StandardConceptWithScore, score_standard_concepts
from omop_graph.reasoning.resolvers import (
    CandidateHit,
    ResolverConfidence,
    ResolverPipeline,
)
from omop_graph.extensions.emb import MissingExtensionError
from omop_llm import LLMClient

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from omop_emb import EmbeddingService



@dataclass(frozen=True)
class GroundingConstraints:
    """
    Configuration for restricting the grounding search space.

    Parameters
    ----------
    parent_ids : tuple[int, ...], optional
        OMOP Concept IDs that act as required ancestors for any valid result.
    search_constraint : SearchConstraintConcept, optional
        Domain and Vocabulary restrictions for the final grounded concepts.
    resolver_search_constraint : SearchConstraintConcept, optional
        Optional pre-filter for the initial resolution phase. If omitted,
        ``ground_term`` falls back to using ``search_constraint`` for backward
        compatibility.
    max_depth : int, optional
        Maximum allowed distance in the hierarchy between a candidate and a parent.
    predicate_kinds : frozenset[ClassIDEnum], optional
        The types of relationships allowed during pathfinding.
    """

    parent_ids: Optional[Tuple[int, ...]]
    search_constraint: Optional[SearchConstraintConcept]
    resolver_search_constraint: Optional[SearchConstraintConcept] = None
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
    embedding_service: Optional["EmbeddingService"] = None,
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
    embedding_service : EmbeddingService, optional
        Optional omop-emb orchestration service. If omitted, the function will
        lazily use ``kg.emb_service`` when embedding workflows are needed.
    constraints : GroundingConstraints
        Contextual constraints (parents, domains, etc.) to apply.
    max_candidates : int, optional
        Limit for the number of candidates returned. If None, returns all candidates.

    Returns
    -------
    list[StandardConceptWithScore]
        A list of standard concepts sorted by their total score (descending).

    """
    standard_concepts: List[StandardConcept] = []

    # 1. Validate Constraints
    search_constraints = constraints.search_constraint
    resolver_search_constraints = (
        constraints.resolver_search_constraint
        if constraints.resolver_search_constraint is not None
        else search_constraints
    )
    if search_constraints is not None:
        search_constraints.check(kg)
    if resolver_search_constraints is not None and resolver_search_constraints is not search_constraints:
        resolver_search_constraints.check(kg)

    # 2. Resolve Text to Candidate Hits
    resolved = list(
        resolver_pipeline.resolve(
            kg,
            text,
            constraints=resolver_search_constraints,
            text_embedding=text_embedding,
            text_embedding_model=text_embedding_model,
            embedding_client=embedding_client,
            embedding_service=embedding_service,
        )
    )

    # 3. Validate Hierarchy and Standardize
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
            standard_concepts.extend(
                find_standard_concepts_unconstrained(
                    kg=kg,
                    candidate=hit,
                    predicate_kinds=constraints.predicate_kinds,
                    max_concepts=3,
                )
            )

    # 4. Filter and Deduplicate
    unique_standard_concepts = get_unique_standard_concepts(standard_concepts)
    if search_constraints is not None:
        unique_standard_concepts = [
            standard_concept
            for standard_concept in unique_standard_concepts
            if _grounded_concept_matches_search_constraint(
                kg,
                standard_concept.concept_id,
                search_constraints,
            )
        ]
    if not unique_standard_concepts:
        logger.info(
            "No standard concepts found for '%s' after grounding candidate standardization.",
            text,
        )
        return []

    # 5. Semantic Scoring (Embeddings)
    try:
        with kg.session_factory() as session:
            similarity_scores_dict = {}
            concept_ids = tuple(sc.concept_id for sc in unique_standard_concepts)
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
                    concept_ids=concept_ids
                )

            missing_concept_ids = tuple(
                sc.concept_id
                for sc in unique_standard_concepts
                if sc.concept_id not in similarity_scores_dict
            )

            if (
                missing_concept_ids and
                text_embedding_model is not None and
                embedding_client is not None and
                text_embedding is not None
            ):
                logger.info(
                    "Backfilling %s grounded concept embeddings for model '%s' during scoring.",
                    len(missing_concept_ids),
                    text_embedding_model,
                )
                service = embedding_service or kg.emb_service
                concept_name_by_id = {
                    sc.concept_id: sc.concept_name
                    for sc in unique_standard_concepts
                }
                missing_embeddings = service.embed_and_upsert_concepts(
                    session=session,
                    model_name=text_embedding_model,
                    concept_ids=missing_concept_ids,
                    concept_texts=tuple(concept_name_by_id[concept_id] for concept_id in missing_concept_ids),
                    embedding_client=embedding_client,
                )
                assert isinstance(text_embedding, np.ndarray), "Text embedding must be a numpy array for fallback similarity scoring."
                assert text_embedding.shape[0] == 1 and text_embedding.ndim == 2, "Text embedding must be a 2D vector with first dim = 1."
                missing_similarity_scores = _cosine_similarity_row(
                    text_embedding[0],
                    missing_embeddings,
                )
                similarity_scores_dict.update({
                    concept_id: float(score)
                    for concept_id, score in zip(missing_concept_ids, missing_similarity_scores.tolist())
                })

            if not similarity_scores_dict:
                if text_embedding_model is not None:
                    logger.warning((
                        f"Embedding model '{text_embedding_model}' is not registered in the KG or no concept embeddings could be scored. "
                        f"Fallback to embedding client is not possible (embedding_client: {embedding_client is not None}, text_embedding: {text_embedding is not None}).")
                    )
                similarity_scores = None
            else:
                similarity_scores = [
                    similarity_scores_dict.get(sc.concept_id)
                    for sc in unique_standard_concepts
                ]
    except MissingExtensionError:
        logger.info(
            "Embedding-based grounding not available. Install omop-graph[emb] for PostgreSQL-backed embeddings "
            "or install omop-emb with the backend extra you need."
        )
        similarity_scores = None

    # 6. Rank Results
    ranked_standard_concepts = score_standard_concepts(
        text=text, 
        standard_concepts=tuple(unique_standard_concepts),
        kg=kg,
        similarity_scores=similarity_scores
    )

    ranked_standard_concepts.sort(key=lambda sc: sc.total_score, reverse=True)
    return ranked_standard_concepts[:max_candidates] if max_candidates is not None else ranked_standard_concepts


def _cosine_similarity_row(query_embedding: np.ndarray, candidate_embeddings: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between one query vector and many candidate vectors.
    """
    query = np.asarray(query_embedding, dtype=np.float32)
    candidates = np.asarray(candidate_embeddings, dtype=np.float32)
    if candidates.ndim != 2:
        raise ValueError(f"Expected candidate_embeddings to be 2D, got shape {candidates.shape}.")

    query_norm = np.linalg.norm(query)
    candidate_norms = np.linalg.norm(candidates, axis=1)
    denom = np.maximum(query_norm * candidate_norms, 1e-12)
    return (candidates @ query) / denom


def _grounded_concept_matches_search_constraint(
    kg: KnowledgeGraph,
    concept_id: int,
    search_constraint: SearchConstraintConcept,
) -> bool:
    """
    Apply a SearchConstraintConcept to a grounded concept view in memory.

    This is used after grounding so we can support permissive resolver-stage
    retrieval while still enforcing stricter constraints on the final grounded
    concepts.
    """
    concept = kg.concept_view(concept_id)

    if search_constraint.domains is not None and concept.domain_id not in search_constraint.domains:
        return False
    if search_constraint.vocabs is not None and concept.vocabulary_id not in search_constraint.vocabs:
        return False
    if search_constraint.require_standard and not concept.standard_concept:
        return False

    return True


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
