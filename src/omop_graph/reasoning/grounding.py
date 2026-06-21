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
from typing import List, Optional, Tuple

import numpy as np

from omop_graph.config import OmopGraphConfig
from omop_graph.extensions.omop_alchemy import PredicateKind
from omop_graph.graph.constraints import SearchConstraintConcept
from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.graph.paths import (
    StandardConcept,
    find_standard_paths,
)
from omop_graph.graph.scoring import (
    StandardConceptWithScore, 
    score_standard_concepts
)
from omop_graph.reasoning.resolvers import (
    CandidateHit,
    ResolverPipeline,
    EmbeddingResolver,
)
from omop_graph.graph.nodes import LabelMatchKind
from omop_graph.extensions.emb import (
    get_embedding_writer_interface,
    semantic_similarity,
    HAS_OMOP_EMB,
)

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
    predicate_kinds : frozenset[PredicateKind], optional
        The types of relationships allowed during pathfinding.
    """

    parent_ids: Optional[Tuple[int, ...]]
    search_constraint: Optional[SearchConstraintConcept]
    max_depth: int = field(
        default_factory=lambda: OmopGraphConfig.get_config().max_depth
    )
    predicate_kinds: frozenset[PredicateKind] = frozenset({PredicateKind.IDENTITY})


def ground_term(
    resolver_pipeline: ResolverPipeline,
    kg: KnowledgeGraph,
    query: str,
    query_embedding: Optional[np.ndarray],
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
    query : str
        The input query to ground.
    query_embedding : np.ndarray
        The embedding vector for the input query. When None and a writer interface is
        available, the embedding is computed on demand from ``query``.
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
        kg.check_search_constraints(search_constraints)

    # Only do on demand calculation if needed (for embedding resolver) and available.
    # Falls back to None to disable embedding-based features if not available or not required.
    if HAS_OMOP_EMB and any(isinstance(resolver, EmbeddingResolver) for resolver in resolver_pipeline.resolvers):
        require_embedding = True
    else:
        require_embedding = False

    if query_embedding is None and require_embedding:
        embedding_writer = get_embedding_writer_interface(kg)
        if embedding_writer is not None:
            from omop_emb.embeddings import EmbeddingRole

            query_embedding = embedding_writer.embed_texts(
                texts=(query,),
                embedding_role=EmbeddingRole.QUERY,
            )

    if query_embedding is not None:
        assert query_embedding.shape[0] == 1, (
            "query_embedding must have shape (1, D) — one vector per call to ground_term."
        )
    else:
        if require_embedding:
            logger.info(
                f"No text embedding provided for '{query}' and no embedding_writer available. "
                "Embedding-based features will be disabled for this grounding operation."
            )

    resolved = list(
        resolver_pipeline.resolve(
            kg,
            query,
            constraints=search_constraints,
            query_embedding=query_embedding,
        )
    )
    if not resolved:
        logger.info(
            f"No candidates found for '{query}' using the resolver pipeline: {resolver_pipeline}"
        )
        return []

    # Hierarchy anchoring
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
                    f"has no path to parents {constraints.parent_ids} "
                    f"(max_depth={constraints.max_depth}, predicates={constraints.predicate_kinds})"
                )
                continue
            standard_concepts.extend(candidate_standard_concepts)
        else:
            raise NotImplementedError("Grounding without parent_ids is not supported.")

    if not standard_concepts:
        logger.info(
            f"No standard concepts found for '{query}' after hierarchy validation."
        )
        return []
    
    # Only calculate nearest concept matches for the embedding resolver matches
    # split is done that semantic similarity has less concepts to search for
    matched_standard_concepts_for_embedding = [
        hit for hit in standard_concepts if hit.match_kind == LabelMatchKind.EMBEDDING
    ]
    if matched_standard_concepts_for_embedding:
        if query_embedding is None:
            raise RuntimeError("Query embedding cannot be None if the EmbeddingResolver is used and returned matches.")
        nearest_concept_matches_for_standard_embedding_concepts = (
            semantic_similarity(
                kg=kg, 
                standard_concepts=matched_standard_concepts_for_embedding, 
                query_embedding=query_embedding
            )
        )
        embedding_standard_concepts_with_score = score_standard_concepts(
            text=query,
            standard_concepts=tuple(matched_standard_concepts_for_embedding),
            kg=kg,
            nearest_concept_matches=nearest_concept_matches_for_standard_embedding_concepts,
        )
    else:
        embedding_standard_concepts_with_score = []

    # Score the other standard concepts that were not matched via embedding resolver 
    non_embedding_standard_concepts_with_score = score_standard_concepts(
        text=query,
        standard_concepts=tuple(sc for sc in standard_concepts if sc.match_kind != LabelMatchKind.EMBEDDING),
        kg=kg,
        nearest_concept_matches=None,  # No embedding-based scoring for non-embedding matches
    )

    # combine the two different ones
    standard_concepts_with_score = embedding_standard_concepts_with_score + non_embedding_standard_concepts_with_score

    best_by_concept_id: dict[int, StandardConceptWithScore] = {}
    for concept in standard_concepts_with_score:
        existing = best_by_concept_id.get(concept.concept_id)
        if existing is None or concept.total_score > existing.total_score:
            best_by_concept_id[concept.concept_id] = concept

    deduped_ranked = sorted(
        best_by_concept_id.values(), key=lambda sc: sc.total_score, reverse=True
    )
    return (
        deduped_ranked[:max_candidates]
        if max_candidates is not None
        else deduped_ranked
    )


def find_standard_concepts(
    kg: KnowledgeGraph,
    candidate: CandidateHit,
    parent_ids: Tuple[int, ...],
    max_depth: int,
    max_paths: Optional[int] = 3,
    predicate_kinds: frozenset[PredicateKind] = frozenset({PredicateKind.IDENTITY}),
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
    return find_standard_paths(
        kg=kg,
        candidate=candidate,
        targets=parent_ids,
        predicate_kinds=predicate_kinds,
        max_depth=max_depth,
        max_concepts=max_paths,
        lowest_cost=lowest_cost,
    )
