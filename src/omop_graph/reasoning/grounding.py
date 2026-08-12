"""
Semantic Grounding Orchestration.

This module provides the high-level `ground_term` function, which orchestrates
the full grounding pipeline:
1.  **Candidate Resolution**: Finding raw concepts that match the input text.
2.  **Hierarchy Validation**: Ensuring candidates have a valid relationship path
    to required parent concepts, when parent concepts are given. Grounding can
    also run unconstrained (no parent concepts), trading disambiguation power
    for coverage.
3.  **Standardization**: Mapping non-standard candidates to standard OMOP concepts.
4.  **Semantic Ranking**: Using embeddings and graph-based scoring to select the
    best mapping.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from omop_alchemy.cdm.query import ConceptFilter

from omop_graph.extensions.omop_alchemy import PredicateKind
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
    try_get_embedding_writer_interface,
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
        OMOP Concept IDs that act as required ancestors for any valid result. When
        None, grounding is unconstrained: candidates are still resolved and
        standardized via identity hops, but without ancestor verification. This
        trades disambiguation power for coverage. Results are ranked by relevance
        and identity-hop distance only, not by proximity to a known hierarchy branch.
    search_constraint : ConceptFilter, optional
        Domain and Vocabulary restrictions for the initial resolution phase.
    max_depth : int, optional
        Maximum allowed distance in the hierarchy between a candidate and a parent.
        Defaults to ``6``.
    predicate_kinds : frozenset[PredicateKind], optional
        Edge types allowed during BFS traversal. Defaults to IDENTITY only, which
        covers the OMOP "Maps to" and "Non-standard to Standard" relationships. Allowing
        HIERARCHY or ASSOCIATION edges would traverse parent-of and cross-domain links,
        expanding candidates to unrelated concepts and diluting grounding results.
    """

    parent_ids: Optional[Tuple[int, ...]]
    search_constraint: Optional[ConceptFilter]
    max_depth: int = 6
    predicate_kinds: frozenset[PredicateKind] = frozenset({PredicateKind.IDENTITY})

    def __post_init__(self) -> None:
        if self.predicate_kinds != frozenset({PredicateKind.IDENTITY}):
            raise ValueError(
                "predicate_kinds must be a frozenset containing only PredicateKind.IDENTITY. "
                "Other predicate kinds are not supported for grounding as scoring is not yet implemented for them."
            )


def _query_text_with_context(query: str, context: Optional[str]) -> str:
    """Fold optional free-form context into the text used for on-demand query embedding.

    Notes
    -----
    No guard on the context. The caller is responsible for what is passed in here.

    Parameters
    ----------
    query : str
        The base query text.
    context : str, optional
        Additional free-form context to append. When None or empty, ``query``
        is returned unchanged.

    Returns
    -------
    str
        ``query`` alone, or ``query`` followed by a blank line and ``context``.
    """
    if not context:
        return query
    return f"{query}\n\n{context}"


def ground_term(
    resolver_pipeline: ResolverPipeline,
    kg: KnowledgeGraph,
    query: str,
    query_embedding: Optional[np.ndarray],
    constraints: GroundingConstraints,
    max_candidates: Optional[int] = None,
    context: Optional[str] = None,
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
    context : str, optional
        Additional free-form context folded into the on-demand query-embedding
        text. Has no effect when ``query_embedding`` is supplied directly.

    Returns
    -------
    list[StandardConceptWithScore]
        A list of standard concepts sorted by their total score (descending).
    """
    standard_concepts: List[StandardConcept] = []

    search_constraints = constraints.search_constraint
    if search_constraints is not None:
        kg.check_search_constraints(search_constraints)

    # Only do on demand embedding calculation if available and needed (having a EmbeddingResolver).
    # Falls back to None to disable embedding-based features if not available or not required.
    require_embedding = HAS_OMOP_EMB and any(isinstance(resolver, EmbeddingResolver) for resolver in resolver_pipeline.resolvers)

    if query_embedding is None and require_embedding:
        embedding_writer = try_get_embedding_writer_interface(kg)
        if embedding_writer is not None:
            from omop_emb import EmbeddingRole

            query_embedding = embedding_writer.embed_texts(
                texts=(_query_text_with_context(query, context),),
                role=EmbeddingRole.QUERY,
            )
        else:
            logger.info(
                f"No embedding_writer available to embed query '{query}' on demand "
                "(the KG's embedding configuration is read-only, or none was provided at all). "
                "Embedding-based features will be disabled for this grounding operation."
            )

    if query_embedding is not None:
        assert query_embedding.shape[0] == 1, (
            "query_embedding must have shape (1, D) — one vector per call to ground_term."
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

    # Hierarchy anchoring (or unconstrained standardization when parent_ids is None)
    for hit in resolved:
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
            if constraints.parent_ids:
                logger.debug(
                    f"Failed hierarchy constraint: {hit.concept_id} ({concept_name}) "
                    f"has no path to parents {constraints.parent_ids} "
                    f"(max_depth={constraints.max_depth}, predicates={constraints.predicate_kinds})"
                )
            else:
                logger.debug(
                    f"No standard concept reachable from {hit.concept_id} ({concept_name}) "
                    f"(max_depth={constraints.max_depth}, predicates={constraints.predicate_kinds})"
                )
            continue
        standard_concepts.extend(candidate_standard_concepts)

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
    parent_ids: Optional[Tuple[int, ...]],
    max_depth: int,
    max_paths: Optional[int] = 3,
    predicate_kinds: frozenset[PredicateKind] = frozenset({PredicateKind.IDENTITY}),
) -> List[StandardConcept]:
    """
    Identify standard concepts related to a candidate, optionally satisfying parent
    constraints.

    Parameters
    ----------
    kg : KnowledgeGraph
        The Knowledge Graph instance.
    candidate : CandidateHit
        The initial match found by a resolver.
    parent_ids : tuple of int, optional
        Acceptable ancestor concept IDs. When None, no ancestor verification is
        performed. Instead, the candidate is standardized via identity hops only.
    max_depth : int
        Maximum min_levels_of_separation allowed in the ancestry check when
        parent_ids is given, or maximum identity-hop count allowed when parent_ids
        is None.
    max_paths : int, optional
        Per-target cap on unique standard concepts collected (or an overall cap when
        parent_ids is None).
    predicate_kinds : frozenset of PredicateKind, optional
        Edge types to traverse. Defaults to IDENTITY only, which covers the OMOP
        "Maps to" relationships between non-standard and standard concepts. See
        GroundingConstraints.predicate_kinds for the rationale.

    Returns
    -------
    list of StandardConcept
        Standard concepts associated with the candidate that satisfy the ancestor
        constraint, or, when parent_ids is None, every standard concept reached.
    """
    return find_standard_paths(
        kg=kg,
        candidate=candidate,
        targets=parent_ids,
        predicate_kinds=predicate_kinds,
        max_depth=max_depth,
        max_concepts=max_paths,
    )
