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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from omop_alchemy.cdm.query import ConceptFilter

from omop_graph.extensions.relationship_classification import PredicateKind
from omop_graph.graph.capabilities.embeddings import EmbeddingModelNotRegisteredError
from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.graph.model.edges import EdgeFilter
from omop_graph.graph.model.matches import LabelMatchKind
from omop_graph.graph.paths import StandardConcept, find_standard_paths
from omop_graph.graph.scoring import StandardConceptWithScore, score_standard_concepts
from omop_graph.reasoning.resolvers import (
    EmbeddingResolver,
    ResolverPipeline,
)

if TYPE_CHECKING:
    from omop_emb import NearestConceptMatch

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
    concept_filter : ConceptFilter, optional
        Domain and Vocabulary restrictions for the initial resolution phase.
    max_depth : int, optional
        With ``parent_ids``, the maximum ``concept_ancestor`` distance between a
        standard concept and a parent. Without it, the maximum number of identity hops
        from the candidate to a standard concept. Defaults to ``6``.
    predicate_kinds : frozenset[PredicateKind], optional
        Edge types allowed during BFS traversal. Defaults to IDENTITY only, which
        covers the OMOP "Maps to" and "Non-standard to Standard" relationships. Allowing
        HIERARCHY or ASSOCIATION edges would traverse parent-of and cross-domain links,
        expanding candidates to unrelated concepts and diluting grounding results.

    Notes
    -----
    ``max_depth`` has a different meaning in each mode:

    - With ``parent_ids`` it caps ``min_levels_of_separation`` in ``concept_ancestor``.
      The identity hops from the candidate to the standard concept are not capped.
    - Without ``parent_ids`` it caps the identity hops and no ancestor check is made.

    ``min_levels_of_separation`` is the length of the shortest chain of ancestry-defining
    relationships. ``concept_ancestor`` merges all such relationship types and does not
    record which one produced a level, so one level is not necessarily one ``Is a`` step.
    The same value is ``StandardConcept.separation``, which ranks the results.
    """

    parent_ids: tuple[int, ...] | None
    concept_filter: ConceptFilter | None
    max_depth: int = 6
    predicate_kinds: frozenset[PredicateKind] = frozenset({PredicateKind.IDENTITY})

    def __post_init__(self) -> None:
        if self.predicate_kinds != frozenset({PredicateKind.IDENTITY}):
            raise ValueError(
                "predicate_kinds must be a frozenset containing only PredicateKind.IDENTITY. "
                "Other predicate kinds are not supported for grounding as scoring is not yet implemented for them."
            )


def _query_text_with_context(query: str, context: str | None) -> str:
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


def _embedding_matches(
    kg: KnowledgeGraph,
    standard_concepts: Sequence[StandardConcept],
    query_embedding: np.ndarray,
) -> tuple[tuple[NearestConceptMatch, ...], ...] | None:
    """
    Score standard concepts against a query embedding, as the matches of a single query.

    Concepts without a stored embedding are embedded first when the KG computes missing
    embeddings, and are otherwise left unscored.

    Parameters
    ----------
    kg : KnowledgeGraph
        The knowledge graph holding the embedding store.
    standard_concepts : Sequence[StandardConcept]
        The concepts to score.
    query_embedding : np.ndarray
        The query vector, of shape ``(1, D)``.

    Returns
    -------
    tuple[tuple[NearestConceptMatch, ...], ...] | None
        The matches of the one query, or None when the KG has no embedding store, its model
        is not registered or nothing matches.
    """
    if not kg.has_embeddings:
        logger.info("The KG has no embedding store. Skipping similarity calculation.")
        return None
    concept_ids = tuple(dict.fromkeys(sc.concept_id for sc in standard_concepts))
    missing = kg.concepts_without_embedding(concept_ids)
    if missing and kg.compute_missing_embeddings:
        logger.debug(
            f"Concepts missing embeddings: {missing}. Computing missing embeddings on-the-fly."
        )
        kg.embed_concepts(missing)
    elif missing:
        logger.info(
            f"Concepts missing embeddings: {missing}.\n"
            "compute_missing_embeddings is disabled; these concepts will be skipped in similarity scoring.\n"
            "Expect missing embedding scores for these concepts in the results."
        )
    try:
        matches = kg.nearest_concepts(
            query_embedding,
            concept_filter=ConceptFilter(concept_ids=concept_ids),
            k=len(concept_ids),
        )
    except EmbeddingModelNotRegisteredError as error:
        logger.info(error)
        return None
    return (matches,) if matches else None


def ground_term(
    resolver_pipeline: ResolverPipeline,
    kg: KnowledgeGraph,
    query: str,
    query_embedding: np.ndarray | None,
    constraints: GroundingConstraints,
    max_candidates: int | None = None,
    context: str | None = None,
) -> list[StandardConceptWithScore]:
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
        The most candidates to return, taken from the ranked candidates of all resolvers
        together. If None, returns all candidates.
    context : str, optional
        Additional free-form context folded into the on-demand query-embedding
        text. Has no effect when ``query_embedding`` is supplied directly.

    Returns
    -------
    list[StandardConceptWithScore]
        A list of standard concepts sorted by their total score (descending).
    """
    standard_concepts: list[StandardConcept] = []

    concept_filter = constraints.concept_filter
    if concept_filter is not None:
        kg.check_concept_filter(concept_filter)

    # Only embed the query on demand when an EmbeddingResolver needs it and the KG can embed text.
    # Otherwise embedding-based features stay disabled for this call.
    if query_embedding is None and any(
        isinstance(resolver, EmbeddingResolver) for resolver in resolver_pipeline.resolvers
    ):
        if kg.can_embed_text:
            query_embedding = kg.embed_text(_query_text_with_context(query, context))
        else:
            logger.info(
                f"The KG cannot embed the query '{query}' on demand "
                "(its embedding configuration is read-only, or none was provided at all). "
                "Embedding-based features will be disabled for this grounding operation."
            )

    if query_embedding is not None:
        assert query_embedding.shape[0] == 1, (
            "query_embedding must have shape (1, D), one vector per call to ground_term."
        )

    resolved = list(
        resolver_pipeline.resolve(
            kg,
            query,
            concept_filter=concept_filter,
            query_embedding=query_embedding,
        )
    )
    if not resolved:
        logger.info(
            f"No candidates found for '{query}' using the resolver pipeline: {resolver_pipeline}"
        )
        return []

    # Hierarchy anchoring (or unconstrained standardization when parent_ids is None)
    found_per_hit = find_standard_paths(
        kg=kg,
        targets=constraints.parent_ids,
        candidates=resolved,
        edge_filter=EdgeFilter(predicate_kinds=constraints.predicate_kinds, within_domain=True),
        max_depth=constraints.max_depth,
    )
    for hit, found in zip(resolved, found_per_hit, strict=True):
        if found:
            standard_concepts.extend(found)
        elif constraints.parent_ids:
            logger.debug(
                f"Failed hierarchy constraint: {hit.concept_id} ({hit.matched_concept_label}) "
                f"has no path to parents {constraints.parent_ids} "
                f"(max_depth={constraints.max_depth}, predicates={constraints.predicate_kinds})"
            )
        else:
            logger.debug(
                f"No standard concept reachable from {hit.concept_id} ({hit.matched_concept_label}) "
                f"(max_depth={constraints.max_depth}, predicates={constraints.predicate_kinds})"
            )

    if not standard_concepts:
        logger.info(f"No standard concepts found for '{query}' after hierarchy validation.")
        return []

    # Only calculate nearest concept matches for the embedding resolver matches
    # split is done that semantic similarity has less concepts to search for
    matched_standard_concepts_for_embedding = [
        hit for hit in standard_concepts if hit.match_kind == LabelMatchKind.EMBEDDING
    ]
    if matched_standard_concepts_for_embedding:
        if query_embedding is None:
            raise RuntimeError(
                "Query embedding cannot be None if the EmbeddingResolver is used and returned matches."
            )
        nearest_concept_matches_for_standard_embedding_concepts = _embedding_matches(
            kg=kg,
            standard_concepts=matched_standard_concepts_for_embedding,
            query_embedding=query_embedding,
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
        standard_concepts=tuple(
            sc for sc in standard_concepts if sc.match_kind != LabelMatchKind.EMBEDDING
        ),
        kg=kg,
        nearest_concept_matches=None,  # No embedding-based scoring for non-embedding matches
    )

    # combine the two different ones
    standard_concepts_with_score = (
        embedding_standard_concepts_with_score + non_embedding_standard_concepts_with_score
    )

    best_by_concept_id: dict[int, StandardConceptWithScore] = {}
    for concept in standard_concepts_with_score:
        existing = best_by_concept_id.get(concept.concept_id)
        if existing is None or concept.total_score > existing.total_score:
            best_by_concept_id[concept.concept_id] = concept

    deduped_ranked = sorted(
        best_by_concept_id.values(), key=lambda sc: sc.total_score, reverse=True
    )
    return deduped_ranked[:max_candidates] if max_candidates is not None else deduped_ranked
