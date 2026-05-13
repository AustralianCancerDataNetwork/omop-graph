"""
Scoring algorithms for ranking resolved concepts.

This module implements the logic for scoring candidate OMOP concepts based on:
1.  **Relevance:** How well the text matches the query (embeddings + string similarity).
2.  **Parsimony:** Penalizing deep graph traversals (finding a concept far away).
3.  **Broadness:** Rewarding concepts that are more general (higher ancestor count), 
    often useful for finding category headers.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, List, Optional, Tuple, Mapping

import numpy as np

# Local Application Imports
from omop_graph.graph.paths import StandardConcept

if TYPE_CHECKING:
    from omop_graph.graph.kg import KnowledgeGraph
    from omop_emb.utils.embedding_utils import NearestConceptMatch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StandardConceptWithScore(StandardConcept):
    """
    A StandardConcept enriched with scoring metrics.

    Attributes
    ----------
    total_score : float
        The final calculated score used for ranking.
        Formula: `relevance - parsimony_penalty + broadness_bonus`
    embedding_score : float, optional
        The cosine similarity score from the embedding model.
    relevance : float
        The relevance score used for ranking: embedding similarity when available,
        textual similarity otherwise.
    parsimony_penalty : float
        Penalty based on graph distance (separation).
    broadness_bonus : float
        Bonus based on the concept's generality (ancestor count).
    """

    total_score: float = field(compare=True, default=0.0)
    embedding_score: Optional[float] = field(compare=False, default=None)
    relevance: float = field(compare=False, default=0.0)
    parsimony_penalty: float = field(compare=False, default=0.0)
    broadness_bonus: float = field(compare=False, default=0.0)

    def __repr__(self) -> str:
        return (
            f"StandardConceptWithScore("
            f"concept_id={self.concept_id} [{self.concept_name}], "
            f"score={self.total_score:.4f})"
        )

    @classmethod
    def from_standard_concept(
        cls,
        standard_concept: StandardConcept,
        embedding_score: Optional[float],
        relevance: float,
        parsimony_penalty: float,
        broadness_bonus: float,
        total_score: float,
    ) -> "StandardConceptWithScore":
        """
        Factory method to promote a StandardConcept to a scored version.
        """
        return cls(
            **vars(standard_concept),
            embedding_score=embedding_score,
            relevance=relevance,
            parsimony_penalty=parsimony_penalty,
            broadness_bonus=broadness_bonus,
            total_score=total_score,
        )


def score_standard_concepts(
    text: str,
    standard_concepts: tuple[StandardConcept, ...],
    kg: "KnowledgeGraph",
    nearest_concept_matches: Optional[Tuple[Tuple[NearestConceptMatch, ...], ...]] = None,
) -> List[StandardConceptWithScore]:
    """
    Attach scoring metrics to each standard concept.

    Notes
    -----
    Scores are computed but the returned list preserves the input order.
    Callers are responsible for sorting if ranking is required.

    Parameters
    ----------
    text : str
        The original query text.
    standard_concepts : tuple[StandardConcept, ...]
        The candidate concepts to score.
    kg : KnowledgeGraph
        The graph instance used for retrieving metadata (ancestor counts).
    nearest_concept_matches : Tuple[Tuple[NearestConceptMatch, ...], ...], optional
        Pre-computed nearest-concept matches from the embedding index.  The outer
        tuple corresponds to query vectors in order; each inner tuple holds the
        nearest matches for that query vector.  Currently only a single query
        vector is supported.

    Returns
    -------
    list[StandardConceptWithScore]
        Scored concepts in the same order as ``standard_concepts``.
    """
    # Get specificity scores (ancestor counts) for the standard concepts
    sc_dict = {sc.concept_id: sc for sc in standard_concepts}
    num_ancestors = kg.get_num_ancestors(tuple(sc_dict.keys()))
    
    # singular text
    if nearest_concept_matches is None:
        nearest_concept_matches_dict = ({}, )
    else:
        nearest_concept_matches_dict = tuple(
            {
                match.concept_id: match.similarity
                for match in matches_for_query
            }
            for matches_for_query in nearest_concept_matches
        )
    

    assert len(nearest_concept_matches_dict) == 1, "Currently only supports scoring with a single query embedding vector for the singular text input"
    nearest_concept_matches_dict_for_single_query = nearest_concept_matches_dict[0]

    ranked_concepts = [
        _score_standard_concept(
            text=text,
            kg=kg,
            standard_concept=sc,
            num_ancestors=num_ancestors.get(sc.concept_id, 0),
            similarity_score=nearest_concept_matches_dict_for_single_query.get(sc.concept_id, None),
        )
        for sc in standard_concepts
    ]
    return ranked_concepts


def _score_standard_concept(
    kg: "KnowledgeGraph",
    text: str,
    standard_concept: StandardConcept,
    num_ancestors: int,
    similarity_score: Optional[float],
    alpha: float = 0.05,
    beta: float = 0.01,
) -> StandardConceptWithScore:
    """
    Calculate the score for a single concept.

    Notes
    -----
    Parsimony: Penalizes concepts that are found deeper in the graph (higher separation).
    Broadness: Rewards concepts that are more general (higher ancestor count). Uses a log scale to dampen the effect of extremely high ancestor counts.
    Relevance: Contextual similarity (if embedding_score is provided) or textual similarity as fallback

    Parameters
    ----------
    kg : KnowledgeGraph
        Graph instance.
    text : str
        Query text.
    standard_concept : StandardConcept
        The concept being scored.
    num_ancestors : int
        Number of ancestors (proxy for generality).
    similarity_score : float, optional
        Embedding cosine similarity. If None, no embedding relevance will be factored in.
    alpha : float, optional
        Weight for parsimony penalty (separation cost). Default 0.05.
    beta : float, optional
        Weight for broadness bonus. Default 0.01.

    Returns
    -------
    StandardConceptWithScore
        The scored concept.
    """

    if similarity_score is None:
        relevance = _textual_similarity_score(
        query_text=text, matched_label=standard_concept.matched_label
    )
    else:
        relevance = similarity_score

    parsimony_penalty = alpha * standard_concept.separation
    broadness_bonus = (beta * np.log(1 + num_ancestors)).item()

    total_score = relevance - parsimony_penalty + broadness_bonus

    return StandardConceptWithScore.from_standard_concept(
        standard_concept=standard_concept,
        embedding_score=similarity_score,
        relevance=relevance,
        parsimony_penalty=parsimony_penalty,
        broadness_bonus=broadness_bonus,
        total_score=total_score,
    )


def _textual_similarity_score(
    query_text: str,
    matched_label: str,
    similarity_threshold: float = 0.85,
    missing_penalty: float = 2.0,
    extra_penalty: float = 0.5,
) -> float:
    """
    Compute a custom token-based similarity score.

    This scoring is asymmetric: it penalizes missing query tokens heavily
    (the concept MUST cover what was asked), but penalizes extra tokens lightly
    (the concept can be more specific).

    Parameters
    ----------
    query_text : str
        The user's query.
    matched_label : str
        The label of the candidate concept.
    similarity_threshold : float, optional
        Minimum Levenshtein ratio to consider two tokens a 'match'. Default 0.85.
    missing_penalty : float, optional
        Penalty weight for tokens in query but not in label. Default 2.0.
    extra_penalty : float, optional
        Penalty weight for tokens in label but not in query. Default 0.5.

    Returns
    -------
    float
        A score between 0.0 and 1.0.
    """
    # 1. Tokenize (keep standard normalization)
    stop_words = {"of", "the", "in", "and", "or", "to", "nos", "a", "an"}

    def tokenize(text: str) -> List[str]:
        tokens = re.findall(r"\w+", text.lower())
        return [t for t in tokens if t not in stop_words]

    q_tokens = tokenize(query_text)
    m_tokens = tokenize(matched_label)

    if not q_tokens or not m_tokens:
        return 0.0

    # 2. Soft Alignment Logic
    # Try to match every Query Token to the best available Match Token
    matched_m_indices = set()
    n_shared = 0

    for q_word in q_tokens:
        best_score = 0.0
        best_idx = -1

        # Find the best match in m_tokens that hasn't been used yet
        for i, m_word in enumerate(m_tokens):
            if i in matched_m_indices:
                continue

            # Levenshtein ratio as similarity score
            score = SequenceMatcher(None, q_word, m_word).ratio()

            if score > best_score:
                best_score = score
                best_idx = i

        # Did we find a match good enough to call "Shared"?
        if best_score >= similarity_threshold:
            n_shared += 1
            matched_m_indices.add(best_idx)

    # 3. Calculate Penalties
    # Any query token that didn't find a buddy is "Missing"
    n_missing = len(q_tokens) - n_shared

    # Any match token that wasn't used is "Extra"
    n_extra = len(m_tokens) - n_shared

    # 4. Final Score
    # Score = Shared / (Shared + Weighted Penalties)
    denominator = n_shared + (missing_penalty * n_missing) + (extra_penalty * n_extra)

    if denominator == 0:
        return 0.0

    return n_shared / denominator