from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from .kg import KnowledgeGraph
from .paths import StandardConcept

from difflib import SequenceMatcher

import logging
import numpy as np
import re
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StandardConceptWithScore(StandardConcept):
    total_score: float = field(compare=True, init=False)
    embedding_score: Optional[float] = field(compare=False, default=None)
    relevance: float = field(compare=False, default=0.0)
    parsimony_penalty: float = field(compare=False, default=0.0)
    broadness_bonus: float = field(compare=False, default=0.0)

    def __post_init__(self):
        object.__setattr__(self, "total_score", self.relevance - self.parsimony_penalty + self.broadness_bonus)

    def __repr__(self):
        return (f"StandardConceptWithScore(concept_id={self.concept_id} [{self.concept_name}], score={self.total_score:.4f})")

    @classmethod
    def from_standard_concept(
        cls,
        standard_concept: StandardConcept,
        embedding_score: float,
        relevance: float,
        parsimony_penalty: float,
        broadness_bonus: float,
    ) -> "StandardConceptWithScore":
        return cls(
            concept_id=standard_concept.concept_id,
            concept_name=standard_concept.concept_name,
            separation=standard_concept.separation,
            resolver_confidence=standard_concept.resolver_confidence,
            original_id=standard_concept.original_id,
            original_name=standard_concept.original_name,
            matched_label=standard_concept.matched_label,
            embedding_score=embedding_score,
            relevance=relevance,
            parsimony_penalty=parsimony_penalty,
            broadness_bonus=broadness_bonus,
        )

def score_standard_concepts(
    text: str,
    standard_concepts: tuple[StandardConcept, ...],
    kg: KnowledgeGraph,
    similarity_scores: Optional[np.ndarray] = None,
) -> list[StandardConceptWithScore]:
    ranked_concepts = []

    # Get specificity scores for the standard concepts
    concept_ids = tuple(sc.concept_id for sc in standard_concepts)
    num_ancestors = kg.get_num_ancestors(concept_ids)

    ranked_concepts = [
        _score_standard_concept(
            text=text,
            kg=kg,
            standard_concept=sc,
            num_ancestors=num_ancestors[sc.concept_id],
            similarity_score=similarity_scores[i] if similarity_scores is not None else None,
        ) for i, sc in enumerate(standard_concepts)
    ]

    return ranked_concepts


def _score_standard_concept(
    kg: KnowledgeGraph,
    text: str,
    standard_concept: StandardConcept,
    num_ancestors: int,
    similarity_score: float,
    alpha: float = 0.05,
    beta: float = 0.01,
) -> StandardConceptWithScore:
       
    textual_similarity = _textual_similarity_score(query_text=text, matched_label=standard_concept.matched_label)
    relevance = similarity_score * textual_similarity

    # Parsimony Component (Depth Penalty)
    parsimony_penalty = alpha * standard_concept.separation

    # Broadness Component (Ancestor Count Bonus - more ancestors = more general = higher bonus)
    # NOTE: np.log10?
    broadness_bonus = beta * np.log(1 + num_ancestors)

    return StandardConceptWithScore.from_standard_concept(
        standard_concept=standard_concept,
        embedding_score=similarity_score,
        relevance=relevance,
        parsimony_penalty=parsimony_penalty,
        broadness_bonus=broadness_bonus,
    )

def _textual_similarity_score(
    query_text: str,
    matched_label: str,
    similarity_threshold: float = 0.85, # Hodgkin/Hodgkins = 0.93. Acute/Subacute = 0.76.
    missing_penalty: float = 2.0,  # High penalty: If I asked for it, it better be there.
    extra_penalty: float = 0.5     # Low penalty: Extra detail is okay, but dilutes the match.
) -> float:

    # 1. Tokenize (keep standard normalization)
    stop_words = {'of', 'the', 'in', 'and', 'or', 'to', 'nos', 'a', 'an'}
    
    def tokenize(text):
        tokens = re.findall(r'\w+', text.lower())
        return [t for t in tokens if t not in stop_words]

    q_tokens = tokenize(query_text)
    m_tokens = tokenize(matched_label)

    if not q_tokens or not m_tokens:
        return 0.0

    # 2. Soft Alignment Logic
    # We try to match every Query Token to the best available Match Token
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
    denominator = n_shared + (missing_penalty * n_missing) + (extra_penalty * n_extra)
    
    if denominator == 0:
        return 0.0
        
    return n_shared / denominator