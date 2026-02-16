from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from .kg import KnowledgeGraph
from .paths import StandardConcept

from omop_graph.utils.types import ResolverConfidence

import logging
import numpy as np
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
            embedding_score=embedding_score,
            relevance=relevance,
            parsimony_penalty=parsimony_penalty,
            broadness_bonus=broadness_bonus,
        )

def score_standard_concepts(
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
            kg=kg,
            standard_concept=sc,
            num_ancestors=num_ancestors[sc.concept_id],
            similarity_score=similarity_scores[i] if similarity_scores is not None else None,
        ) for i, sc in enumerate(standard_concepts)
    ]

    return ranked_concepts


def _score_standard_concept(
    kg: KnowledgeGraph,
    standard_concept: StandardConcept,
    num_ancestors: int,
    similarity_score: Optional[float] = None,
    alpha: float = 0.05,
    beta: float = 0.01,
) -> StandardConceptWithScore:
    
    # NOTE: They may need adaptation
    match_multipliers = {
        ResolverConfidence.EXACT: 1.0,
        ResolverConfidence.EXACT_SYNONYM: 0.95,
        ResolverConfidence.PARTIAL: 0.90,
        ResolverConfidence.PARTIAL_SYNONYM: 0.85
    }

    embedding_score = similarity_score if similarity_score is not None else 1.0

    relevance = embedding_score * match_multipliers.get(standard_concept.resolver_confidence, 0.8)

    # 2. Parsimony Component (The "Anti-Drift" Brake)
    # Subtract score for depth. 
    # Example: Depth 3 costs 0.15 score.
    parsimony_penalty = alpha * standard_concept.separation

    # 3. Broadness Component (The "Safety" Net)
    # Log scale: 10 children is better than 0, but 1000 isn't 100x better.
    broadness_bonus = beta * np.log(1 + num_ancestors)

    return StandardConceptWithScore.from_standard_concept(
        standard_concept=standard_concept,
        embedding_score=embedding_score,
        relevance=relevance,
        parsimony_penalty=parsimony_penalty,
        broadness_bonus=broadness_bonus,
    )