from abc import ABC, abstractmethod
from typing import Iterable, Tuple
from dataclasses import dataclass
from enum import Enum
from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.graph.nodes import LabelMatch

from dataclasses import dataclass
from typing import Optional, Iterable

class ResolverConfidence(Enum):
    EXACT = 0
    EXACT_SYNONYM = 1
    PARTIAL = 2
    EMBEDDING = 3
    EXTERNAL = 4

    def __lt__(self, other: "ResolverConfidence") -> bool:
        return self.value < other.value

@dataclass(frozen=True)
class CandidateHit:
    concept_id: int
    resolver_confidence: ResolverConfidence

class CandidateResolver(ABC):
    """
    Interface for resolving free text to OMOP concept_ids.

    This stage is recall-oriented and constraint-agnostic.
    """
    confidence: ResolverConfidence

    @abstractmethod
    def get_matches(
        self,
        kg: KnowledgeGraph,
        text: str,
    ) -> Tuple[LabelMatch, ...]:
        ...

    def resolve(
        self,
        kg: KnowledgeGraph,
        text: str,
        *,
        limit: int | None = None,
    ) -> Iterable[CandidateHit]:
        matches = self.get_matches(kg, text)
        hits = [
            CandidateHit(m.concept_id, self.confidence)
            for m in matches
        ]
        return hits[:limit] if limit else hits

class ExactLabelResolver(CandidateResolver):
    confidence = ResolverConfidence.EXACT

    def get_matches(self, kg: KnowledgeGraph, text: str) -> Tuple[LabelMatch, ...]:
        return kg.label_lookup(text)

class ExactSynonymResolver(ExactLabelResolver):
    confidence = ResolverConfidence.EXACT_SYNONYM
    
    def get_matches(self, kg: KnowledgeGraph, text: str) -> Tuple[LabelMatch, ...]:
        return kg.synonym_lookup(text)
    

class PartialLabelResolver(CandidateResolver):
    confidence = ResolverConfidence.PARTIAL

    def get_matches(self, kg: KnowledgeGraph, text: str) -> Tuple[LabelMatch, ...]:
        matches = kg.label_lookup(text, fuzzy=True)
        ranked = sorted(
            matches,
            key=lambda m: self._similarity_score(text, m.matched_label)
        )
        return tuple(ranked)
    
    @staticmethod
    def _similarity_score(query: str, label: str) -> tuple:
        q = query.lower()
        l = label.lower()

        return (
            not l.startswith(q),        # startswith is best
            l.count(" "),               # fewer words
            abs(len(l) - len(q)),       # length difference
        )