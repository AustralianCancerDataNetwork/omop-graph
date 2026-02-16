from abc import ABC, abstractmethod
from typing import Iterable, Tuple
from dataclasses import dataclass

from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.graph.nodes import LabelMatch
from omop_graph.utils.types import ResolverConfidence
from omop_graph.graph.constraints import SearchConstraintConcept

from dataclasses import dataclass
from typing import Optional, Iterable



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
        constraints: Optional[SearchConstraintConcept] = None,
    ) -> Tuple[LabelMatch, ...]:
        ...

    def resolve(
        self,
        kg: KnowledgeGraph,
        text: str,
        constraints: Optional[SearchConstraintConcept] = None,
        limit: int | None = None,
    ) -> Iterable[CandidateHit]:
        matches = self.get_matches(kg, text, constraints=constraints)
        hits = [
            CandidateHit(m.concept_id, self.confidence)
            for m in matches
        ]
        return hits[:limit] if limit else hits

class ExactLabelResolver(CandidateResolver):
    confidence = ResolverConfidence.EXACT

    def get_matches(self, kg: KnowledgeGraph, text: str, constraints: Optional[SearchConstraintConcept] = None) -> Tuple[LabelMatch, ...]:
        return tuple([match for match in kg.label_lookup(text, search_constraint=constraints)])    

class ExactSynonymResolver(ExactLabelResolver):
    confidence = ResolverConfidence.EXACT_SYNONYM
    
    def get_matches(self, kg: KnowledgeGraph, text: str, constraints: Optional[SearchConstraintConcept] = None) -> Tuple[LabelMatch, ...]:
        return tuple([match for match in kg.synonym_lookup(text, search_constraint=constraints)])
    

class PartialLabelResolver(CandidateResolver):
    confidence = ResolverConfidence.PARTIAL

    def get_matches(self, kg: KnowledgeGraph, text: str, constraints: Optional[SearchConstraintConcept] = None) -> Tuple[LabelMatch, ...]:
        matches = kg.label_lookup(text, fuzzy=True, search_constraint=constraints)
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
    
class PartialSynonymResolver(PartialLabelResolver):
    confidence = ResolverConfidence.PARTIAL_SYNONYM

    def get_matches(self, kg: KnowledgeGraph, text: str, constraints: Optional[SearchConstraintConcept] = None) -> Tuple[LabelMatch, ...]:
        matches = kg.synonym_lookup(text, fuzzy=True, search_constraint=constraints)
        ranked = sorted(
            matches,
            key=lambda m: self._similarity_score(text, m.matched_label)
        )
        return tuple(ranked)
    

ALL_RESOLVERS = (
    ExactLabelResolver(),
    ExactSynonymResolver(),
    PartialLabelResolver(),
    PartialSynonymResolver(),
)