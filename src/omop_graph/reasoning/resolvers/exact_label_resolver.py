from typing import Iterable, Tuple
from omop_graph.graph import KnowledgeGraph
from .base import CandidateResolver, CandidateHit, ResolverConfidence
from omop_graph.graph.nodes import LabelMatch

class ExactLabelResolver(CandidateResolver):
    name = "exact_label"
    confidence = ResolverConfidence.EXACT

    def get_matches(self, kg: KnowledgeGraph, text: str) -> Tuple[LabelMatch, ...]:
        return kg.label_lookup(text)
    
    def resolve(self, kg: KnowledgeGraph, text: str, *, limit: int | None = None) -> Iterable[CandidateHit]:
        matches = self.get_matches(kg, text)
        hits = [
            CandidateHit(m.concept_id, self.name)
            for m in matches
        ]
        return hits[:limit] if limit else hits


class ExactSynonymResolver(ExactLabelResolver):
    name = "exact_synonym"
    
    def get_matches(self, kg: KnowledgeGraph, text: str) -> Tuple[LabelMatch, ...]:
        return kg.synonym_lookup(text)