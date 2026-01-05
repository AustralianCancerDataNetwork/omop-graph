from typing import Iterable
from omop_graph.graph import KnowledgeGraph
from .base import CandidateResolver, CandidateHit, ResolverConfidence

def _similarity_score(query: str, label: str) -> tuple:
    q = query.lower()
    l = label.lower()

    return (
        not l.startswith(q),        # startswith is best
        l.count(" "),               # fewer words
        abs(len(l) - len(q)),       # length difference
    )


class PartialLabelResolver(CandidateResolver):
    name = "partial_label"
    confidence = ResolverConfidence.PARTIAL

    def resolve(self, kg: KnowledgeGraph, text: str, *, limit: int | None = None) -> Iterable[CandidateHit]:
        matches = kg.label_lookup(text, fuzzy=True)
        ranked = sorted(
            matches,
            key=lambda m: _similarity_score(text, m.matched_label)
        )
        hits = [
            CandidateHit(m.concept_id, self.name)
            for m in ranked
        ]
        return hits[:limit] if limit else hits