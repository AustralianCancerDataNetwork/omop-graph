from abc import ABC, abstractmethod
from typing import Iterable
from dataclasses import dataclass
from enum import Enum
from omop_graph.graph.kg import KnowledgeGraph


@dataclass(frozen=True)
class CandidateHit:
    concept_id: int
    resolver: str

class ResolverConfidence(Enum):
    EXACT = 0
    PARTIAL = 1
    EMBEDDING = 2
    EXTERNAL = 3

class CandidateResolver(ABC):
    """
    Interface for resolving free text to OMOP concept_ids.

    This stage is recall-oriented and constraint-agnostic.
    """
    confidence: ResolverConfidence
    name: str

    @abstractmethod
    def resolve(
        self,
        kg: KnowledgeGraph,
        text: str,
        *,
        limit: int | None = None,
    ) -> Iterable[CandidateHit]:
        ...
