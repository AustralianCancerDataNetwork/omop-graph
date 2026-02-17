"""
Resolver strategies for mapping text to OMOP Concepts.

This module defines a hierarchy of `CandidateResolver` classes. Each resolver implements
a specific strategy (Exact Match, Partial Match, Full-Text Search) to find OMOP concepts
that match a given input string.

They are designed to be used in a `ResolverPipeline` where high-confidence resolvers
(Exact) are tried before lower-confidence ones (Partial/Fuzzy).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Optional, Tuple

from omop_graph.graph.constraints import SearchConstraintConcept
from omop_graph.graph.nodes import LabelMatch
from omop_graph.utils.types import ResolverConfidence

if TYPE_CHECKING:
    from omop_graph.graph.kg import KnowledgeGraph


@dataclass(frozen=True)
class CandidateHit:
    """
    A resolved candidate concept found by a resolver.

    Parameters
    ----------
    concept_id : int
        The OMOP Concept ID.
    resolver_confidence : ResolverConfidence
        The confidence level of the strategy that found this hit.
    matched_label : str
        The specific text in the database (name or synonym) that matched.
    """

    concept_id: int
    resolver_confidence: ResolverConfidence
    matched_label: str


class CandidateResolver(ABC):
    """
    Abstract interface for resolving free text to OMOP concept_ids.

    Resolvers are 'recall-oriented' and 'constraint-agnostic' regarding the
    validity of the concept in a specific context (that is handled later by reasoning).

    Attributes
    ----------
    confidence : ResolverConfidence
        The static confidence level assigned to hits found by this resolver.
    """

    confidence: ResolverConfidence

    @abstractmethod
    def get_matches(
        self,
        kg: "KnowledgeGraph",
        text: str,
        constraints: Optional[SearchConstraintConcept] = None,
    ) -> Tuple[LabelMatch, ...]:
        """
        Execute the search strategy against the Knowledge Graph.

        Parameters
        ----------
        kg : KnowledgeGraph
            The graph instance.
        text : str
            The input text to search for.
        constraints : SearchConstraintConcept, optional
            Filters for domain/vocabulary.

        Returns
        -------
        Tuple[LabelMatch, ...]
            A tuple of raw label matches.
        """
        ...

    def resolve(
        self,
        kg: "KnowledgeGraph",
        text: str,
        constraints: Optional[SearchConstraintConcept] = None,
        limit: Optional[int] = None,
    ) -> Iterable[CandidateHit]:
        """
        Public API to find and format candidates.

        Parameters
        ----------
        kg : KnowledgeGraph
            The graph instance.
        text : str
            The input text.
        constraints : SearchConstraintConcept, optional
            Filters.
        limit : int, optional
            Max number of hits to return.

        Returns
        -------
        Iterable[CandidateHit]
            The formatted candidate hits.
        """
        matches = self.get_matches(kg, text, constraints=constraints)
        hits = [
            CandidateHit(
                concept_id=m.concept_id,
                resolver_confidence=self.confidence,
                matched_label=m.matched_label,
            )
            for m in matches
        ]
        return hits[:limit] if limit else hits


class ExactLabelResolver(CandidateResolver):
    """
    Strategy: Exact case-insensitive match on `Concept.concept_name`.
    """

    confidence = ResolverConfidence.EXACT

    def get_matches(
        self,
        kg: "KnowledgeGraph",
        text: str,
        constraints: Optional[SearchConstraintConcept] = None,
    ) -> Tuple[LabelMatch, ...]:
        return tuple(kg.label_lookup(text, search_constraint=constraints))


class ExactSynonymResolver(CandidateResolver):
    """
    Strategy: Exact case-insensitive match on `Concept_Synonym.concept_synonym_name`.
    """

    confidence = ResolverConfidence.EXACT_SYNONYM

    def get_matches(
        self,
        kg: "KnowledgeGraph",
        text: str,
        constraints: Optional[SearchConstraintConcept] = None,
    ) -> Tuple[LabelMatch, ...]:
        return tuple(kg.synonym_lookup(text, search_constraint=constraints))


class FullTextResolver(CandidateResolver):
    """
    Strategy: Postgres Full-Text Search (tsvector) on `Concept.concept_name`.
    Matches irrespective of word order (e.g., "Kidney Cancer" -> "Cancer of Kidney").
    """

    confidence = ResolverConfidence.FULLTEXT

    def get_matches(
        self,
        kg: "KnowledgeGraph",
        text: str,
        constraints: Optional[SearchConstraintConcept] = None,
    ) -> Tuple[LabelMatch, ...]:
        return tuple(
            kg.fulltext_lookup(text, search_constraint=constraints, fuzzy=False)
        )


class FullTextSynonymResolver(CandidateResolver):
    """
    Strategy: Postgres Full-Text Search (tsvector) on `Concept_Synonym.concept_synonym_name`.
    """

    confidence = ResolverConfidence.FULLTEXT_SYNONYM

    def get_matches(
        self,
        kg: "KnowledgeGraph",
        text: str,
        constraints: Optional[SearchConstraintConcept] = None,
    ) -> Tuple[LabelMatch, ...]:
        return tuple(
            kg.fulltext_lookup(text, search_constraint=constraints, fuzzy=True)
        )


class PartialLabelResolver(CandidateResolver):
    """
    Strategy: Substring match (ILIKE %term%) on `Concept.concept_name`.
    Ranked by similarity heuristics (starts_with, length diff).
    """

    confidence = ResolverConfidence.PARTIAL

    def get_matches(
        self,
        kg: "KnowledgeGraph",
        text: str,
        constraints: Optional[SearchConstraintConcept] = None,
    ) -> Tuple[LabelMatch, ...]:
        matches = kg.label_lookup(text, fuzzy=True, search_constraint=constraints)
        ranked = sorted(
            matches, key=lambda m: self._similarity_score(text, m.matched_label)
        )
        return tuple(ranked)

    @staticmethod
    def _similarity_score(query: str, label: str) -> tuple:
        q = query.lower()
        l = label.lower()
        return (
            not l.startswith(q),  # Prefer matches starting with query (False < True)
            l.count(" "),  # Prefer fewer words
            abs(len(l) - len(q)),  # Prefer closer length
        )


class PartialSynonymResolver(PartialLabelResolver):
    """
    Strategy: Substring match (ILIKE %term%) on `Concept_Synonym.concept_synonym_name`.
    Inherits ranking logic from PartialLabelResolver.
    """

    confidence = ResolverConfidence.PARTIAL_SYNONYM

    def get_matches(
        self,
        kg: "KnowledgeGraph",
        text: str,
        constraints: Optional[SearchConstraintConcept] = None,
    ) -> Tuple[LabelMatch, ...]:
        matches = kg.synonym_lookup(text, fuzzy=True, search_constraint=constraints)
        ranked = sorted(
            matches, key=lambda m: self._similarity_score(text, m.matched_label)
        )
        return tuple(ranked)


# Default sequence of resolvers to be used in a pipeline
ALL_RESOLVERS = (
    ExactLabelResolver(),
    ExactSynonymResolver(),
    PartialLabelResolver(),
    PartialSynonymResolver(),
    FullTextResolver(),
    FullTextSynonymResolver(),
)