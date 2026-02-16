from dataclasses import dataclass, field
from typing import Optional

from omop_alchemy.cdm.model.vocabulary import (
    Concept,
    Concept_Ancestor,
    Concept_Relationship,
    Relationship,
    Concept_Synonym,
)

from sqlalchemy.sql import Select


@dataclass(frozen=True)
class SearchConstraintConcept:
    "Search constraints that can be applied to the Concept table."
    domains: Optional[tuple[str, ...]] = field(default=None)
    vocabs: Optional[tuple[str, ...]] = field(default=None)
    require_standard: bool = False

    def apply(self, stmt: Select) -> Select:
        if self.domains is not None:
            stmt = stmt.where(Concept.domain_id.in_(self.domains))
        if self.vocabs is not None:
            stmt = stmt.where(Concept.vocabulary_id.in_(self.vocabs))
        if self.require_standard:
            stmt = stmt.where(Concept.standard_concept.in_(["S", "C"]))
        return stmt
    
    def check(self, kg) -> None:
        """
        Checks that the specified constraints are valid (e.g., that the specified domains and vocabularies exist in the KG).
        Raises a ValueError if any of the constraints are invalid.
        """
        from omop_graph.graph.kg import KnowledgeGraph  # dynamic import to avoid circular import
        assert isinstance(kg, KnowledgeGraph), "kg should be an instance of KnowledgeGraph"

        if self.domains is not None:
            valid_domains = kg.get_all_concept_domain_ids()
            assert all(domain in valid_domains for domain in self.domains), f"Invalid domain constraint: {self.domains}."
        
        if self.vocabs is not None:
            valid_vocabs = kg.get_all_concept_vocabulary_ids()
            assert all(vocab in valid_vocabs for vocab in self.vocabs), f"Invalid vocabulary constraint: {self.vocabs}."
