from dataclasses import dataclass, field
from typing import Optional, Tuple, TYPE_CHECKING

from sqlalchemy.sql import Select

from omop_alchemy.cdm.model.vocabulary import Concept

if TYPE_CHECKING:
    from omop_graph.graph.kg import KnowledgeGraph


@dataclass(frozen=True)
class SearchConstraintConcept:
    """
    Search constraints that can be applied to queries against the OMOP Concept table.

    This class encapsulates filters for domains, vocabularies, and standardization
    flags, allowing for composable query construction.

    Parameters
    ----------
    concept_ids : tuple[int, ...], optional
        A tuple of OMOP Concept IDs to filter by.
        If None, no concept ID filtering is applied.
    domains : tuple[str, ...], optional
        A tuple of OMOP Domain IDs to filter by (e.g., ('Condition', 'Drug')).
        If None, no domain filtering is applied.
    vocabs : tuple[str, ...], optional
        A tuple of OMOP Vocabulary IDs to filter by (e.g., ('SNOMED', 'RxNorm')).
        If None, no vocabulary filtering is applied.
    require_standard : bool, optional
        If True, restricts results to standard ('S') or classification ('C') concepts.
        Default is False.
    """
    concept_ids: Optional[Tuple[int, ...]] = field(default=None)
    domains: Optional[Tuple[str, ...]] = field(default=None)
    vocabs: Optional[Tuple[str, ...]] = field(default=None)
    require_standard: bool = False

    def apply(self, query: Select) -> Select:
        """
        Apply the constraints to a SQLAlchemy Select statement.

        Parameters
        ----------
        query : Select
            The initial SQLAlchemy Select statement targeting the Concept table.

        Returns
        -------
        Select
            The modified Select statement with where clauses appended.
        """
        if self.concept_ids is not None:
            query = query.where(Concept.concept_id.in_(self.concept_ids))

        if self.domains is not None:
            query = query.where(Concept.domain_id.in_(self.domains))
        
        if self.vocabs is not None:
            query = query.where(Concept.vocabulary_id.in_(self.vocabs))
            
        if self.require_standard:
            # Filters for 'S' (Standard) or 'C' (Classification)
            query = query.where(Concept.standard_concept.in_(["S", "C"]))
            
        return query

    def check(self, kg: "KnowledgeGraph") -> None:
        """
        Validate that the specified constraints exist within the Knowledge Graph.

        This method checks if the requested domains and vocabularies are actually
        present in the connected database.

        Parameters
        ----------
        kg : KnowledgeGraph
            The Knowledge Graph instance to validate against.

        Raises
        ------
        TypeError
            If the provided `kg` is not an instance of `KnowledgeGraph`.
        ValueError
            If any specified domain or vocabulary ID is invalid/missing in the DB.
        """
        # Dynamic import to avoid circular dependency
        from omop_graph.graph.kg import KnowledgeGraph
        
        if not isinstance(kg, KnowledgeGraph):
            raise TypeError("The 'kg' argument must be an instance of KnowledgeGraph.")

        if self.domains is not None:
            valid_domains = kg.get_all_concept_domain_ids()
            invalid = [d for d in self.domains if d not in valid_domains]
            if invalid:
                raise ValueError(
                    f"Invalid domain constraint(s): {invalid}. "
                    f"Available domains: {sorted(list(valid_domains))}"
                )

        if self.vocabs is not None:
            valid_vocabs = kg.get_all_concept_vocabulary_ids()
            invalid = [v for v in self.vocabs if v not in valid_vocabs]
            if invalid:
                raise ValueError(
                    f"Invalid vocabulary constraint(s): {invalid}. "
                    f"Available vocabularies: {sorted(list(valid_vocabs))}"
                )