"""Counts over the concepts, relationships and synonyms of the graph."""

from __future__ import annotations

from omop_alchemy.cdm.query import ConceptFilter

from ..queries import (
    q_concept_counts,
    q_relationship_counts,
    q_relationship_counts_by,
    q_synonym_counts,
    q_synonym_counts_by,
)
from ..scope import ALL_CONCEPTS
from ..statistics import ConceptCount, ConceptCounts, Grouping, SynonymCounts
from ._core import GraphCore


class StatisticsMixin(GraphCore):
    """Counts, computed by the database and never by reading the concepts."""

    def concept_counts(self, scope: ConceptFilter = ALL_CONCEPTS) -> ConceptCounts:
        """Count the concepts in *scope* by vocabulary, domain and validity."""
        with self.session_factory() as session:
            return ConceptCounts(
                tuple(
                    ConceptCount(row.vocabulary_id, row.domain_id, bool(row.deprecated), row.n)
                    for row in session.execute(q_concept_counts(scope))
                )
            )

    def relationship_counts(self, scope: ConceptFilter = ALL_CONCEPTS) -> dict[str, int]:
        """
        Count the active relationships of each relationship id.

        Parameters
        ----------
        scope : ConceptFilter
            Only the relationships whose subject it selects are counted. A scope that
            selects every concept counts the whole table without joining the concepts.

        Returns
        -------
        dict[str, int]
            The relationships of each relationship id that has any, classified or not.
        """
        with self.session_factory() as session:
            return {
                row.relationship_id: row.n for row in session.execute(q_relationship_counts(scope))
            }

    def synonym_counts(self, scope: ConceptFilter = ALL_CONCEPTS) -> SynonymCounts:
        """Count the synonyms of the concepts in *scope*, and their distinct names."""
        with self.session_factory() as session:
            row = session.execute(q_synonym_counts(scope)).one()
        return SynonymCounts(row.statements, row.distinct)

    def relationship_counts_by(self, grouping: Grouping) -> dict[str, dict[str, int]]:
        """
        Count the active relationships by the group of their subject, in one pass.

        Returns
        -------
        dict[str, dict[str, int]]
            For each vocabulary or domain, the relationships of each relationship id.
        """
        found: dict[str, dict[str, int]] = {}
        with self.session_factory() as session:
            for row in session.execute(q_relationship_counts_by(grouping)):
                found.setdefault(row.group, {})[row.relationship_id] = row.n
        return found

    def synonym_counts_by(self, grouping: Grouping) -> dict[str, SynonymCounts]:
        """Count the synonyms and their distinct names by the group of their concept, in one pass."""
        with self.session_factory() as session:
            return {
                row.group: SynonymCounts(row.statements, row.distinct)
                for row in session.execute(q_synonym_counts_by(grouping))
            }
