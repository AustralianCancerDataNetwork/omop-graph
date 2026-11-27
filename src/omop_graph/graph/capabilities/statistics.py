"""Counts over the concepts, relationships and synonyms of the graph."""

from __future__ import annotations

import functools
from dataclasses import replace

from omop_alchemy.cdm.query import ConceptFilter

from ..queries import (
    q_concept_counts,
    q_relationship_counts,
    q_synonym_counts,
)
from ..scope import ALL_CONCEPTS, every_concept
from ..statistics import ConceptCount, ConceptCounts, Grouping, GroupStatistics, SynonymCounts
from ._core import GraphCore


class StatisticsMixin(GraphCore):
    """Counts, computed by the database and never by reading the concepts."""

    def concept_counts(self, concept_filter: ConceptFilter = ALL_CONCEPTS) -> ConceptCounts:
        """
        Count the concepts in ``concept_filter`` by vocabulary, domain and validity.

        Parameters
        ----------
        concept_filter : ConceptFilter
            The concepts to count.

        Returns
        -------
        ConceptCounts
            The counts by vocabulary, domain and validity.
        """
        with self.session_factory() as session:
            return ConceptCounts(
                tuple(
                    ConceptCount(row.vocabulary_id, row.domain_id, bool(row.deprecated), row.n)
                    for row in session.execute(q_concept_counts(concept_filter))
                )
            )

    def relationship_counts(self, concept_filter: ConceptFilter = ALL_CONCEPTS) -> dict[str, int]:
        """
        Count the active relationships of each relationship id.

        Parameters
        ----------
        concept_filter : ConceptFilter
            Only the relationships whose subject it selects are counted. A scope that
            selects every concept counts the whole table without joining the concepts.

        Returns
        -------
        dict[str, int]
            The relationships of each relationship id that has any, classified or not.
        """
        with self.session_factory() as session:
            return {
                row.relationship_id: row.n
                for row in session.execute(q_relationship_counts(concept_filter))
            }

    def synonym_counts(self, concept_filter: ConceptFilter = ALL_CONCEPTS) -> SynonymCounts:
        """
        Count the synonyms of the concepts in ``concept_filter``, and their distinct names.

        Parameters
        ----------
        concept_filter : ConceptFilter
            The concepts whose synonyms are counted.

        Returns
        -------
        SynonymCounts
            The number of synonyms and of distinct names.
        """
        with self.session_factory() as session:
            row = session.execute(q_synonym_counts(concept_filter)).one()
        return SynonymCounts(row.statements, row.distinct)

    def relationship_counts_by(self, grouping: Grouping) -> dict[str, dict[str, int]]:
        """
        Count the active relationships by the group of their subject, in one pass.

        Parameters
        ----------
        grouping : Grouping
            Whether to group by vocabulary or by domain.

        Returns
        -------
        dict[str, dict[str, int]]
            For each vocabulary or domain, the relationships of each relationship id.
        """
        found: dict[str, dict[str, int]] = {}
        with self.session_factory() as session:
            for row in session.execute(q_relationship_counts(ALL_CONCEPTS, grouping)):
                found.setdefault(row.group, {})[row.relationship_id] = row.n
        return found

    def synonym_counts_by(self, grouping: Grouping) -> dict[str, SynonymCounts]:
        """
        Count the synonyms and their distinct names by the group of their concept, in one pass.

        Parameters
        ----------
        grouping : Grouping
            Whether to group by vocabulary or by domain.

        Returns
        -------
        dict[str, SynonymCounts]
            The synonym counts of each group.
        """
        with self.session_factory() as session:
            return {
                row.group: SynonymCounts(row.statements, row.distinct)
                for row in session.execute(q_synonym_counts(ALL_CONCEPTS, grouping))
            }

    @functools.cached_property
    def _group_statistics(self) -> dict[Grouping, dict[str, GroupStatistics]]:
        """The statistics of every group, filled when a grouping is first asked for."""
        return {}

    def group_statistics(self, grouping: Grouping) -> dict[str, GroupStatistics]:
        """
        The relationships and synonyms of each vocabulary or domain, counted in one pass.

        The first call for a grouping counts every group together, which is slow on a full
        Athena release, and the result is kept for the life of the graph. It is cheaper
        than counting each group on its own when several groups are wanted.

        Parameters
        ----------
        grouping : Grouping
            Whether to group by vocabulary or by domain.

        Returns
        -------
        dict[str, GroupStatistics]
            The statistics of each group that has a relationship or a synonym.
        """
        if grouping not in self._group_statistics:
            relationships = self.relationship_counts_by(grouping)
            synonyms = self.synonym_counts_by(grouping)
            self._group_statistics[grouping] = {
                group: GroupStatistics(
                    relationships.get(group, {}), synonyms.get(group, SynonymCounts(0, 0))
                )
                for group in sorted(relationships.keys() | synonyms.keys())
            }
        return self._group_statistics[grouping]

    def scope_statistics(
        self, concept_filter: ConceptFilter = ALL_CONCEPTS, by_group: bool = False
    ) -> GroupStatistics:
        """
        Count the relationships and synonyms of the concepts concept_filter selects.

        Concepts are counted whatever their standardness or validity, so only the
        vocabularies, domains and ids of concept_filter apply.

        Parameters
        ----------
        concept_filter : ConceptFilter
            The scope to count.
        by_group : bool
            If True and the scope is a single vocabulary or domain, it is read from
            :meth:`group_statistics`, which counts every group once for the life of the
            graph. This pays off when several groups are asked for, and is slow on a full
            Athena release for the first.

        Returns
        -------
        GroupStatistics
            The active relationships of each relationship id, classified or not, counted by
            their subject, and the synonyms.
        """
        every = every_concept(concept_filter)
        if by_group:
            for grouping, field in (
                (Grouping.VOCABULARY, "vocabularies"),
                (Grouping.DOMAIN, "domains"),
            ):
                group = getattr(every, field)
                if group and len(group) == 1 and replace(every, **{field: None}).is_empty():
                    return self.group_statistics(grouping).get(
                        group[0], GroupStatistics({}, SynonymCounts(0, 0))
                    )
        return GroupStatistics(self.relationship_counts(every), self.synonym_counts(every))
