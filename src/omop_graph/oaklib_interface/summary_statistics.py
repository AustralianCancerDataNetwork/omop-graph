"""The summary statistics interface over the OMOP knowledge graph."""

import logging
from dataclasses import replace
from functools import cached_property
from typing import Any

from oaklib.datamodels.summary_statistics_datamodel import FacetedCount, UngroupedStatistics
from oaklib.interfaces.summary_statistics_interface import SummaryStatisticsInterface
from oaklib.types import CURIE, PRED_CURIE
from omop_alchemy.cdm.query import ConceptFilter

from omop_graph.graph.scope import ALL_CONCEPTS, every_concept, narrowed_to_vocabularies
from omop_graph.graph.statistics import ConceptCounts, Grouping, SynonymCounts
from omop_graph.oaklib_interface.basic_ontology import OMOPBasicOntologyInterface
from omop_graph.oaklib_interface.codec import OMOPMetadataKey
from omop_graph.oaklib_interface.unsupported import OMOPUnsupportedError

logger = logging.getLogger(__name__)


class OMOPSummaryStatisticsInterface(OMOPBasicOntologyInterface, SummaryStatisticsInterface):
    """
    Mixin reporting counts that the database computes.

    The concepts counted are those of the adapter, with its deprecated concepts. The edges
    counted are the active ones of the exposed relationships whose subject is in the
    adapter's vocabularies and domains. Groups are made by ``omop:vocabulary`` or
    ``omop:domain``. Entailed edges, and the descendants of branch roots, are not counted,
    since both need the relationships of every concept.
    """

    def _statistics_scope(
        self, property_values: dict[CURIE, Any] | None, prefixes: list[CURIE] | None
    ) -> ConceptFilter:
        """The adapter's concepts narrowed to the vocabularies, domains and prefixes asked for."""
        scope = self.concept_filter
        if prefixes:
            scope = narrowed_to_vocabularies(
                scope, (self.codec.vocabulary_id(prefix) for prefix in prefixes)
            )
        for key, value in (property_values or {}).items():
            if key == OMOPMetadataKey.VOCABULARY:
                scope = narrowed_to_vocabularies(scope, [value])
            elif key == OMOPMetadataKey.DOMAIN:
                scope = replace(scope, domains=(value,))
            else:
                raise ValueError(
                    f"Only {OMOPMetadataKey.VOCABULARY} and {OMOPMetadataKey.DOMAIN} group."
                )
        return scope

    def _counts(self, scope: ConceptFilter) -> ConceptCounts:
        """Count the active concepts of *scope* and the deprecated ones of its vocabularies and domains."""
        active = tuple(row for row in self.kg.concept_counts(scope).rows if not row.deprecated)
        deprecated = tuple(
            row for row in self.kg.concept_counts(every_concept(scope)).rows if row.deprecated
        )
        return ConceptCounts((*active, *deprecated))

    def metadata_property_summary_statistics(self, metadata_property: PRED_CURIE) -> dict[Any, int]:
        """Count the concepts of each vocabulary or domain, with the deprecated ones."""
        counts = self._counts(self.concept_filter)
        if metadata_property == OMOPMetadataKey.VOCABULARY:
            return counts.by_vocabulary()
        if metadata_property == OMOPMetadataKey.DOMAIN:
            return counts.by_domain()
        return {}

    @cached_property
    def _grouped_counts(
        self,
    ) -> dict[Grouping, tuple[dict[str, dict[str, int]], dict[str, SynonymCounts]]]:
        """The edge and synonym counts of every group, filled when a grouping is first asked for."""
        return {}

    def _edge_and_synonym_counts(
        self,
        scope: ConceptFilter,
        property_values: dict[CURIE, Any] | None,
        prefixes: list[CURIE] | None,
    ) -> tuple[dict[str, int], SynonymCounts]:
        """
        The counts of the exposed relationships and of the synonyms of the concepts in scope.

        A single vocabulary or domain is read from counts of every group made in one pass
        and kept, when the adapter does not restrict its vocabularies and domains.
        """
        every = every_concept(scope)
        groups = {
            OMOPMetadataKey.VOCABULARY: Grouping.VOCABULARY,
            OMOPMetadataKey.DOMAIN: Grouping.DOMAIN,
        }
        if (
            property_values
            and len(property_values) == 1
            and not prefixes
            and every_concept(self.concept_filter) == ALL_CONCEPTS
        ):
            [(key, value)] = property_values.items()
            grouping = groups[OMOPMetadataKey(key)]
            if grouping not in self._grouped_counts:
                self._grouped_counts[grouping] = (
                    self.kg.relationship_counts_by(grouping),
                    self.kg.synonym_counts_by(grouping),
                )
            by_relationship, by_synonym = self._grouped_counts[grouping]
            counts = by_relationship.get(value, {})
            synonyms = by_synonym.get(value, SynonymCounts(0, 0))
        else:
            counts = self.kg.relationship_counts(every)
            synonyms = self.kg.synonym_counts(every)
        return {
            relationship_id: count
            for relationship_id, count in counts.items()
            if relationship_id in self.kg.default_relationship_ids
        }, synonyms

    def branch_summary_statistics(
        self,
        branch_name: str | None = None,
        branch_roots: list[CURIE] | None = None,
        property_values: dict[CURIE, Any] | None = None,
        include_entailed: bool = False,
        parent=None,
        prefixes: list[CURIE] | None = None,
    ) -> UngroupedStatistics:
        """
        Count the concepts, relationships and synonyms of the adapter or of one group of it.

        Parameters
        ----------
        branch_name : str | None
            The id of the statistics, ``AllBranches`` by default.
        branch_roots : list[CURIE] | None
            Not supported.
        property_values : dict[CURIE, Any] | None
            Restrict to one ``omop:vocabulary`` or ``omop:domain``.
        include_entailed : bool
            Not supported.
        parent : GroupedStatistics | None
            Set when this is a partition of a larger group, so no metadata is added.
        prefixes : list[CURIE] | None
            Restrict to these vocabularies, by id or CURIE prefix.

        Raises
        ------
        ValueError
            If branch roots or entailed edges are asked for, or the group is by another property.
        """
        if branch_roots is not None:
            raise ValueError("Counting the branches below roots is not supported.")
        if include_entailed:
            raise ValueError("Counting entailed edges is not supported.")
        scope = self._statistics_scope(property_values, prefixes)
        counts = self._counts(scope)
        statistics = UngroupedStatistics(branch_name or "AllBranches")
        if not parent:
            self._add_statistics_metadata(statistics)
        statistics.class_count = counts.total
        statistics.deprecated_class_count = counts.deprecated
        statistics.class_count_with_text_definitions = 0
        statistics.object_property_count = len(self.kg.default_relationship_ids)
        statistics.deprecated_object_property_count = 0
        statistics.ontology_count = len(counts.by_vocabulary())
        edges, synonyms = self._edge_and_synonym_counts(scope, property_values, prefixes)
        statistics.edge_count_by_predicate = {
            curie: FacetedCount(curie, filtered_count=count)
            for relationship_id, count in edges.items()
            if (curie := self.codec.predicate_curie(relationship_id))
        }
        statistics.edge_count = sum(edges.values())
        statistics.synonym_statement_count = synonyms.statements
        statistics.distinct_synonym_count = synonyms.distinct
        self._add_derived_statistics(statistics)
        return statistics

    def summary_statistic_description(self, metric: str) -> str:
        """Not supported, no data dictionary of the statistics is bundled."""
        raise OMOPUnsupportedError("SummaryStatisticsInterface.summary_statistic_description")
