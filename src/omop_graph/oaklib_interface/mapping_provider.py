"""The mapping provider interface over the OMOP knowledge graph."""

from collections.abc import Iterable, Iterator
from datetime import date
from typing import ClassVar, cast

from oaklib.datamodels.vocabulary import (
    HAS_DBXREF,
    SKOS_CLOSE_MATCH,
    SKOS_EXACT_MATCH,
    SKOS_RELATED_MATCH,
)
from oaklib.interfaces.mapping_provider_interface import MappingProviderInterface
from oaklib.types import CURIE, PRED_CURIE
from sssom.constants import SEMAPV
from sssom_schema import Mapping

from omop_graph.graph.model.edges import EdgeFilter, MappingStrength
from omop_graph.graph.model.nodes import ConceptView, MappingCardinality, MappingView
from omop_graph.graph.model.scope import narrowed_to_vocabularies
from omop_graph.oaklib_interface.base import as_list
from omop_graph.oaklib_interface.basic_ontology import OMOPBasicOntologyInterface
from omop_graph.oaklib_interface.unsupported import OMOPUnsupportedError


class OMOPMappingProviderInterface(OMOPBasicOntologyInterface, MappingProviderInterface):
    """
    Mixin serving the OMOP mappings as SSSOM.

    A mapping is an identity relationship of the concept, or an association classified as a
    mapping, and the concept is always its subject. Exact equivalence is
    ``skos:exactMatch`` when it is one to one and ``skos:closeMatch`` otherwise, since a
    concept that maps to several, or that shares its target, is not equivalent to it alone.
    Near equivalence is ``skos:closeMatch`` and the association ``skos:relatedMatch``.
    Every mapping states its ``mapping_cardinality``. Version lineage, such as
    ``Concept replaced by``, and the value family, such as ``Maps to value``, which only
    completes a ``Maps to``, are not mappings. Each pair is stored in both directions, so it
    is a mapping of both concepts.
    The vocabulary code of a concept is a mapping with ``oio:hasDbXref``, and is looked up
    from either the concept or its ``vocabulary_id:concept_code``.

    OMOP does not record how a mapping was made, so its justification is
    ``semapv:UnspecifiedMatching``, its date is the start of the relationship's validity
    unless that is Athena's placeholder, and its comment names the OMOP relationship.
    The source of each side is the CURIE of the vocabulary concept of its vocabulary.
    Mappings are not limited to the adapter's ``concept_filter``.
    """

    SKOS_PREDICATES: ClassVar[dict[MappingStrength, PRED_CURIE]] = {
        MappingStrength.EXACT: SKOS_EXACT_MATCH,
        MappingStrength.CLOSE: SKOS_CLOSE_MATCH,
        MappingStrength.RELATED: SKOS_RELATED_MATCH,
    }
    """The SKOS predicate of each strength of mapping."""

    PLACEHOLDER_DATE: ClassVar[date] = date(1970, 1, 1)
    """The start Athena gives to a relationship without one, which dates no mapping."""

    @property
    def _mapping_predicates(self) -> dict[str, PRED_CURIE]:
        """The SKOS predicate of each mapping relationship id."""
        return {
            relationship_id: self.SKOS_PREDICATES[strength]
            for relationship_id, strength in self.kg.mapping_predicates().items()
        }

    def _mapping(
        self,
        subject: CURIE,
        predicate: PRED_CURIE,
        object_: CURIE,
        subject_source: str,
        object_source: str,
        justification: SEMAPV = SEMAPV.UnspecifiedMatching,
        **fields,
    ) -> Mapping:
        """Build a mapping with the sources and justification."""
        return Mapping(
            subject_id=subject,
            predicate_id=predicate,
            object_id=object_,
            subject_source=subject_source,
            object_source=object_source,
            mapping_justification=justification.value,
            **fields,
        )

    def _cross_references(
        self, views: Iterable[ConceptView], source: str | None, subject: dict[int, CURIE]
    ) -> Iterator[Mapping]:
        """Yield each concept's vocabulary code as an ``oio:hasDbXref`` mapping."""
        for view in views:
            if source is None or view.vocabulary_id == source:
                vocabulary = self.codec.vocabulary_curie(view.vocabulary_id)
                yield self._mapping(
                    subject[view.concept_id],
                    HAS_DBXREF,
                    self.codec.code_curie(view.vocabulary_id, view.concept_code),
                    vocabulary,
                    vocabulary,
                )

    def sssom_mappings(
        self, curies: CURIE | Iterable[CURIE] | None = None, source: str | None = None
    ) -> Iterator[Mapping]:
        """
        Yield the mappings of concepts, or of every concept without curies.

        Parameters
        ----------
        curies : CURIE | Iterable[CURIE] | None
            Concepts to look up, as ``OMOP`` CURIEs or ``vocabulary_id:concept_code``.
        source : str | None
            Only mappings with this vocabulary on either side, as its ``vocabulary_id`` or
            its CURIE prefix.

        Yields
        ------
        Mapping
            Mappings with the looked up concept as subject. Without curies they are
            all the stored mappings, in both directions of each pair, and then the
            cross-reference of each concept within the adapter's scope.
        """
        predicates = self._mapping_predicates
        edge_filter = EdgeFilter(predicate_ids=frozenset(predicates))
        if source is not None:
            source = self.codec.vocabulary_id(source)
        if curies is None:
            for edge in self.kg.mappings(None, edge_filter, source):
                yield self._edge_mapping(edge, predicates)
            concept_filter = (
                self._population
                if source is None
                else narrowed_to_vocabularies(self._population, (source,))
            )
            for batch in self.kg.batches(self.kg.entities(concept_filter)):
                subject = {concept_id: self.codec.concept_curie(concept_id) for concept_id in batch}
                yield from self._cross_references(
                    self.kg.concept_views(batch, sort=False), source, subject
                )
            return
        requested = as_list(curies)
        ids = self._resolve_concepts(requested)
        concept_ids = tuple(dict.fromkeys(ids.values()))
        for edge in self.kg.mappings(concept_ids, edge_filter, source):
            yield self._edge_mapping(edge, predicates)
        subject = {concept_id: self.codec.concept_curie(concept_id) for concept_id in concept_ids}
        yield from self._cross_references(
            self.kg.concept_views(concept_ids, sort=False), source, subject
        )

    @staticmethod
    def _match_predicate(predicate: PRED_CURIE, cardinality: MappingCardinality) -> PRED_CURIE:
        """The predicate of a mapping, where ``skos:exactMatch`` holds only one to one."""
        if predicate == SKOS_EXACT_MATCH and cardinality is not MappingCardinality.ONE_TO_ONE:
            return SKOS_CLOSE_MATCH
        return predicate

    def _edge_mapping(self, edge: MappingView, predicates: dict[str, PRED_CURIE]) -> Mapping:
        """Build the mapping of a stored relationship."""
        return self._mapping(
            self.codec.concept_curie(edge.subject_id),
            self._match_predicate(predicates[edge.predicate_id], edge.cardinality),
            self.codec.concept_curie(edge.object_id),
            self.codec.vocabulary_curie(edge.subject_source),
            self.codec.vocabulary_curie(edge.object_source),
            mapping_date=None
            if edge.valid_start_date == self.PLACEHOLDER_DATE
            else edge.valid_start_date,
            mapping_cardinality=edge.cardinality.value,
            comment=f"OMOP relationship {edge.predicate_id}",
        )

    def sssom_mappings_by_source(self, subject_or_object_source: str | None = None):
        """Yield every mapping with the vocabulary on either side, see :meth:`sssom_mappings`."""
        yield from self.sssom_mappings(None, subject_or_object_source)

    def simple_mappings_by_curie(self, curie: CURIE) -> Iterator[tuple[PRED_CURIE, CURIE]]:
        """Yield the predicate and object of each mapping of a concept."""
        for mapping in self.sssom_mappings([curie]):
            yield mapping.predicate_id, cast(CURIE, mapping.object_id)

    def get_transitive_mappings_by_curie(self, curie: CURIE) -> Iterator[Mapping]:
        """
        Yield a mapping to each standard concept a non-standard concept maps to.

        The mapping is ``skos:exactMatch`` when it is one to one and ``skos:closeMatch``
        otherwise, as in :meth:`sssom_mappings`.

        OMOP stores a mapping in both directions and maps many concepts to one, so
        following the exact equivalences both ways would join every concept that shares
        a target. The chain is followed only from a non-standard concept to the standard
        concepts it maps to, and ends there. A standard concept has none.

        Notes
        -----
        Only one step is taken. A concept whose identity edge leads to another non-standard
        concept, which itself maps to a standard concept, gets no mapping. Following such
        chains is the horizontal transitive closure, see
        https://github.com/AustralianCancerDataNetwork/omop-graph/issues/20
        """
        concept_id = self._resolve_concepts([curie]).get(curie)
        if concept_id is None:
            return
        targets = self.kg.standard_mappings([concept_id])[concept_id]
        cardinalities = {
            edge.object_id: edge.cardinality
            for edge in self.kg.mappings(
                [concept_id],
                EdgeFilter(
                    predicate_ids=frozenset(
                        relationship_id
                        for relationship_id, strength in self.kg.mapping_predicates().items()
                        if strength is MappingStrength.EXACT
                    )
                ),
            )
        }
        views = {
            view.concept_id: view
            for view in self.kg.concept_views((concept_id, *targets), sort=False)
        }
        for other in targets:
            yield self._mapping(
                self.codec.concept_curie(concept_id),
                self._match_predicate(SKOS_EXACT_MATCH, cardinalities[other]),
                self.codec.concept_curie(other),
                self.codec.vocabulary_curie(views[concept_id].vocabulary_id),
                self.codec.vocabulary_curie(views[other].vocabulary_id),
                mapping_cardinality=cardinalities[other].value,
            )

    def get_mapping_clusters(self, *args, **kwargs):
        raise OMOPUnsupportedError("MappingProviderInterface.get_mapping_clusters")
