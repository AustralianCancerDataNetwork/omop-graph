"""The mapping provider interface over the OMOP knowledge graph."""

from collections.abc import Iterable, Iterator
from dataclasses import replace
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

from omop_graph.extensions.omop_alchemy import PredicateKind, PredicateSubkind
from omop_graph.graph.edges import EdgeFilter
from omop_graph.graph.nodes import ConceptView, MappingView
from omop_graph.oaklib_interface.basic_ontology import OMOPBasicOntologyInterface
from omop_graph.oaklib_interface.unsupported import OMOPUnsupportedError


class OMOPMappingProviderInterface(OMOPBasicOntologyInterface, MappingProviderInterface):
    """
    Mixin serving the OMOP mappings as SSSOM.

    A mapping is an identity relationship of the concept, or an association classified as a
    mapping, and the concept is always its subject. Exact equivalence is
    ``skos:exactMatch``, near equivalence ``skos:closeMatch`` and the association
    ``skos:relatedMatch``. Version lineage, such as ``Concept replaced by``, is not a
    mapping. Each pair is stored in both directions, so it is a mapping of both concepts.
    The vocabulary code of a concept is a mapping with ``oio:hasDbXref``, and is looked up
    from either the concept or its ``vocabulary_id:concept_code``.

    OMOP does not record how a mapping was made, so its justification is
    ``semapv:UnspecifiedMatching``, its date is the start of the relationship's validity
    and its comment names the OMOP relationship. Mappings are not limited to the adapter's
    ``concept_filter``.
    """

    MAPPING_PREDICATES: ClassVar[dict[PredicateSubkind, PRED_CURIE]] = {
        PredicateSubkind(PredicateKind.IDENTITY, "Exact equivalence"): SKOS_EXACT_MATCH,
        PredicateSubkind(PredicateKind.IDENTITY, "Near equivalence"): SKOS_CLOSE_MATCH,
        PredicateSubkind(PredicateKind.ASSOCIATION, "Mapping"): SKOS_RELATED_MATCH,
    }
    """The SKOS predicate of each relationship subkind that is a mapping."""

    @property
    def _mapping_predicates(self) -> dict[str, PRED_CURIE]:
        """The SKOS predicate of each mapping relationship id."""
        return {
            predicate.relationship_id: self.MAPPING_PREDICATES[subkind]
            for predicate in self.kg.predicates()
            if (subkind := PredicateSubkind(predicate.predicate_kind, predicate.predicate_subkind))
            in self.MAPPING_PREDICATES
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
                prefix = self.codec.vocabulary_prefix(view.vocabulary_id)
                yield self._mapping(
                    subject[view.concept_id],
                    HAS_DBXREF,
                    self.codec.code_curie(view.vocabulary_id, view.concept_code),
                    prefix,
                    prefix,
                )

    def sssom_mappings(
        self, curies: CURIE | Iterable[CURIE] | None = None, source: str | None = None
    ) -> Iterator[Mapping]:
        """
        Yield the mappings of concepts, or of every concept without *curies*.

        Parameters
        ----------
        curies : CURIE | Iterable[CURIE] | None
            Concepts to look up. A ``vocabulary_id:concept_code`` gives only the
            cross-reference mapping of its concept.
        source : str | None
            Only mappings with this vocabulary on either side, as its ``vocabulary_id`` or
            its CURIE prefix.

        Yields
        ------
        Mapping
            Mappings with the looked up concept as subject. Without *curies* they are
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
            scope = (
                self.concept_filter
                if source is None
                else replace(self.concept_filter, vocabularies=(source,))
            )
            for batch in self.kg.batches(self.kg.entities(scope)):
                subject = {concept_id: self.codec.concept_curie(concept_id) for concept_id in batch}
                yield from self._cross_references(
                    self.kg.concept_views(batch, sort=False), source, subject
                )
            return
        requested = [curies] if isinstance(curies, str) else list(curies)
        ids = self._resolve_concepts(requested)
        concepts = {c: i for c, i in ids.items() if self.codec.is_omop_prefix(c.partition(":")[0])}
        for edge in self.kg.mappings(concepts.values(), edge_filter, source):
            yield self._edge_mapping(edge, predicates)
        subject = {concept_id: self.codec.concept_curie(concept_id) for concept_id in ids.values()}
        yield from self._cross_references(
            self.kg.concept_views(tuple(dict.fromkeys(ids.values())), sort=False), source, subject
        )

    def _edge_mapping(self, edge: MappingView, predicates: dict[str, PRED_CURIE]) -> Mapping:
        """Build the mapping of a stored relationship."""
        return self._mapping(
            self.codec.concept_curie(edge.subject_id),
            predicates[edge.predicate_id],
            self.codec.concept_curie(edge.object_id),
            self.codec.vocabulary_prefix(edge.subject_source),
            self.codec.vocabulary_prefix(edge.object_source),
            mapping_date=edge.valid_start_date,
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
        Yield a ``skos:exactMatch`` to each standard concept a non-standard concept maps to.

        OMOP stores a mapping in both directions and maps many concepts to one, so
        following the exact equivalences both ways would join every concept that shares
        a target. The chain is followed only from a non-standard concept to the standard
        concepts it maps to, and ends there. A standard concept has none.
        """
        exact = frozenset(r for r, p in self._mapping_predicates.items() if p == SKOS_EXACT_MATCH)
        concept_id = self._resolve_concepts([curie]).get(curie)
        if concept_id is None:
            return
        reached = self.kg.standardise([concept_id], EdgeFilter(predicate_ids=exact))[concept_id]
        targets = tuple(other for other in reached if other != concept_id)
        views = {
            view.concept_id: view
            for view in self.kg.concept_views((concept_id, *targets), sort=False)
        }
        for other in targets:
            yield self._mapping(
                self.codec.concept_curie(concept_id),
                SKOS_EXACT_MATCH,
                self.codec.concept_curie(other),
                self.codec.vocabulary_prefix(views[concept_id].vocabulary_id),
                self.codec.vocabulary_prefix(views[other].vocabulary_id),
            )

    def get_mapping_clusters(self, *args, **kwargs):
        raise OMOPUnsupportedError("MappingProviderInterface.get_mapping_clusters")
