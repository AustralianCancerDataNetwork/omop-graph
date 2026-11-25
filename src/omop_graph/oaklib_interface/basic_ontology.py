"""The basic ontology interface over the OMOP knowledge graph."""

import logging
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import replace
from functools import cached_property
from itertools import batched

from oaklib.datamodels.vocabulary import (
    DEPRECATED_PREDICATE,
    HAS_DBXREF,
    HAS_EXACT_SYNONYM,
    INVERSE_OF,
    LABEL_PREDICATE,
    OWL_CLASS,
    OWL_META_CLASSES,
    OWL_OBJECT_PROPERTY,
    OWL_VERSION_INFO,
    TERM_REPLACED_BY,
)
from oaklib.interfaces.basic_ontology_interface import (
    ALIAS_MAP,
    METADATA_MAP,
    BasicOntologyInterface,
)
from oaklib.types import CURIE, PRED_CURIE
from omop_alchemy.cdm.query import ConceptFilter

from omop_graph.extensions.omop_alchemy import HierarchyDirection, PredicateKind
from omop_graph.graph import KnowledgeGraph
from omop_graph.graph.edges import IS_A_RELATIONSHIP_ID, EdgeFilter, EdgeView, Predicate
from omop_graph.graph.nodes import ConceptView
from omop_graph.graph.queries import DEFAULT_CONCEPT_FILTER
from omop_graph.oaklib_interface.base import OMOPBaseInterface
from omop_graph.oaklib_interface.codec import (
    REPLACED_BY_RELATIONSHIP_ID,
    OMOPLanguage,
)
from omop_graph.oaklib_interface.unsupported import OMOPUnsupportedError

logger = logging.getLogger(__name__)


class OMOPBasicOntologyInterface(OMOPBaseInterface, BasicOntologyInterface):
    """
    Mixin implementing the basic ontology interface over an OMOP knowledge graph.

    Concepts are ``owl:Class`` entities and relationships are ``owl:ObjectProperty``
    entities, both addressed by CURIE. Each vocabulary is an ontology, identified by the
    CURIE of its vocabulary concept.

    Parameters
    ----------
    kg : KnowledgeGraph
        The OMOP Knowledge Graph instance.
    concept_filter : ConceptFilter
        The concepts the adapter enumerates. Defaults to active standard and
        classification concepts. ``limit`` and ``concept_ids`` are rejected because they
        would silently truncate every enumeration.
    standardise : bool
        If True, a non-standard concept stands in for the standard concepts it maps to
        when walking the hierarchy and entailing relationships. Its walks include those
        concepts and their ancestors, and its entailed relationships those of each of them,
        reported for the concept itself. Off by default, since the result then holds
        concepts that no stored edge connects to the concept.

    Raises
    ------
    ValueError
        If ``concept_filter`` sets ``limit`` or ``concept_ids``.
    """

    def __init__(
        self,
        *args,
        kg: KnowledgeGraph,
        concept_filter: ConceptFilter = DEFAULT_CONCEPT_FILTER,
        standardise: bool = False,
        **kwargs,
    ):
        if concept_filter.limit is not None or concept_filter.concept_ids is not None:
            raise ValueError("The adapter's concept_filter must not set limit or concept_ids.")
        super().__init__(*args, kg=kg, **kwargs)
        self.concept_filter = concept_filter
        self.standardise = standardise

    @cached_property
    def predicates_by_curie(self) -> dict[PRED_CURIE, Predicate]:
        """The classified relationships, keyed by their predicate CURIE."""
        return {
            self.codec.predicate_curie(predicate.relationship_id): predicate
            for predicate in self.kg.predicates()
        }

    @cached_property
    def default_relationship_ids(self) -> frozenset[str]:
        """
        The classified relationships exposed when no predicate is named.

        A Hierarchy relationship is exposed in its up direction only, so each hierarchical
        edge appears once, from the child to the parent. The down direction is available
        by naming it.
        """
        return frozenset(
            predicate.relationship_id
            for predicate in self.predicates_by_curie.values()
            if not (
                predicate.predicate_kind is PredicateKind.HIERARCHY
                and predicate.direction is HierarchyDirection.DOWN
            )
        )

    def _scope(
        self, filter_obsoletes: bool, id_prefixes: Iterable[str] | None = None
    ) -> ConceptFilter:
        """
        The adapter's concept filter with the oaklib enumeration arguments applied.

        ``filter_obsoletes`` replaces the adapter's choice. ``id_prefixes`` narrows the
        vocabularies, each named by its ``vocabulary_id`` or its CURIE prefix. The ``OMOP``
        prefix is every concept's own prefix, so naming it does not narrow.
        """
        scope = replace(self.concept_filter, require_active=filter_obsoletes)
        if not id_prefixes or any(map(self.codec.is_omop_prefix, id_prefixes)):
            return scope
        vocabularies = dict.fromkeys(self.codec.vocabulary_id(prefix) for prefix in id_prefixes)
        return replace(
            scope,
            vocabularies=tuple(
                vocabulary
                for vocabulary in vocabularies
                if scope.vocabularies is None or vocabulary in scope.vocabularies
            ),
        )

    @property
    def _every_concept(self) -> ConceptFilter:
        """
        The adapter's vocabularies and domains, whatever the standardness or validity.

        It selects every concept that ``entities(filter_obsoletes=False)`` lists, which
        includes the ones ``obsoletes`` yields.
        """
        return replace(
            self.concept_filter,
            require_standard=False,
            include_classification=False,
            require_active=False,
        )

    def _resolve_concepts(self, curies: Iterable[CURIE]) -> dict[CURIE, int]:
        """Concept ids of the CURIEs that name a concept in the graph, others left out."""
        curies = list(curies)
        resolved = {
            curie: concept_id
            for curie in curies
            if (concept_id := self.codec.resolve_concept(curie)) is not None
        }
        if len(resolved) < len(set(curies)):
            left_out = [curie for curie in dict.fromkeys(curies) if curie not in resolved]
            logger.debug("%d CURIEs name no concept, such as %s.", len(left_out), left_out[:3])
        return resolved

    def _concept_ids(self, curies: Iterable[CURIE] | None) -> tuple[int, ...] | None:
        """
        The ids of the CURIEs that name a concept, or None for no restriction.

        An info message names the CURIEs when none of them names a concept, since nothing
        then matches.
        """
        if curies is None:
            return None
        curies = list(curies)
        ids = tuple(dict.fromkeys(self._resolve_concepts(curies).values()))
        if curies and not ids:
            logger.info("No CURIE names a concept, so nothing matches: %s", curies[:3])
        return ids

    def _concept_views(self, curies: Iterable[CURIE]) -> dict[CURIE, ConceptView]:
        """The concept views of the CURIEs that name an existing concept, others left out."""
        ids = self._resolve_concepts(curies)
        views = {
            view.concept_id: view for view in self.kg.concept_views(tuple(ids.values()), sort=False)
        }
        return {curie: views[cid] for curie, cid in ids.items() if cid in views}

    def ontologies(self) -> Iterable[CURIE]:
        """Yield one ontology per vocabulary, as the CURIE of its vocabulary concept."""
        for vocabulary in self.kg.vocabularies():
            yield self.codec.concept_curie(vocabulary.vocabulary_concept_id)

    def ontology_metadata_map(self, ontology: CURIE) -> METADATA_MAP:
        """
        Name, version and reference of a vocabulary. Empty when *ontology* is not one.
        """
        for vocabulary in self.kg.vocabularies():
            if self.codec.concept_curie(vocabulary.vocabulary_concept_id) != ontology:
                continue
            metadata: METADATA_MAP = {
                "id": [ontology],
                LABEL_PREDICATE: [vocabulary.vocabulary_name],
                "omop:vocabulary_id": [vocabulary.vocabulary_id],
            }
            if vocabulary.vocabulary_version:
                metadata[OWL_VERSION_INFO] = [vocabulary.vocabulary_version]
            if vocabulary.vocabulary_reference:
                metadata["omop:vocabulary_reference"] = [vocabulary.vocabulary_reference]
            return metadata
        return {}

    def ontology_versions(self, ontology: CURIE) -> Iterable[str]:
        """The version of a vocabulary, when it has one."""
        return iter(self.ontology_metadata_map(ontology).get(OWL_VERSION_INFO, []))

    def defined_bys(  # ty: ignore[invalid-method-override]
        self, entities: Iterable[CURIE]
    ) -> Iterable[tuple[CURIE, CURIE | None]]:
        """
        Yield each entity with the ontology of its vocabulary. Relationships and unknown
        entities have none.
        """
        entities = list(entities)
        vocabulary_curies = {
            vocabulary.vocabulary_id: self.codec.concept_curie(vocabulary.vocabulary_concept_id)
            for vocabulary in self.kg.vocabularies()
        }
        views = self._concept_views(entities)
        for entity in entities:
            view = views.get(entity)
            yield entity, vocabulary_curies.get(view.vocabulary_id) if view else None

    def obsoletes(self, include_merged: bool = True) -> Iterable[CURIE]:
        """
        Yield the deprecated concepts, within the adapter's vocabularies and domains.

        Standardness is not required, so every concept listed here is also listed by
        ``entities(filter_obsoletes=False)``.

        Parameters
        ----------
        include_merged : bool
            If True, include the concepts updated into a replacement (``invalid_reason``
            ``U``) besides the deleted ones (``D``).
        """
        for cid in self.kg.obsolete_concepts(include_merged, self._every_concept):
            yield self.codec.concept_curie(cid)

    def dangling(self, curies: Iterable[CURIE] | None = None) -> Iterable[CURIE]:
        """
        Yield the CURIEs that name no concept or relationship.

        Foreign keys keep relationships from pointing at missing concepts, so nothing is
        dangling when *curies* is None.
        """
        if curies is None:
            return
        curies = list(curies)
        known = {curie for curie, _ in self.owl_types(curies)}
        yield from (curie for curie in curies if curie not in known)

    def owl_types(self, entities: Iterable[CURIE]) -> Iterable[tuple[CURIE, CURIE]]:
        """Yield ``owl:Class`` for each concept and ``owl:ObjectProperty`` for each relationship."""
        entities = list(entities)
        for entity in entities:
            if entity in self.predicates_by_curie:
                yield entity, OWL_OBJECT_PROPERTY
        for entity in self._concept_views(entities):
            yield entity, OWL_CLASS

    def terms_categories(self, curies: Iterable[CURIE]) -> Iterable[tuple[CURIE, CURIE]]:
        """Yield each concept with the CURIE of the concept that stands for its domain."""
        domain_ids = self.kg.domain_concept_ids()
        for curie, view in self._concept_views(curies).items():
            if view.domain_id in domain_ids:
                yield curie, self.codec.concept_curie(domain_ids[view.domain_id])

    def subsets(self) -> Iterable[CURIE]:
        """OMOP defines no subsets."""
        return iter(())

    def subset_members(self, subset: CURIE) -> Iterable[CURIE]:
        """OMOP defines no subsets."""
        return iter(())

    def definition(self, curie: CURIE, lang: str | None = None) -> str | None:
        """OMOP has no text definitions."""
        return None

    def comments(
        self, curies: Iterable[CURIE], allow_none: bool = True, lang: str | None = None
    ) -> Iterable[tuple[CURIE, str | None]]:
        """OMOP has no comments, so each CURIE gets None when *allow_none* is set."""
        if allow_none:
            for curie in curies:
                yield curie, None

    def languages(self) -> Iterable[str]:
        """The ISO 639-1 codes of the languages that synonyms are written in."""
        for language_id in self.kg.synonym_language_ids():
            language = OMOPLanguage.from_concept_id(language_id)
            if language is not None:
                yield language.code

    def entities(
        self,
        filter_obsoletes: bool = True,
        owl_type: str | None = None,
        *,
        concept_filter: ConceptFilter | None = None,
    ) -> Iterable[CURIE]:
        """
        Iterate over the concepts, or the relationships, of the graph.

        Parameters
        ----------
        filter_obsoletes : bool
            If True, the concepts the adapter's ``concept_filter`` selects, active ones
            only. With the default filter these are the active standard and
            classification concepts. If False, every concept in the filter's vocabularies
            and domains. The filter's standardness and validity requirements are dropped,
            so deprecated and non-standard concepts are included.
        owl_type : str | None
            ``owl:Class`` or None for concepts, ``owl:ObjectProperty`` for relationships
            as predicate CURIEs. The other OWL types have no entities here and yield nothing.
        concept_filter : ConceptFilter | None
            Replaces the adapter's ``concept_filter`` for this call and is used as given,
            so ``filter_obsoletes`` does not apply.

        Yields
        -------
        CURIE
            Concept or predicate identifiers.

        Raises
        ------
        ValueError
            If ``owl_type`` is not an OWL type.
        """
        if owl_type is not None and owl_type not in OWL_META_CLASSES:
            raise ValueError(
                f"{owl_type!r} is not an OWL type; expected one of {OWL_META_CLASSES}."
            )
        if owl_type == OWL_OBJECT_PROPERTY:
            yield from self.predicates_by_curie
            return
        if owl_type not in (None, OWL_CLASS):
            return

        scope = concept_filter or (
            self._scope(filter_obsoletes) if filter_obsoletes else self._every_concept
        )
        # Consume the generator so no session stays open.
        for cid in tuple(self.kg.entities(scope)):
            yield self.codec.concept_curie(cid)

    def _enumerate(
        self,
        build: Callable[[ConceptFilter, EdgeFilter | None], tuple[int, ...]],
        predicates: Iterable[PRED_CURIE] | None,
        scope: ConceptFilter,
    ) -> Iterator[CURIE]:
        """Yield the concepts *build* selects, over the closure or over the named predicates."""
        edge_filter = (
            EdgeFilter(predicate_ids=self.codec.relationship_ids(predicates))
            if predicates
            else None
        )
        for cid in build(scope, edge_filter):
            yield self.codec.concept_curie(cid)

    def roots(
        self,
        predicates: list[PRED_CURIE] | None = None,
        ignore_owl_thing: bool = True,
        filter_obsoletes: bool = True,
        annotated_roots: bool = False,
        id_prefixes: list[str] | None = None,
    ) -> Iterable[CURIE]:
        """
        Yield the concepts without a parent.

        Without *predicates* the parents come from the ``concept_ancestor`` closure, which
        is OMOP's own ancestry. With them, a parent is the target of an edge of one of the
        predicates. ``ignore_owl_thing`` and ``annotated_roots`` have no effect.
        ``id_prefixes`` are vocabulary ids or CURIE prefixes.
        """
        return self._enumerate(
            self.kg.roots, predicates, self._scope(filter_obsoletes, id_prefixes)
        )

    def leafs(
        self,
        predicates: list[PRED_CURIE] | None = None,
        ignore_owl_nothing: bool = True,
        filter_obsoletes: bool = True,
    ) -> Iterable[CURIE]:
        """Yield the concepts without a child. See :meth:`roots` for *predicates*."""
        return self._enumerate(self.kg.leaves, predicates, self._scope(filter_obsoletes))

    def singletons(
        self, predicates: list[PRED_CURIE] | None = None, filter_obsoletes: bool = True
    ) -> Iterable[CURIE]:
        """Yield the concepts without a parent or a child. See :meth:`roots` for *predicates*."""
        return self._enumerate(self.kg.singletons, predicates, self._scope(filter_obsoletes))

    def label(self, curie: CURIE, lang: str | None = None) -> str | None:
        """The label of a concept or relationship, or None when it does not exist."""
        return next(iter(self.labels([curie], allow_none=True, lang=lang)))[1]

    def labels(  # ty: ignore[invalid-method-override]
        self, curies: Iterable[CURIE], allow_none: bool = True, lang: str | None = None
    ) -> Iterable[tuple[CURIE, str | None]]:
        """
        Yield each CURIE with its label, looking up all concepts in one query.

        Returns None for a missing label, which oaklib allows although it annotates a string.

        The label is the concept name, or the relationship name. For another language than
        ``default_language`` it is the first synonym in that language, which OMOP does not
        rank, so concepts without one get None. Relationships are only named in the
        default language.
        """
        curies = list(curies)
        if lang is None or lang == self.default_language:
            names = {
                curie: predicate.name
                for curie in curies
                if (predicate := self.predicates_by_curie.get(curie))
            }
            names |= {
                curie: view.concept_name for curie, view in self._concept_views(curies).items()
            }
        else:
            names = self._translations(self._resolve_concepts(curies), lang)
        for curie in curies:
            if curie in names or allow_none:
                yield curie, names.get(curie)

    def _translations(self, concept_ids: dict[CURIE, int], lang: str) -> dict[CURIE, str]:
        """The first synonym of each concept in *lang*, by name."""
        language = OMOPLanguage.from_code(lang)
        if language is None:
            return {}
        by_concept: dict[int, str] = {}
        for synonym in self.kg.synonyms(tuple(concept_ids.values())):
            if synonym.language_concept_id == language.concept_id:
                by_concept.setdefault(synonym.concept_id, synonym.name)
        return {curie: by_concept[cid] for curie, cid in concept_ids.items() if cid in by_concept}

    def multilingual_labels(  # ty: ignore[invalid-method-override]
        self,
        curies: Iterable[CURIE],
        allow_none: bool = True,
        langs: list[str] | None = None,
    ) -> Iterable[tuple[CURIE, str | None, str | None]]:
        """
        Yield each concept name with no language tag, then its synonyms in other languages
        with theirs, restricted to *langs* when given.
        """
        curies = list(curies)
        ids = self._resolve_concepts(curies)
        views = self._concept_views(curies)
        synonyms = self.kg.synonyms(tuple(ids.values()))
        curie_by_id = {cid: curie for curie, cid in ids.items()}
        for curie in curies:
            view = views.get(curie)
            if view is None:
                if allow_none:
                    yield curie, None, None
                continue
            if langs is None or self.default_language in langs:
                yield curie, view.concept_name, None
        for synonym in synonyms:
            language = OMOPLanguage.from_concept_id(synonym.language_concept_id)
            if language is None or language.code == self.default_language:
                continue
            if synonym.concept_id in curie_by_id and (langs is None or language.code in langs):
                yield curie_by_id[synonym.concept_id], synonym.name, language.code

    def curies_by_label(self, label: str) -> list[CURIE]:
        """
        Retrieve CURIEs that match the exact label.
        """
        # Prefer exact concept_name matches
        cids = self.kg.concept_ids_by_label(label.strip())
        return [self.codec.concept_curie(cid) for cid in cids]

    def relationships(
        self,
        subjects: Iterable[CURIE] | None = None,
        predicates: Iterable[PRED_CURIE] | None = None,
        objects: Iterable[CURIE] | None = None,
        include_tbox: bool = True,
        include_abox: bool = True,
        include_entailed: bool = False,
        exclude_blank: bool = True,
        invert: bool = False,
    ) -> Iterator[tuple[CURIE, PRED_CURIE, CURIE]]:
        """
        Query relationships between concepts.

        Every OMOP relationship links two concepts, so it is a TBox statement and
        ``include_tbox=False`` yields nothing. There are no individuals and no blank
        nodes, so ``include_abox`` and ``exclude_blank`` have no effect.

        Parameters
        ----------
        subjects : Iterable[CURIE] | None
            Subject CURIEs to filter on.
        predicates : Iterable[PRED_CURIE] | None
            Predicate (relationship) IDs to filter on.
        objects : Iterable[CURIE] | None
            Object CURIEs to filter on.
        include_tbox : bool
            If False, yield nothing.
        include_abox : bool
            Ignored.
        include_entailed : bool
            If True, yield the relationships entailed by the strict ``Is a`` hierarchy:
            (S, R, E) for every asserted (S, R, D) where E is D or an ancestor of D.
            Relationships of ancestors are not inherited. Without subjects and objects
            this streams the entailed relationships of every concept.
        exclude_blank : bool
            Ignored.
        invert : bool
            If True, swaps subjects and objects in the query and result.

        Yields
        -------
        Tuple[CURIE, PRED_CURIE, CURIE]
            Triples (subject, predicate, object).

        """
        if not include_tbox:
            return

        subject_ids = self._concept_ids(subjects)
        object_ids = self._concept_ids(objects)
        predicate_ids = (
            self.codec.relationship_ids(predicates) if predicates else self.default_relationship_ids
        )
        if not predicate_ids:
            return

        edge_filter = EdgeFilter(predicate_ids=predicate_ids)

        if include_entailed:
            if subject_ids is None and object_ids is None:
                allowed = {self.codec.predicate_curie(p) for p in predicate_ids}
                for s, p, o in self._all_entailed_relationships():
                    if p in allowed:
                        yield (o, p, s) if invert else (s, p, o)
                return
            if invert:
                subject_ids, object_ids = object_ids, subject_ids
            entailed = (
                (o, p, s) if invert else (s, p, o)
                for s, p, o in self._entailed(subject_ids, object_ids, edge_filter)
            )
            relationships = tuple(entailed)
        else:
            relationships = tuple(
                self.kg.relationships(
                    subjects=subject_ids,
                    objects=object_ids,
                    edge_filter=edge_filter,
                    invert=invert,
                )
            )

        for s, p, o in relationships:
            yield (
                self.codec.concept_curie(s),
                self.codec.predicate_curie(p),
                self.codec.concept_curie(o),
            )

    def relationships_metadata(
        self, relationships: Iterable[tuple[CURIE, PRED_CURIE, CURIE]], **kwargs
    ) -> Iterator[tuple[tuple[CURIE, PRED_CURIE, CURIE], list[tuple[PRED_CURIE, str]]]]:
        """
        Yield each relationship with the validity dates and ``invalid_reason`` of its edge.

        Deprecated edges are included, so a relationship from :meth:`relationships` always
        finds its edge.
        """
        relationships = list(relationships)
        concept_ids = self._resolve_concepts({c for s, _, o in relationships for c in (s, o)})
        subject_ids = tuple({concept_ids[s] for s, _, _ in relationships if s in concept_ids})
        predicate_ids = self.codec.relationship_ids(p for _, p, _ in relationships)
        edges: dict[tuple[int | None, str | None, int | None], EdgeView] = {
            (edge.subject_id, edge.predicate_id, edge.object_id): edge
            for edge in self.kg.edges(
                subject_ids,
                direction="out",
                edge_filter=EdgeFilter(predicate_ids=predicate_ids, active_only=False),
            )
        }
        for relationship in relationships:
            s, p, o = relationship
            edge = edges.get(
                (concept_ids.get(s), self.codec.parse_predicate(p), concept_ids.get(o))
            )
            metadata: list[tuple[PRED_CURIE, str]] = []
            if edge is not None:
                metadata = [
                    ("omop:valid_start_date", f"{edge.valid_start_date}"),
                    ("omop:valid_end_date", f"{edge.valid_end_date}"),
                ]
                if edge.invalid_reason:
                    metadata.append(("omop:invalid_reason", edge.invalid_reason))
            yield relationship, metadata

    def _all_relationships(self) -> Iterator[tuple[CURIE, PRED_CURIE, CURIE]]:
        """Stream every active edge of the exposed relationships, without loading them all."""
        for s, p, o in self.kg.all_relationships(
            EdgeFilter(predicate_ids=self.default_relationship_ids)
        ):
            yield (
                self.codec.concept_curie(s),
                self.codec.predicate_curie(p),
                self.codec.concept_curie(o),
            )

    def _standing_in(self, concept_ids: Iterable[int]) -> dict[int, tuple[int, ...]]:
        """
        Map each concept to the concepts it stands in for: itself, and with ``standardise``
        also the standard concepts it maps to.
        """
        ids = tuple(dict.fromkeys(concept_ids))
        if not self.standardise:
            return {concept_id: (concept_id,) for concept_id in ids}
        return {
            concept_id: tuple(dict.fromkeys((concept_id, *targets)))
            for concept_id, targets in self.kg.standardise(ids).items()
        }

    def _entailed(
        self,
        subject_ids: tuple[int, ...] | None,
        object_ids: tuple[int, ...] | None,
        edge_filter: EdgeFilter,
    ) -> Iterator[tuple[int, str, int]]:
        """
        Entailed relationships of ids, where with ``standardise`` a concept also has those of
        its standard concepts. The subjects, or the objects when there are no subjects,
        are the ones that stand in, and are reported in place of their standard concepts.
        """
        if not self.standardise:
            yield from self.kg.entailed_relationships(
                subject_ids, object_ids, edge_filter, self.concept_filter
            )
            return
        by_subject = subject_ids is not None
        assert subject_ids is not None or object_ids is not None
        standing = self._standing_in(subject_ids if subject_ids is not None else object_ids or ())
        origins: dict[int, list[int]] = defaultdict(list)
        for origin, stand_ins in standing.items():
            for stand_in in stand_ins:
                origins[stand_in].append(origin)
        expanded = tuple(origins)
        seen: set[tuple[int, str, int]] = set()
        if by_subject:
            triples = self.kg.entailed_relationships(
                expanded, object_ids, edge_filter, self.concept_filter
            )
            reported = ((origin, p, o) for s, p, o in triples for origin in origins[s])
        else:
            triples = self.kg.entailed_relationships(
                None, expanded, edge_filter, self.concept_filter
            )
            reported = ((s, p, origin) for s, p, o in triples for origin in origins[o])
        for triple in reported:
            if triple not in seen:
                seen.add(triple)
                yield triple

    def _all_entailed_relationships(self) -> Iterator[tuple[CURIE, PRED_CURIE, CURIE]]:
        """Stream the entailed relationships of every concept, one batch of subjects at a time."""
        edge_filter = EdgeFilter(predicate_ids=self.default_relationship_ids)
        for batch in batched(self.kg.entities(self.concept_filter), self.kg.LOOKUP_BATCH_SIZE):
            for s, p, o in self.kg.entailed_relationships(
                batch, None, edge_filter, self.concept_filter
            ):
                yield (
                    self.codec.concept_curie(s),
                    self.codec.predicate_curie(p),
                    self.codec.concept_curie(o),
                )

    @property
    def edge_index(self):
        raise OMOPUnsupportedError("BasicOntologyInterface.edge_index")

    @property
    def entailed_edge_index(self):
        raise OMOPUnsupportedError("BasicOntologyInterface.entailed_edge_index")

    def hierarchical_parents(self, curie: CURIE, isa_only: bool = False) -> list[CURIE]:
        """
        Get hierarchical parents.

        With ``isa_only`` the parents are the targets of ``Is a`` edges. Otherwise they are
        the level 1 ancestors of ``concept_ancestor``, OMOP's own ancestry.
        """
        cid = self.codec.resolve_concept(curie)
        if cid is None:
            return []
        if isa_only:
            parents = [
                edge.object_id
                for edge in self.kg.edges(
                    cid,
                    direction="out",
                    edge_filter=EdgeFilter(predicate_ids=frozenset({IS_A_RELATIONSHIP_ID})),
                )
            ]
        else:
            parents = self.kg.parents(cid)
        return [self.codec.concept_curie(p) for p in parents]

    def entity_alias_map(self, curie: CURIE) -> ALIAS_MAP:
        """
        Get a map of alias types to alias values.
        """
        view = self._concept_views([curie]).get(curie)
        if view is None:
            return {}

        m = defaultdict(list)

        m[LABEL_PREDICATE].append(view.concept_name)

        for s in self.kg.synonyms_for_concept(view.concept_id):
            m[HAS_EXACT_SYNONYM].append(s)

        return m

    def entity_metadata_map(self, curie: CURIE) -> METADATA_MAP:
        """
        Retrieve the metadata of a concept or relationship.

        A relationship has its name and the CURIE of its reverse relationship as
        ``owl:inverseOf``. A deprecated concept is ``owl:deprecated`` and names its
        replacements as ``IAO:0100001``. Empty for an unknown CURIE.
        """
        predicate = self.predicates_by_curie.get(curie)
        if predicate is not None:
            metadata: METADATA_MAP = {"id": [curie], LABEL_PREDICATE: [predicate.name]}
            if predicate.reverse_id:
                metadata[INVERSE_OF] = [self.codec.predicate_curie(predicate.reverse_id)]
            return metadata

        cv = self._concept_views([curie]).get(curie)
        if cv is None:
            return {}

        m: METADATA_MAP = defaultdict(list)

        # required
        m["id"].append(curie)
        m[LABEL_PREDICATE].append(cv.concept_name)

        # core OMOP metadata
        m["omop:concept_id"].append(f"{cv.concept_id}")
        m["omop:domain"].append(cv.domain_id)
        m["omop:vocabulary"].append(cv.vocabulary_id)
        m["omop:concept_class"].append(cv.concept_class_id)
        m["omop:concept_code"].append(cv.concept_code)

        if cv.standard_concept:
            m["omop:standard_concept"].append("S")

        # validity
        m["omop:valid_start_date"].append(f"{cv.valid_start_date}")
        m["omop:valid_end_date"].append(f"{cv.valid_end_date}")

        if cv.invalid_reason:
            m["omop:invalid_reason"].append(cv.invalid_reason)

        if not cv.is_active:
            m[DEPRECATED_PREDICATE].append("true")
            for edge in self.kg.edges(
                cv.concept_id,
                direction="out",
                edge_filter=EdgeFilter(predicate_ids=frozenset({REPLACED_BY_RELATIONSHIP_ID})),
            ):
                m[TERM_REPLACED_BY].append(self.codec.concept_curie(edge.object_id))

        # vocabulary-qualified identifier (important!)
        m[HAS_DBXREF].append(self.codec.code_curie(cv.vocabulary_id, cv.concept_code))

        return dict(m)
