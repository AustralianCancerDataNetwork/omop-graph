"""The OBO graph interface over the OMOP knowledge graph."""

from collections import defaultdict
from collections.abc import Iterable, Iterator

from oaklib.datamodels.obograph import (
    BasicPropertyValue,
    Edge,
    Graph,
    Meta,
    Node,
    NodeTypeEnum,
    SynonymPropertyValue,
    XrefPropertyValue,
)
from oaklib.datamodels.vocabulary import (
    DEPRECATED_PREDICATE,
    HAS_DBXREF,
    HAS_EXACT_SYNONYM,
    IS_A,
    TERM_REPLACED_BY,
)
from oaklib.interfaces.obograph_interface import GraphTraversalMethod, OboGraphInterface
from oaklib.types import CURIE, PRED_CURIE

from omop_graph.extensions.omop_alchemy import HierarchyDirection
from omop_graph.graph.edges import EdgeFilter
from omop_graph.oaklib_interface.basic_ontology import OMOPBasicOntologyInterface
from omop_graph.oaklib_interface.codec import OMOPLanguage


class OMOPOboGraphInterface(OMOPBasicOntologyInterface, OboGraphInterface):
    """
    Mixin implementing the OBO graph interface over an OMOP knowledge graph.

    Walks follow the predicates given, and the strict ``Is a`` hierarchy
    (``rdfs:subClassOf``) when none are given, since a walk over every OMOP relationship
    reaches most of the vocabulary within a few hops. Only concepts within the adapter's
    ``concept_filter`` are reported by ``ancestors`` and ``descendants``. The graphs
    returned by ``ancestor_graph`` and ``descendant_graph`` keep every concept on the walk,
    so they stay connected through concepts outside the scope.

    Notes
    -----
    ``nodes``, ``edges`` and ``as_obograph`` enumerate the whole graph and are unbounded on
    a full vocabulary.
    """

    def _traversal_predicates(self, predicates: Iterable[PRED_CURIE] | None) -> list[PRED_CURIE]:
        """The predicates to walk: those given, or ``rdfs:subClassOf``."""
        return list(predicates) if predicates else [IS_A]

    def node(
        self,
        curie: CURIE,
        strict: bool = False,
        include_metadata: bool = False,
        expand_curies=False,
    ) -> Node | None:
        """
        Look up a concept or relationship as an OBO graph node.

        Parameters
        ----------
        curie : CURIE
            Concept or predicate CURIE.
        strict : bool
            If True, raise when the entity does not exist.
        include_metadata : bool
            If True, add the synonyms, cross-reference, deprecation and replacements of a
            concept.
        expand_curies : bool
            If True, the node id is a URI.

        Returns
        -------
        Node | None
            The node, or None for an unknown CURIE.

        Raises
        ------
        ValueError
            If ``strict`` is True and the entity does not exist.
        """
        node = self._nodes([curie], include_metadata, expand_curies).get(curie)
        if node is None and strict:
            raise ValueError(f"Node {curie} does not exist")
        return node

    def nodes(self, expand_curies=False) -> Iterator[Node]:
        """
        Yield a node, with its metadata, for every concept in scope.

        The concepts are read in batches, so the number of queries does not grow with the
        number of concepts.
        """
        for batch in self.kg.batches(self.entities()):
            yield from self._nodes(batch, True, expand_curies).values()

    def _nodes(
        self, curies: Iterable[CURIE], include_metadata: bool, expand_curies: bool
    ) -> dict[CURIE, Node]:
        """The node of each CURIE that names a concept or relationship, read together."""
        curies = list(dict.fromkeys(curies))
        metadata = self._metadata_maps(curies) if include_metadata else {}
        synonyms: dict[CURIE, list[dict | SynonymPropertyValue]] = defaultdict(list)
        if include_metadata:
            for curie, spv in self.synonym_property_values(curies):
                synonyms[curie].append(spv)
        views = self._concept_views(
            curie for curie in curies if curie not in self.predicates_by_curie
        )
        nodes: dict[CURIE, Node] = {}
        for curie in curies:
            node_id = (self.curie_to_uri(curie) or curie) if expand_curies else curie
            predicate = self.predicates_by_curie.get(curie)
            if predicate is not None:
                nodes[curie] = Node(id=node_id, lbl=predicate.name, type=NodeTypeEnum.PROPERTY.text)
            elif curie in views:
                nodes[curie] = Node(
                    id=node_id, lbl=views[curie].concept_name, type=NodeTypeEnum.CLASS.text
                )
                if include_metadata:
                    nodes[curie].meta = Meta(
                        synonyms=synonyms[curie],
                        xrefs=[
                            XrefPropertyValue(val=xref)
                            for xref in metadata[curie].get(HAS_DBXREF, [])
                        ],
                        basicPropertyValues=[
                            BasicPropertyValue(pred=TERM_REPLACED_BY, val=replacement)
                            for replacement in metadata[curie].get(TERM_REPLACED_BY, [])
                        ],
                        deprecated=True if DEPRECATED_PREDICATE in metadata[curie] else None,
                    )
        return nodes

    def synonym_property_values(
        self, subject: CURIE | Iterable[CURIE]
    ) -> Iterator[tuple[CURIE, SynonymPropertyValue]]:
        """
        Yield each synonym of the concepts as an exact synonym, tagged with its language.

        OMOP does not scope synonyms, so all are exact. The language is the ISO 639-1 code,
        or None for a language concept that has none.
        """
        curies = [subject] if isinstance(subject, str) else list(subject)
        ids = self._resolve_concepts(curies)
        curie_by_id = {concept_id: curie for curie, concept_id in ids.items()}
        for synonym in self.kg.synonyms(tuple(ids.values())):
            language = OMOPLanguage.from_concept_id(synonym.language_concept_id)
            yield (
                curie_by_id[synonym.concept_id],
                SynonymPropertyValue(
                    pred=HAS_EXACT_SYNONYM.removeprefix("oio:"),
                    val=synonym.name,
                    lang=language.code if language else None,
                ),
            )

    def _reached(
        self,
        start_curies: CURIE | list[CURIE],
        predicates: Iterable[PRED_CURIE] | None,
        direction: HierarchyDirection,
        scoped: bool,
    ) -> dict[CURIE, list[CURIE]]:
        """Map each start CURIE to the CURIEs reached from it, in one query per batch."""
        starts = [start_curies] if isinstance(start_curies, str) else list(start_curies)
        ids = self._resolve_concepts(starts)
        edge_filter = EdgeFilter(
            predicate_ids=self.codec.relationship_ids(self._traversal_predicates(predicates))
        )
        standing = self._standing_in(ids.values())
        reached = self.kg.closure(
            {stand_in for stand_ins in standing.values() for stand_in in stand_ins},
            direction,
            edge_filter,
            self.concept_filter if scoped else None,
        )
        return {
            curie: [
                self.codec.concept_curie(other)
                for other in sorted(
                    set().union(
                        *(reached[stand_in] for stand_in in standing[concept_id]),
                        set(standing[concept_id]) - {concept_id},
                    )
                )
            ]
            for curie, concept_id in ids.items()
        }

    def _walk(
        self,
        start_curies: CURIE | list[CURIE],
        predicates: list[PRED_CURIE] | None,
        reflexive: bool,
        method: GraphTraversalMethod | None,
        direction: HierarchyDirection,
    ) -> Iterator[CURIE]:
        """Yield the concepts reached from the starts, with the starts when reflexive."""
        starts = [start_curies] if isinstance(start_curies, str) else list(start_curies)
        predicates = self._traversal_predicates(predicates)
        if method == GraphTraversalMethod.ENTAILMENT:
            walker = (
                super().ancestors if direction is HierarchyDirection.UP else super().descendants
            )
            yield from walker(starts, predicates, reflexive, method)
            return
        found = dict.fromkeys(starts) if reflexive else {}
        for reached in self._reached(starts, predicates, direction, scoped=True).values():
            found.update(dict.fromkeys(reached))
        if not reflexive:
            for curie in starts:
                found.pop(curie, None)
        yield from found

    def ancestors(
        self,
        start_curies: CURIE | list[CURIE],
        predicates: list[PRED_CURIE] | None = None,
        reflexive: bool = True,
        method: GraphTraversalMethod | None = None,
    ) -> Iterable[CURIE]:
        """
        Yield the concepts reached by following the predicates upwards, in one query.

        Parameters
        ----------
        start_curies : CURIE | list[CURIE]
            Concepts to start from.
        predicates : list[PRED_CURIE] | None
            The predicates to follow. Defaults to ``rdfs:subClassOf``.
        reflexive : bool
            If True, include the starting concepts.
        method : GraphTraversalMethod | None
            ``ENTAILMENT`` reads the entailed relationships instead of walking.

        Yields
        ------
        CURIE
            Ancestor concepts within the adapter's scope.
        """
        yield from self._walk(start_curies, predicates, reflexive, method, HierarchyDirection.UP)

    def descendants(
        self,
        start_curies: CURIE | list[CURIE],
        predicates: list[PRED_CURIE] | None = None,
        reflexive: bool = True,
        method: GraphTraversalMethod | None = None,
    ) -> Iterable[CURIE]:
        """
        Yield the concepts that reach the start concepts by the predicates, in one query.

        The parameters are those of :meth:`ancestors`, in the other direction.
        """
        yield from self._walk(start_curies, predicates, reflexive, method, HierarchyDirection.DOWN)

    def _graph(self, triples: Iterable[tuple[CURIE, PRED_CURIE, CURIE]], nodes=()) -> Graph:
        """Build a graph, naming all its concepts and predicates with one lookup per kind."""
        edges: list[Edge | dict] = [Edge(sub=s, pred=p, obj=o) for s, p, o in triples]
        concepts = dict.fromkeys(
            [
                *nodes,
                *(e.sub for e in edges if isinstance(e, Edge)),
                *(e.obj for e in edges if isinstance(e, Edge)),
            ]
        )
        predicates = dict.fromkeys(e.pred for e in edges if isinstance(e, Edge))
        labels = dict(self.labels([*concepts, *predicates]))
        graph_nodes: list[Node | dict] = [
            Node(id=curie, lbl=labels.get(curie), type=NodeTypeEnum.CLASS.text)
            for curie in concepts
        ]
        graph_nodes += [
            Node(id=p, lbl=labels.get(p), type=NodeTypeEnum.PROPERTY.text) for p in predicates
        ]
        return Graph(id="omop", nodes=graph_nodes, edges=edges)

    def _walked(
        self,
        start_curies: CURIE | list[CURIE],
        predicates: list[PRED_CURIE] | None,
        direction: HierarchyDirection,
    ) -> tuple[list[tuple[CURIE, PRED_CURIE, CURIE]], dict[CURIE, None]]:
        """The edges between the starts and the concepts reached from them, and those concepts."""
        starts = [start_curies] if isinstance(start_curies, str) else list(start_curies)
        predicates = self._traversal_predicates(predicates)
        members = dict.fromkeys(starts)
        for reached in self._reached(starts, predicates, direction, scoped=False).values():
            members.update(dict.fromkeys(reached))
        triples = []
        for batch in self.kg.batches(members):
            triples.extend(
                triple
                for triple in self.relationships(subjects=batch, predicates=predicates)
                if triple[2] in members
            )
        return triples, members

    def _walked_graph(
        self,
        start_curies: CURIE | list[CURIE],
        predicates: list[PRED_CURIE] | None,
        direction: HierarchyDirection,
    ) -> Graph:
        """The starts, the concepts reached from them, and the edges between those."""
        triples, members = self._walked(start_curies, predicates, direction)
        return self._graph(triples, nodes=members)

    def ancestor_graph(
        self,
        start_curies: CURIE | list[CURIE],
        predicates: list[PRED_CURIE] | None = None,
        method: GraphTraversalMethod | None = None,
        **kwargs,
    ) -> Graph:
        """
        Build the graph of the start concepts, their ancestors and the edges between them.

        The predicates default to ``rdfs:subClassOf``, and the graph includes ancestors
        outside the adapter's scope so it stays connected.
        """
        if method == GraphTraversalMethod.ENTAILMENT:
            return super().ancestor_graph(
                start_curies, self._traversal_predicates(predicates), method, **kwargs
            )
        return self._walked_graph(start_curies, predicates, HierarchyDirection.UP)

    def descendant_graph(
        self,
        start_curies: CURIE | list[CURIE],
        predicates: list[PRED_CURIE] | None = None,
    ) -> Graph:
        """Build the graph of the start concepts, their descendants and the edges between them."""
        return self._walked_graph(start_curies, predicates, HierarchyDirection.DOWN)

    def walk_up_relationship_graph(
        self,
        start_curies: CURIE | list[CURIE],
        predicates: list[PRED_CURIE] | None = None,
    ) -> Iterable[tuple[CURIE, PRED_CURIE, CURIE]]:
        """Yield the edges of the ancestor graph."""
        triples, _ = self._walked(start_curies, predicates, HierarchyDirection.UP)
        yield from triples
