"""The relation graph interface over the OMOP knowledge graph."""

from collections.abc import Iterable

from oaklib.interfaces.relation_graph_interface import RelationGraphInterface
from oaklib.types import CURIE, PRED_CURIE

from omop_graph.oaklib_interface.basic_ontology import OMOPBasicOntologyInterface


class OMOPRelationGraphInterface(OMOPBasicOntologyInterface, RelationGraphInterface):
    """
    Mixin implementing entailed relationships over the strict ``Is a`` hierarchy.

    See Also
    --------
    omop_graph.graph.kg.KnowledgeGraph.entailed_relationships : The entailment rule.

    Notes
    -----
    oaklib has largely subsumed the relation graph interface into the basic ontology
    interface. The entailment is implemented there, as ``relationships`` with
    ``include_entailed``, and the methods here wrap it. It applies to the relationships
    that generalise, and the others are returned as stored. Only concepts within the
    adapter's ``concept_filter`` are reported as entailed objects, and for incoming
    relationships as entailed subjects, though the walk passes through the others.
    """

    def entailed_outgoing_relationships(
        self, curie: CURIE, predicates: list[PRED_CURIE] | None = None
    ) -> Iterable[tuple[PRED_CURIE, CURIE]]:
        """
        Yield the predicate and object of every relationship entailed for a concept.

        Parameters
        ----------
        curie : CURIE
            The subject concept.
        predicates : list[PRED_CURIE] | None
            Restrict to these predicates. Defaults to the exposed relationships.

        Yields
        ------
        tuple[PRED_CURIE, CURIE]
            Predicate and object.
        """
        for _, predicate, object_ in self.relationships(
            subjects=[curie], predicates=predicates, include_entailed=True
        ):
            yield predicate, object_

    def entailed_incoming_relationships(
        self, curie: CURIE, predicates: list[PRED_CURIE] | None = None
    ) -> Iterable[tuple[PRED_CURIE, CURIE]]:
        """
        Yield the predicate and subject of every relationship entailed into a concept.

        These are the relationships asserted into the concept or into any of its
        ``Is a`` descendants. A concept near the top of a hierarchy has very many.

        Parameters
        ----------
        curie : CURIE
            The object concept.
        predicates : list[PRED_CURIE] | None
            Restrict to these predicates. Defaults to the exposed relationships.

        Yields
        ------
        tuple[PRED_CURIE, CURIE]
            Predicate and subject.
        """
        for subject, predicate, _ in self.relationships(
            objects=[curie], predicates=predicates, include_entailed=True
        ):
            yield predicate, subject

    def entailed_relationships_between(self, subject: CURIE, object: CURIE) -> Iterable[PRED_CURIE]:
        """
        Yield the predicates that connect a subject to an object, asserted or entailed.

        Parameters
        ----------
        subject : CURIE
            The subject concept.
        object : CURIE
            The object concept.

        Yields
        ------
        PRED_CURIE
            Predicate.
        """
        for _, predicate, _ in self.relationships(
            subjects=[subject], objects=[object], include_entailed=True
        ):
            yield predicate
