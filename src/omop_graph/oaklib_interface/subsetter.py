"""The subsetter interface over the OMOP knowledge graph."""

from collections.abc import Iterator

from oaklib.interfaces.basic_ontology_interface import BasicOntologyInterface
from oaklib.interfaces.subsetter_interface import SubsetStrategy, SubsetterInterface
from oaklib.types import CURIE, PRED_CURIE

from omop_graph.graph.edges import EdgeFilter
from omop_graph.oaklib_interface.basic_ontology import OMOPBasicOntologyInterface
from omop_graph.oaklib_interface.unsupported import OMOPUnsupportedError


class OMOPSubsetterInterface(OMOPBasicOntologyInterface, SubsetterInterface):
    """
    Mixin that connects a set of concepts through their entailed relationships.

    Gap filling follows :meth:`omop_graph.graph.kg.KnowledgeGraph.gap_filled_relationships`.
    An ontology subset cannot be extracted, since it would need an in-memory ontology.
    """

    def extract_subset_ontology(
        self, seed_curies: list[CURIE], strategy: SubsetStrategy | None = None
    ) -> BasicOntologyInterface:
        """Not supported, the subset would need an in-memory ontology."""
        raise OMOPUnsupportedError("SubsetterInterface.extract_subset_ontology")

    def gap_fill_relationships(
        self, seed_curies: list[CURIE], predicates: list[PRED_CURIE] | None = None
    ) -> Iterator[tuple[CURIE, PRED_CURIE, CURIE]]:
        """
        Yield the non-redundant relationships that connect the seed concepts.

        Parameters
        ----------
        seed_curies : list[CURIE]
            The concepts to connect.
        predicates : list[PRED_CURIE] | None
            Only these predicates, every exposed relationship when None, and nothing when
            empty.
        """
        predicate_ids = (
            self.kg.default_relationship_ids
            if predicates is None
            else self.codec.relationship_ids(predicates)
        )
        ids = tuple(dict.fromkeys(self._resolve_concepts(seed_curies).values()))
        yield from self.triple_curies(
            self.kg.gap_filled_relationships(ids, EdgeFilter(predicate_ids=predicate_ids))
        )
