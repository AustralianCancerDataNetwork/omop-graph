"""Shared base of the oaklib interface mixins."""

from collections.abc import Iterable, Iterator

from oaklib.types import CURIE, PRED_CURIE

from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.oaklib_interface.codec import OMOPCodec


def as_list(value: str | Iterable[str]) -> list[str]:
    """A CURIE as a list of it, and any other iterable of CURIEs as a list of its items."""
    return [value] if isinstance(value, str) else list(value)


class OMOPBaseInterface:
    """
    Base class for OMOP interfaces holding the Knowledge Graph and the CURIE codec.

    Parameters
    ----------
    kg : KnowledgeGraph
        The OMOP Knowledge Graph instance.
    """

    def __init__(self, kg: KnowledgeGraph, **kwargs):
        super().__init__(**kwargs)
        self.kg = kg
        self.codec = OMOPCodec(kg)

    def triple_curies(
        self, triples: Iterable[tuple[int, str, int]]
    ) -> Iterator[tuple[CURIE, PRED_CURIE, CURIE]]:
        """Convert (subject id, relationship id, object id) triples to CURIE triples."""
        for subject_id, relationship_id, object_id in triples:
            yield (
                self.codec.concept_curie(subject_id),
                self.codec.predicate_curie(relationship_id),
                self.codec.concept_curie(object_id),
            )

    def prefix_map(self) -> dict[str, str]:
        """Return the prefix map for OMOP entities."""
        return self.codec.prefix_map()
