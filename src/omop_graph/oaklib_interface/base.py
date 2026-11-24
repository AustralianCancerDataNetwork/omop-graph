"""Shared base of the oaklib interface mixins."""

import logging

from omop_graph.graph import KnowledgeGraph
from omop_graph.oaklib_interface.codec import OMOPCodec

logger = logging.getLogger(__name__)


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

    def prefix_map(self) -> dict[str, str]:
        """Return the prefix map for OMOP entities."""
        return self.codec.prefix_map()
