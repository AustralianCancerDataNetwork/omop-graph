"""Translation between OMOP identifiers and oaklib CURIEs."""

from __future__ import annotations

from urllib.parse import quote, unquote

from oaklib.datamodels.vocabulary import IS_A
from oaklib.types import CURIE, PRED_CURIE

from omop_graph.graph import KnowledgeGraph

OMOP_PREFIX = "OMOP"
OMOP_REL_PREFIX = "OMOP_REL"
IS_A_RELATIONSHIP_ID = "Is a"


class OMOPCodec:
    """
    Converts OMOP concept and relationship ids to CURIEs and back.

    Parameters
    ----------
    kg : KnowledgeGraph
        The Knowledge Graph used to resolve ``vocab:code`` CURIEs.
    """

    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg

    def prefix_map(self) -> dict[str, str]:
        """
        Return the prefix map for OMOP entities.

        Returns
        -------
        dict[str, str]
            Prefixes mapped to URI bases.
        """
        return {
            OMOP_PREFIX: "https://athena.ohdsi.org/concept/",
            OMOP_REL_PREFIX: "https://athena.ohdsi.org/relationship/",
        }

    def concept_curie(self, concept_id: int) -> CURIE:
        """
        Convert an OMOP concept id to a CURIE.

        Parameters
        ----------
        concept_id : int
            The OMOP concept id.

        Returns
        -------
        CURIE
            The CURIE, for example ``OMOP:12345``.

        Raises
        ------
        AssertionError
            If ``concept_id`` is not an integer.
        """
        assert isinstance(concept_id, int), "Expected concept_id to be an integer"
        return ":".join((OMOP_PREFIX, str(concept_id)))

    def parse_concept(self, curie: CURIE) -> int:
        """
        Parse a CURIE into an OMOP concept id.

        An ``OMOP`` CURIE carries the id itself. Any other prefix is read as
        ``vocab:code`` and resolved through the Knowledge Graph.

        Parameters
        ----------
        curie : CURIE
            The CURIE to parse.

        Returns
        -------
        int
            The OMOP concept id.
        """
        prefix, _, local = curie.partition(":")
        if prefix == OMOP_PREFIX:
            return int(local)
        return self.kg.concept_id_by_code(prefix, local)

    def predicate_curie(self, relationship_id: str | int) -> PRED_CURIE:
        """
        Convert a relationship id to a predicate CURIE.

        ``Is a`` is ``rdfs:subClassOf``, the predicate oaklib recognises as the hierarchy.
        Every other relationship is ``OMOP_REL`` followed by the percent-encoded id.

        Parameters
        ----------
        relationship_id : str | int
            The OMOP relationship id.

        Returns
        -------
        PRED_CURIE
            The CURIE, for example ``OMOP_REL:Has%20dir%20proc%20site``.
        """
        if relationship_id == IS_A_RELATIONSHIP_ID:
            return IS_A
        return ":".join((OMOP_REL_PREFIX, quote(str(relationship_id), safe="")))

    def parse_predicate(self, pred: PRED_CURIE) -> str:
        """
        Parse a predicate CURIE into an OMOP relationship id.

        Accepts ``rdfs:subClassOf`` for ``Is a`` and ``OMOP_REL`` with a percent-encoded id,
        which also covers ``Is a``.

        Parameters
        ----------
        pred : PRED_CURIE
            The predicate CURIE.

        Returns
        -------
        str
            The relationship id.

        Raises
        ------
        ValueError
            If the CURIE is neither ``rdfs:subClassOf`` nor an ``OMOP_REL`` CURIE.
        """
        if pred == IS_A:
            return IS_A_RELATIONSHIP_ID
        prefix, _, local = pred.partition(":")
        if prefix != OMOP_REL_PREFIX:
            raise ValueError(f"Unsupported predicate CURIE: {pred}")
        return unquote(local)
