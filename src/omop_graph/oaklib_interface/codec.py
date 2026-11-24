"""Translation between OMOP identifiers and oaklib CURIEs."""

from __future__ import annotations

from enum import Enum
from urllib.parse import quote, unquote

from oaklib.datamodels.vocabulary import IS_A
from oaklib.types import CURIE, PRED_CURIE

from omop_graph.graph import KnowledgeGraph
from omop_graph.graph.edges import IS_A_RELATIONSHIP_ID

OMOP_PREFIX = "OMOP"
OMOP_REL_PREFIX = "OMOP_REL"
REPLACED_BY_RELATIONSHIP_ID = "Concept replaced by"


class OMOPLanguage(Enum):
    """
    The language concepts used by ``concept_synonym``, with their ISO 639-1 codes.

    The concepts are SNOMED language concepts. A language concept outside this list has
    no code and its synonyms are not exposed as translations.
    """

    ENGLISH = (4180186, "en")
    SPANISH = (4182511, "es")
    KOREAN = (4175771, "ko")
    CHINESE = (4182948, "zh")
    FRENCH = (4180190, "fr")
    GERMAN = (4182504, "de")
    JAPANESE = (4181524, "ja")
    PORTUGUESE = (4181536, "pt")
    DUTCH = (4182503, "nl")
    HEBREW = (4180047, "he")
    ITALIAN = (4182507, "it")
    SWEDISH = (4175777, "sv")
    DANISH = (4180183, "da")
    ARABIC = (4181374, "ar")

    def __init__(self, concept_id: int, code: str):
        self.concept_id = concept_id
        self.code = code

    @classmethod
    def from_concept_id(cls, concept_id: int) -> OMOPLanguage | None:
        """Return the language with this concept id, or None when it has no code."""
        return next((language for language in cls if language.concept_id == concept_id), None)

    @classmethod
    def from_code(cls, code: str) -> OMOPLanguage | None:
        """Return the language with this ISO 639-1 code, or None when it is unknown."""
        return next((language for language in cls if language.code == code), None)


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

    def is_predicate_curie(self, curie: CURIE) -> bool:
        """Return True for a predicate CURIE: ``rdfs:subClassOf`` or an ``OMOP_REL`` CURIE."""
        return curie == IS_A or curie.partition(":")[0] == OMOP_REL_PREFIX

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
