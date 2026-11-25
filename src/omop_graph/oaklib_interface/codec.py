"""Translation between OMOP identifiers and oaklib CURIEs."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from collections.abc import Iterable
from enum import Enum
from functools import cached_property
from urllib.parse import quote, unquote

from oaklib.datamodels.vocabulary import IS_A
from oaklib.interfaces.basic_ontology_interface import get_default_prefix_map
from oaklib.types import CURIE, PRED_CURIE
from prefixmaps import load_context

from omop_graph.graph import KnowledgeGraph
from omop_graph.graph.edges import IS_A_RELATIONSHIP_ID

logger = logging.getLogger(__name__)

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
        self._ignored_predicates: set[PRED_CURIE] = set()

    @cached_property
    def _classified_relationship_ids(self) -> frozenset[str]:
        """The ids of the relationships the knowledge graph has a classification for."""
        return frozenset(predicate.relationship_id for predicate in self.kg.predicates())

    @cached_property
    def _vocabulary_ids(self) -> frozenset[str]:
        """The ``vocabulary_id`` of every vocabulary, read on first use and then kept."""
        return frozenset(view.vocabulary_id for view in self.kg.vocabularies())

    @cached_property
    def _vocabulary_ids_by_prefix(self) -> dict[str, tuple[str, ...]]:
        """The ``vocabulary_id`` values that become each CURIE prefix."""
        by_prefix: dict[str, list[str]] = defaultdict(list)
        for vocabulary_id in sorted(self._vocabulary_ids):
            by_prefix[self.vocabulary_prefix(vocabulary_id)].append(vocabulary_id)
        return {prefix: tuple(ids) for prefix, ids in by_prefix.items()}

    @staticmethod
    def vocabulary_prefix(vocabulary_id: str) -> str:
        """
        Convert a ``vocabulary_id`` to a valid CURIE prefix.

        Characters that are not word characters become ``_``, so ``Nebraska Lexicon`` is
        ``Nebraska_Lexicon``.
        """
        return re.sub(r"\W", "_", vocabulary_id)

    def vocabulary_id(self, prefix: str) -> str:
        """
        Convert a CURIE prefix back to its ``vocabulary_id``.

        A prefix that is already a ``vocabulary_id`` is returned as it is. It is returned
        as it is too when it names no vocabulary, or several, in which case a warning
        lists them. Nothing then resolves under it.
        """
        if prefix in self._vocabulary_ids:
            return prefix
        ids = self._vocabulary_ids_by_prefix.get(prefix, ())
        if len(ids) > 1:
            logger.warning(
                "The prefix %r stands for several vocabularies (%s). Use a vocabulary id.",
                prefix,
                ", ".join(ids),
            )
            return prefix
        return ids[0] if ids else prefix

    def code_curie(self, vocabulary_id: str, concept_code: str) -> CURIE:
        """
        Convert a vocabulary and concept code to a CURIE, for example ``SNOMED:254637007``.

        The code is percent-encoded, so ``[iU]/L`` in ``UCUM`` gives ``UCUM:%5BiU%5D%2FL``.
        """
        return ":".join((self.vocabulary_prefix(vocabulary_id), quote(concept_code, safe="")))

    @cached_property
    def _prefix_map(self) -> dict[str, str]:
        """
        The prefixes of the CURIEs the adapter emits and the URI each stands for.

        These are the oaklib defaults, then each vocabulary whose prefix is a canonical
        Bioregistry prefix, then ``OMOP`` and ``OMOP_REL``. A vocabulary outside the
        Bioregistry has no URI.
        """
        registry = {
            prefix.lower(): namespace
            for prefix, namespace in load_context("bioregistry").as_dict().items()
        }
        vocabularies = {
            prefix: registry[prefix.lower()]
            for prefix in self._vocabulary_ids_by_prefix
            if prefix.lower() in registry
        }
        unmapped = sorted(set(self._vocabulary_ids_by_prefix) - vocabularies.keys())
        logger.debug("No Bioregistry URI for the vocabulary prefixes %s.", ", ".join(unmapped))
        return {
            **get_default_prefix_map(),
            **vocabularies,
            OMOP_PREFIX: "https://athena.ohdsi.org/search-terms/terms/",
            OMOP_REL_PREFIX: "https://athena.ohdsi.org/relationship/",
        }

    def prefix_map(self) -> dict[str, str]:
        """
        Return the prefix map for the CURIEs of OMOP entities.

        Returns
        -------
        dict[str, str]
            Prefixes mapped to URI bases.
        """
        return dict(self._prefix_map)

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

    @staticmethod
    def is_omop_prefix(prefix: str) -> bool:
        """Return True for the ``OMOP`` prefix, whose local part is the concept id."""
        return prefix == OMOP_PREFIX

    def resolve_reference(self, reference: str) -> int | None:
        """
        Resolve a bare concept id or a CURIE to an OMOP concept id.

        Parameters
        ----------
        reference : str
            A concept id such as ``"12345"``, or any CURIE accepted by :meth:`resolve_concept`.

        Returns
        -------
        int | None
            The OMOP concept id, or None when the reference names no concept.
        """
        if ":" not in reference:
            reference = self.concept_curie(int(reference)) if reference.isdecimal() else reference
        return self.resolve_concept(reference)

    def is_predicate_curie(self, curie: CURIE) -> bool:
        """Return True for a predicate CURIE: ``rdfs:subClassOf`` or an ``OMOP_REL`` CURIE."""
        return curie == IS_A or curie.partition(":")[0] == OMOP_REL_PREFIX

    def resolve_concept(self, curie: CURIE) -> int | None:
        """
        Resolve a CURIE to an OMOP concept id.

        An ``OMOP`` CURIE carries the id itself. Any other prefix is read as
        ``vocab:code``, where the prefix is a ``vocabulary_id`` or its CURIE prefix and
        the code is percent-encoded. OMOP does not keep a code unique, so a valid concept
        is preferred over deprecated ones. When several concepts still share the code, a
        warning lists them and the CURIE is unresolved.

        Parameters
        ----------
        curie : CURIE
            The CURIE to resolve.

        Returns
        -------
        int | None
            The OMOP concept id, or None when the CURIE names no concept or is ambiguous.
            An ``OMOP`` CURIE is not checked against the concept table.
        """
        prefix, separator, local = curie.partition(":")
        if not separator or self.is_predicate_curie(curie):
            return None
        if self.is_omop_prefix(prefix):
            return int(local) if local.isascii() and local.isdigit() else None
        matches = self.kg.concepts_by_code(self.vocabulary_id(prefix), unquote(local))
        candidates = [concept_id for concept_id, is_valid in matches.items() if is_valid]
        candidates = candidates or list(matches)
        if not candidates:
            logger.debug("%s names no concept.", curie)
        if len(candidates) > 1:
            logger.warning(
                "%s is ambiguous. Use one of %s.",
                curie,
                ", ".join(self.concept_curie(concept_id) for concept_id in candidates),
            )
        return candidates[0] if len(candidates) == 1 else None

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

    def parse_predicate(self, pred: PRED_CURIE) -> str | None:
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
        str | None
            The relationship id, or None when the CURIE is neither ``rdfs:subClassOf``
            nor an ``OMOP_REL`` CURIE, so it names no OMOP relationship.
        """
        if pred == IS_A:
            return IS_A_RELATIONSHIP_ID
        prefix, _, local = pred.partition(":")
        return unquote(local) if prefix == OMOP_REL_PREFIX else None

    def relationship_ids(self, predicates: Iterable[PRED_CURIE]) -> frozenset[str]:
        """
        The relationship ids of the predicate CURIEs.

        A predicate that is not an OMOP one, or names no classified relationship, selects
        nothing. An info message names it, once for each predicate.
        """
        relationship_ids: set[str] = set()
        for predicate in predicates:
            relationship_id = self.parse_predicate(predicate)
            if relationship_id is not None:
                relationship_ids.add(relationship_id)
            if (
                relationship_id not in self._classified_relationship_ids
                and predicate not in self._ignored_predicates
            ):
                self._ignored_predicates.add(predicate)
                logger.warning(
                    "The predicate %s names no OMOP relationship and matches nothing.", predicate
                )
        return frozenset(relationship_ids)
