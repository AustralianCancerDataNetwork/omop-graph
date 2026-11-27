"""Translation between OMOP identifiers and oaklib CURIEs."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from collections.abc import Iterable
from enum import StrEnum
from functools import cached_property
from typing import ClassVar
from urllib.parse import quote, unquote

from oaklib.datamodels.vocabulary import IS_A
from oaklib.interfaces.basic_ontology_interface import get_default_prefix_map
from oaklib.types import CURIE, PRED_CURIE

from omop_graph.graph import KnowledgeGraph
from omop_graph.graph.edges import IS_A_RELATIONSHIP_ID

logger = logging.getLogger(__name__)

OMOP_PREFIX = "OMOP"
OMOP_REL_PREFIX = "OMOP_REL"


class OMOPMetadataKey(StrEnum):
    """The metadata keys the adapter adds to the entities and relationships it describes."""

    CONCEPT_ID = "omop:concept_id"
    DOMAIN = "omop:domain"
    VOCABULARY = "omop:vocabulary"
    CONCEPT_CLASS = "omop:concept_class"
    CONCEPT_CODE = "omop:concept_code"
    STANDARD_CONCEPT = "omop:standard_concept"
    VALID_START_DATE = "omop:valid_start_date"
    VALID_END_DATE = "omop:valid_end_date"
    INVALID_REASON = "omop:invalid_reason"
    VOCABULARY_ID = "omop:vocabulary_id"
    VOCABULARY_REFERENCE = "omop:vocabulary_reference"


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

    @cached_property
    def _vocabulary_curies(self) -> dict[str, CURIE]:
        """The CURIE of the vocabulary concept of every vocabulary, by ``vocabulary_id``."""
        return {
            view.vocabulary_id: self.concept_curie(view.vocabulary_concept_id)
            for view in self.kg.vocabularies()
        }

    @staticmethod
    def vocabulary_prefix(vocabulary_id: str) -> str:
        """
        Convert a ``vocabulary_id`` to a valid CURIE prefix.

        Characters that are not word characters become ``_``, so ``Nebraska Lexicon`` is
        ``Nebraska_Lexicon``.
        """
        return re.sub(r"\W", "_", vocabulary_id)

    def vocabulary_curie(self, vocabulary_id: str) -> CURIE:
        """
        Convert a ``vocabulary_id`` to the CURIE of its vocabulary concept, as ``ontologies`` lists it.

        Raises
        ------
        KeyError
            If there is no such vocabulary.
        """
        return self._vocabulary_curies[vocabulary_id]

    def vocabulary_id(self, prefix: str) -> str:
        """
        Convert a CURIE prefix, or the CURIE of a vocabulary concept, back to its ``vocabulary_id``.

        A prefix that is already a ``vocabulary_id`` is returned as it is. It is returned
        as it is too when it names no vocabulary, or several, in which case a warning
        lists them. Nothing then resolves under it.
        """
        if prefix in self._vocabulary_ids:
            return prefix
        for vocabulary_id, curie in self._vocabulary_curies.items():
            if curie == prefix:
                return vocabulary_id
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

        These are the oaklib defaults, ``OMOP`` and ``OMOP_REL``. A vocabulary has no URI,
        since its name does not say whether it means what a registry prefix of the same
        name means.
        """
        return {
            **get_default_prefix_map(),
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
        TypeError
            If ``concept_id`` is not an integer.
        """
        if not isinstance(concept_id, int) or isinstance(concept_id, bool):
            raise TypeError(f"Expected concept_id to be an integer, got {concept_id!r}")
        return ":".join((OMOP_PREFIX, str(concept_id)))

    MAX_CONCEPT_ID: ClassVar[int] = 2**31 - 1
    """The largest concept id the OMOP ``integer`` column holds."""

    @classmethod
    def _is_concept_id(cls, local: str) -> bool:
        """
        Whether local is the canonical decimal form of a concept id.

        Leading zeros are not canonical, so each concept has one ``OMOP`` spelling, and a
        number beyond the column is no concept, as the database would reject it.
        """
        return (
            local.isascii()
            and local.isdigit()
            and local == str(int(local[:12]))
            and int(local) <= cls.MAX_CONCEPT_ID
        )

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

    def concept_candidates(self, curie: CURIE) -> tuple[int, ...]:
        """
        The OMOP concept ids a CURIE can name.

        An ``OMOP`` CURIE carries the id itself. Any other prefix is read as
        ``vocab:code``, where the prefix is a ``vocabulary_id`` or its CURIE prefix and
        the code is percent-encoded. OMOP does not keep a code unique, so the valid
        concepts are preferred over deprecated ones, and several remain when they share
        the code.

        Parameters
        ----------
        curie : CURIE
            The CURIE to look up.

        Returns
        -------
        tuple[int, ...]
            The concept ids, empty when the CURIE names no concept. An ``OMOP`` CURIE is
            only parsed, so the adapter checks it against the concept table.
        """
        prefix, separator, local = curie.partition(":")
        if not separator or self.is_predicate_curie(curie):
            return ()
        if self.is_omop_prefix(prefix):
            return (int(local),) if self._is_concept_id(local) else ()
        matches = self.kg.concepts_by_code(self.vocabulary_id(prefix), unquote(local))
        candidates = [concept_id for concept_id, is_valid in matches.items() if is_valid]
        return tuple(candidates or matches)

    def resolve_concept(self, curie: CURIE) -> int | None:
        """
        Resolve a CURIE to an OMOP concept id.

        When several concepts share the code of a ``vocab:code`` CURIE, a warning lists
        them and the CURIE is unresolved, see :meth:`concept_candidates`.

        Parameters
        ----------
        curie : CURIE
            The CURIE to resolve.

        Returns
        -------
        int | None
            The OMOP concept id, or None when the CURIE names no concept or is ambiguous.
            An ``OMOP`` CURIE is only parsed, so the adapter checks it against the concept table.
        """
        candidates = self.concept_candidates(curie)
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
        nothing. It is named once, in a warning when it has the ``OMOP_REL`` prefix, since
        it is then probably a typo, and at debug level otherwise, since oaklib callers pass
        predicates of other ontologies.
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
                level = logging.WARNING if relationship_id is not None else logging.DEBUG
                logger.log(
                    level,
                    "The predicate %s names no OMOP relationship and matches nothing.",
                    predicate,
                )
        return frozenset(relationship_ids)
