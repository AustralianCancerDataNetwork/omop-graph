"""Search over the OMOP knowledge graph."""

import logging
from collections.abc import Iterable

from oaklib.datamodels.search import (
    SearchConfiguration,
    SearchProperty,
    SearchTermSyntax,
)
from oaklib.interfaces import SearchInterface
from oaklib.types import CURIE
from sqlalchemy.exc import MultipleResultsFound, NoResultFound

from omop_graph.graph import KnowledgeGraph
from omop_graph.graph.nodes import LabelMatchKind
from omop_graph.oaklib_interface.base import OMOPBaseInterface

logger = logging.getLogger(__name__)

SUPPORTED_PROPERTIES = {
    SearchProperty.LABEL.text,
    SearchProperty.ALIAS.text,
    SearchProperty.IDENTIFIER.text,
    SearchProperty.MAPPED_IDENTIFIER.text,
}


def _normalise_properties(config: SearchConfiguration) -> list[str]:
    """
    Extract property text values from a search configuration.

    Parameters
    ----------
    config : SearchConfiguration
        The search configuration object.

    Returns
    -------
    list[str]
        A list of property strings.
    """
    return [getattr(p, "text", None) or str(p) for p in (config.properties or [])]


class OMOPSearchInterface(OMOPBaseInterface, SearchInterface):
    """
    Mixin providing search capabilities over an OMOP knowledge graph.
    """

    def __init__(self, *args, kg: KnowledgeGraph, **kwargs):
        super().__init__(*args, kg=kg, **kwargs)

    def basic_search(
        self,
        search_term: str,
        config: SearchConfiguration | None = None,
    ) -> Iterable[CURIE]:
        """
        Perform a basic search over the knowledge graph.

        Parameters
        ----------
        search_term : str
            The term to search for.
        config : SearchConfiguration, optional
            Configuration for search behavior (fuzzy, properties to search).

        Yields
        -------
        CURIE
            The CURIEs of matching concepts.

        Raises
        ------
        ValueError
            If Regex search is requested (not supported).
        """
        if config is None:
            config = SearchConfiguration()

        props = _normalise_properties(config)

        if SearchProperty.ANYTHING.text in props:
            props = list(SUPPORTED_PROPERTIES)

        props = [p for p in props if p in SUPPORTED_PROPERTIES]

        seen: set[int] = set()

        if config.syntax == SearchTermSyntax.REGULAR_EXPRESSION:
            raise ValueError("REGULAR_EXPRESSION search is not supported for OMOP")

        if config.is_partial or config.syntax == SearchTermSyntax.STARTS_WITH:
            match_kind = LabelMatchKind.PARTIAL
        else:
            match_kind = LabelMatchKind.EXACT

        if SearchProperty.LABEL.text in props:
            matches = self.kg.concept_lookup(
                search_term,
                match_kind=match_kind,
                synonym=False,
            )
            for lm in matches:
                cid = lm.matched_concept_id
                if cid not in seen:
                    seen.add(cid)
                    yield self.codec.concept_curie(cid)

        if SearchProperty.ALIAS.text in props:
            matches = self.kg.concept_lookup(
                search_term,
                match_kind=match_kind,
                synonym=True,
            )
            for lm in matches:
                cid = lm.matched_concept_id
                if cid not in seen:
                    seen.add(cid)
                    yield self.codec.concept_curie(cid)

        if (
            SearchProperty.IDENTIFIER.text in props
            or SearchProperty.MAPPED_IDENTIFIER.text in props
        ) and ":" in search_term:
            vocab, code = search_term.split(":", 1)
            try:
                cid = self.kg.concept_id_by_code(vocab, code)
                if cid not in seen:
                    seen.add(cid)
                    yield self.codec.concept_curie(cid)
            except NoResultFound:
                pass
            except MultipleResultsFound:
                logger.warning(
                    "Skipping ambiguous identifier %r: several concepts match", search_term
                )
