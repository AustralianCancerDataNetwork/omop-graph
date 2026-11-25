"""Search over the OMOP knowledge graph."""

from collections.abc import Iterator
from typing import ClassVar

from linkml_runtime.linkml_model.meta import PermissibleValue
from oaklib.datamodels.search import (
    SearchConfiguration,
    SearchProperty,
    SearchTermSyntax,
)
from oaklib.interfaces import SearchInterface
from oaklib.types import CURIE
from omop_alchemy.cdm.query import ConceptFilter

from omop_graph.graph import KnowledgeGraph
from omop_graph.graph.nodes import LabelMatchKind
from omop_graph.oaklib_interface.base import OMOPBaseInterface

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

    UNSUPPORTED_SYNTAXES: ClassVar[tuple[PermissibleValue, ...]] = (
        SearchTermSyntax.REGULAR_EXPRESSION,
        SearchTermSyntax.SQL,
        SearchTermSyntax.LUCENE,
    )
    """The search term syntaxes that raise ``ValueError``."""

    def __init__(self, *args, kg: KnowledgeGraph, **kwargs):
        super().__init__(*args, kg=kg, **kwargs)

    def basic_search(
        self,
        search_term: str,
        config: SearchConfiguration | None = None,
    ) -> Iterator[CURIE]:
        """
        Search concepts by label, alias and identifier, best matches first.

        Labels are always searched, aliases with the ``ALIAS`` property and identifiers with
        ``IDENTIFIER`` or ``MAPPED_IDENTIFIER``. An identifier is an ``OMOP`` CURIE or a
        ``vocabulary:code``. Matches are ordered by kind, then labels before aliases, and
        within a kind by standard, active, shorter and lower id. The kinds are:

        - an identifier match
        - an exact label, matched case-insensitively
        - a label starting with the term, for the ``STARTS_WITH`` syntax or ``is_partial``
        - a label containing the term, for ``is_partial``
        - a full-text match, for ``is_fuzzy``, when the database has the index

        Deprecated concepts are left out unless ``include_obsoletes_in_results`` is set.

        Parameters
        ----------
        search_term : str
            The term to search for.
        config : SearchConfiguration, optional
            Which properties to search and how to match. ``limit`` caps the results.

        Yields
        ------
        CURIE
            The CURIEs of matching concepts, each once.

        Raises
        ------
        ValueError
            If the syntax is regular expression, SQL or Lucene, which are not supported.
        """
        config = config or SearchConfiguration()
        for syntax in self.UNSUPPORTED_SYNTAXES:
            if config.syntax == SearchTermSyntax(syntax.text):
                raise ValueError(f"{syntax.text} search is not supported for OMOP")

        props = set(_normalise_properties(config))
        if SearchProperty.ANYTHING.text in props:
            props = SUPPORTED_PROPERTIES
        by_identifier = bool(
            props & {SearchProperty.IDENTIFIER.text, SearchProperty.MAPPED_IDENTIFIER.text}
        )
        starts_with = config.syntax == SearchTermSyntax(SearchTermSyntax.STARTS_WITH.text)
        kinds = [LabelMatchKind.EXACT]
        if starts_with or config.is_partial:
            kinds.append(LabelMatchKind.STARTS_WITH)
        if config.is_partial and not starts_with:
            kinds.append(LabelMatchKind.PARTIAL)
        if config.is_fuzzy:
            kinds.append(LabelMatchKind.FTS)

        include_obsoletes = bool(config.include_obsoletes_in_results)
        constraint = ConceptFilter(require_standard=False, require_active=not include_obsoletes)
        found: dict[int, None] = {}

        def full() -> bool:
            return config.limit is not None and len(found) >= config.limit

        if by_identifier:
            concept_id = self.codec.resolve_concept(search_term)
            views = self.kg.concept_views((concept_id,)) if concept_id is not None else ()
            for view in views:
                if view.is_active or include_obsoletes:
                    found[view.concept_id] = None
                    yield self.codec.concept_curie(view.concept_id)

        for kind in kinds:
            for synonym in (False, True):
                if full():
                    return
                if synonym and SearchProperty.ALIAS.text not in props:
                    continue
                matches = self.kg.concept_lookup(
                    search_term,
                    match_kind=kind,
                    synonym=synonym,
                    search_constraint=constraint,
                    limit=None if config.limit is None else config.limit + len(found),
                )
                for match in matches:
                    if match.matched_concept_id not in found:
                        found[match.matched_concept_id] = None
                        yield self.codec.concept_curie(match.matched_concept_id)
                        if full():
                            return
