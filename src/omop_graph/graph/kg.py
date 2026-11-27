"""
OMOP-backed knowledge graph.

This module provides the ``KnowledgeGraph`` class, the primary interface to the OMOP
Common Data Model database. It holds construction and the relationship classification,
and takes its methods from the capability mixins in ``omop_graph.graph.capabilities``.

The capabilities are concepts, predicates and edges, enumeration of roots, leaves and
singletons, the ``Is a`` hierarchy and the relationships entailed through it, mappings,
vocabularies and synonyms, search, structural similarity, statistics and the optional
embedding store.
"""

from collections.abc import Callable

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

# Local Application Imports
from ..extensions.relationship_classification import (
    RelationshipMappingElement,
    load_relationship_mapping,
)
from .capabilities.concepts import ConceptsMixin
from .capabilities.edges import EdgesMixin
from .capabilities.embeddings import EmbeddingsMixin, KnowledgeGraphEmbeddingConfiguration
from .capabilities.entailment import EntailmentMixin
from .capabilities.enumeration import EnumerationMixin
from .capabilities.hierarchy import HierarchyMixin
from .capabilities.mappings import MappingsMixin
from .capabilities.predicates import PredicatesMixin
from .capabilities.search import SearchMixin
from .capabilities.statistics import StatisticsMixin
from .capabilities.structural_similarity import StructuralSimilarityMixin
from .capabilities.vocabulary import VocabularyMixin


class KnowledgeGraph(
    EmbeddingsMixin,
    StructuralSimilarityMixin,
    SearchMixin,
    StatisticsMixin,
    VocabularyMixin,
    MappingsMixin,
    EntailmentMixin,
    HierarchyMixin,
    EnumerationMixin,
    EdgesMixin,
    PredicatesMixin,
    ConceptsMixin,
):
    """
    The OMOP concepts and relationships as a graph.

    The methods come from the capability mixins in
    ``omop_graph.graph.capabilities``: concepts, predicates, edges,
    enumeration, hierarchy, entailment, mappings, vocabulary, search,
    structural similarity, statistics and embeddings.

    Parameters
    ----------
    cdm_sessions : Callable[[], Session]
        Opens sessions on the OMOP CDM, e.g. from
        ``omop_alchemy.cross_database.cdm_sessionmaker``, which sends each
        table to the database hosting it. The relationship classification
        is read once at construction and applied to vocabulary rows in
        Python, so no query joins the two databases.
    emb_config : KnowledgeGraphEmbeddingConfiguration, optional
        Where the embeddings are stored and which model made them. Without it, the
        methods that need embeddings raise EmbeddingsNotConfiguredError.

    Raises
    ------
    TypeError
        If cdm_sessions is an Engine or is not callable.
    RuntimeError
        If the relationship classification is missing or empty.
    """

    def __init__(
        self,
        cdm_sessions: Callable[[], Session],
        *,
        emb_config: KnowledgeGraphEmbeddingConfiguration | None = None,
    ):
        if isinstance(cdm_sessions, Engine) or not callable(cdm_sessions):
            raise TypeError(
                "cdm_sessions must be a callable session factory, not an Engine; "
                "pass cdm_session_factory() or a sessionmaker."
            )
        self.session_factory = cdm_sessions
        with self.session_factory() as session:
            self._relationship_mapping: dict[str, RelationshipMappingElement] = (
                load_relationship_mapping(session)
            )

        if not self._relationship_mapping:
            raise RuntimeError(
                "RelationshipMapping table is empty. "
                "Run `omop-graph relationship-classification` to populate it."
            )

        self._emb_config = emb_config
