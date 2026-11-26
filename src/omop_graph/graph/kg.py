"""
OMOP-backed knowledge graph.

This module provides the `KnowledgeGraph` class, the primary interface to the OMOP
Common Data Model database. It holds construction and the relationship classification,
and takes its methods from the capability mixins in `omop_graph.graph.capabilities`.

Capabilities
------------
* **Concepts, search, enumeration:** concept views, lookup by name or code, roots and leaves.
* **Predicates and edges:** relationship classification, relationships and edges.
* **Hierarchy and entailment:** parents, children, closure, and relationships entailed through them.
* **Mappings, vocabularies, embeddings:** mappings between vocabularies, vocabulary and synonym
  lookups, and the optional embedding store.
"""

import logging
from collections.abc import Callable

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

# Local Application Imports
from ..extensions.omop_alchemy import (
    RelationshipMappingElement,
    load_relationship_mapping,
)
from .base import GraphBackend
from .capabilities.concepts import ConceptsMixin
from .capabilities.edges import EdgesMixin
from .capabilities.embeddings import EmbeddingsMixin, KnowledgeGraphEmbeddingConfiguration
from .capabilities.entailment import EntailmentMixin
from .capabilities.enumeration import EnumerationMixin
from .capabilities.hierarchy import HierarchyMixin
from .capabilities.mappings import MappingsMixin
from .capabilities.predicates import PredicatesMixin
from .capabilities.search import SearchMixin
from .capabilities.similarity import SimilarityMixin
from .capabilities.vocabulary import VocabularyMixin

logger = logging.getLogger(__name__)


class KnowledgeGraph(
    EmbeddingsMixin,
    SimilarityMixin,
    SearchMixin,
    VocabularyMixin,
    MappingsMixin,
    EntailmentMixin,
    HierarchyMixin,
    EnumerationMixin,
    EdgesMixin,
    PredicatesMixin,
    ConceptsMixin,
    GraphBackend,
):
    """
    The main entry point for interacting with the OMOP Graph.

    This class wraps a SQLAlchemy session and provides high-level methods
    to query concepts, relationships, and metadata.

    Parameters
    ----------
    cdm_sessions : Callable[[], Session]
        Opens sessions on the OMOP CDM, e.g. from
        ``omop_alchemy.cross_database.cdm_sessionmaker``, which sends each
        table to the database hosting it. The relationship classification
        is read once at construction and applied to vocabulary rows in
        Python, so no query joins the two databases.
    emb_config : KnowledgeGraphEmbeddingConfiguration, optional
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

        # Embedding-specific private args
        self._emb_config = emb_config
        self._emb = None

        self._descendant_counts = {}
        self._scope_sizes = {}
