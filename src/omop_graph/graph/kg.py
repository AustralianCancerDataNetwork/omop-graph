"""
OMOP-backed graph facade.

This module provides the `KnowledgeGraph` class, which acts as the primary
interface (facade) to the OMOP Common Data Model database.

Responsibilities
----------------
* **SQLAlchemy Access:** Manages the database session and executes queries.
* **Caching:** Implements LRU caching for high-frequency lookups (concepts, predicates).
* **Predicate Semantics:** Resolves relationship IDs to `Predicate` objects and Kinds.
* **Edge/Node Retrieval:** Provides methods to traverse the graph (parents, children, edges).
"""

from __future__ import annotations

import functools
import logging
import re
from datetime import date
from collections import defaultdict
from typing import Dict, Optional, Tuple, Literal, Generator, TYPE_CHECKING
from dataclasses import dataclass

from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker
from omop_alchemy.backends import FullTextError
from oa_configurator import ResolvedModel

if TYPE_CHECKING:
    from omop_emb import (
        EmbeddingWriterInterface,
        EmbeddingReaderInterface,
    )

# Local Application Imports
from ..extensions.emb import (
    MissingExtensionError,
    EmbeddingBackendType,
    EmbeddingProviderType,
    EmbeddingMetricType,
)
from ..extensions.omop_alchemy import (
    PredicateKind,
    RelationshipMappingElement,
    load_relationship_mapping,
)
from .base import GraphBackend
from .constraints import SearchConstraintConcept
from .edges import EdgeView, Predicate
from .nodes import (
    AncestorMatch,
    ConceptView,
    LabelMatch,
    LabelMatchKind,
)
from .queries import (
    q_all_predicates_with_ancestry,
    q_concept_domain_ids,
    q_concept_id_by_code,
    q_concept_name_fulltext,
    q_concept_name_ilike,
    q_concept_name_match,
    q_concept_num_ancestors,
    q_concept_potential_ancestor,
    q_concept_potential_ancestors_batch,
    q_concept_synonym_filtered,
    q_concept_view,
    q_concept_views,
    q_concept_vocabulary_ids,
    q_edges,
    q_leaves,
    q_parents,
    q_children,
    q_predicate_name,
    q_predicate_row_with_ancestry,
    q_roots,
    q_singletons,
    q_entities,
    q_relationships,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KnowledgeGraphEmbeddingConfiguration:
    """
    Configuration for embedding-based operations in the knowledge graph.

    Two independent things are being configured here, and they have different costs:

    1. Knowing *which* model (``model_name`` + ``provider_type``, plain strings) 
    is needed for any embedding use at all, read or write, with no exceptions
        * The embedding registry is keyed by ``(model_name, provider_type)``, 
            so even a purely read-only lookup against already-computed embeddings needs both 
            to canonicalize the name and find the right row.

    2. Knowing *how to call* that model (a fully resolved ``oa_configurator.ResolvedModel``, 
    carrying a real provider *connection* [`base_url``/``api_key``], resolved from an actual 
    ``[providers.*]`` config entry):
        - only needed when the KG must build a live model backend, i.e. when ``write=True``. 
        
    A read-only consumer (e.g. one that only ever searches against a pre-computed
    ``query_embedding`` passed in externally) still needs to know the
    provider's plain string key, but that key never has to correspond to a
    resolved, connectable ``[providers.*]`` entry: a bare string is enough,
    since the connection details are never used.

    Parameters
    ----------
    metric_type : EmbeddingMetricType
        The similarity/distance metric to use for embedding comparisons (e.g., cosine, euclidean).
        This is required to ensure that the correct type of index is used in the backend and that
        similarity computations are consistent.
    backend_type : EmbeddingBackendType
        The embedding backend name (e.g., 'faiss', 'pinecone') or type to use.
    model_name : str, optional
        The model name to use for the embedding interface (e.g., 'text-embedding-3-small:0.6b').
        Ignored when ``resolved_model`` is set (derived from it instead via
        ``effective_model_name``); required otherwise.
    provider_type : EmbeddingProviderType, optional
        The omop-llm provider key (e.g. 'ollama'). Ignored when ``resolved_model`` is set
        (derived from it instead via ``effective_provider_type``); required otherwise.
    resolved_model : oa_configurator.ResolvedModel, optional
        A model resolved via ``oa_configurator.Resolver.from_active_config().resolve_model(name)``
        (or an explicit ``Resolver(stack)`` for a non-default stack), carrying real provider
        connection details. Required when ``write=True``: the KG builds its own embedding
        model backend from it (``omop_llm.build_model_backend_from_resolved``) to generate
        and persist embeddings. Optional otherwise. Note this is oa-configurator's own
        resolution step, not something ``omop_llm`` wraps: ``omop_llm`` only takes an
        already-resolved model and builds a live backend from it
        (``build_model_backend_from_resolved``), which wouldn't help the read-only case here
        anyway -- a read-only KG needs the bare ``ResolvedModel``, never a constructed backend.
    write : bool
        If True, the KG holds a write-capable interface (``resolved_model`` required) that can
        generate and persist embeddings. If False (default), the KG only holds a read-only
        interface over already-computed embeddings.
    compute_missing_embeddings : bool
        If True, the system will compute embeddings on-the-fly for any concept that is not yet present
        in the embedding store, and persist those embeddings back to the DB before running similarity scoring.
        **Requires ``write=True``** (validated at construction, see ``__post_init__``): on-the-fly
        computation needs a write-capable interface. Defaults to ``False`` so that unexpected
        writes do not occur when only a read-only configuration is given.
    """

    metric_type: EmbeddingMetricType
    backend_type: Optional[EmbeddingBackendType] = None
    model_name: Optional[str] = None
    provider_type: Optional[EmbeddingProviderType] = None
    resolved_model: Optional[ResolvedModel] = None
    write: bool = False
    compute_missing_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.write and self.resolved_model is None:
            raise ValueError(
                "write=True requires resolved_model "
                "(a write-capable interface needs real provider credentials to build a model backend)."
            )
        if self.compute_missing_embeddings and not self.write:
            raise ValueError(
                "compute_missing_embeddings=True requires write=True "
                "(on-the-fly embedding computation needs a write-capable interface)."
            )
        if self.resolved_model is None and (self.model_name is None or self.provider_type is None):
            raise ValueError(
                "Provide either resolved_model, or both model_name and provider_type."
            )

    @property
    def effective_model_name(self) -> str:
        """The model name to use, from ``resolved_model`` if set, else ``model_name``."""
        if self.resolved_model is not None:
            return self.resolved_model.model
        assert self.model_name is not None  # guaranteed by __post_init__
        return self.model_name

    @property
    def effective_provider_type(self) -> str:
        """The provider key to use, from ``resolved_model`` if set, else ``provider_type``."""
        if self.resolved_model is not None:
            return self.resolved_model.provider.provider
        assert self.provider_type is not None  # guaranteed by __post_init__
        return self.provider_type


class KnowledgeGraph(GraphBackend):
    """
    The main entry point for interacting with the OMOP Graph.

    This class wraps a SQLAlchemy session and provides high-level methods
    to query concepts, relationships, and metadata.

    Parameters
    ----------
    cdm_engine : Engine
        The SQLAlchemy engine for the OMOP CDM database.
    """

    def __init__(
        self,
        cdm_engine: Engine,
        emb_config: Optional[KnowledgeGraphEmbeddingConfiguration] = None,
    ):
        self.cdm_engine = cdm_engine
        self.session_factory = sessionmaker(bind=self.cdm_engine, future=True)

        try:
            with self.session_factory() as session:
                self._relationship_mapping: dict[str, RelationshipMappingElement] = (
                    load_relationship_mapping(session)
                )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load relationship mapping. "
                "The KnowledgeGraph requires relationship classification data. "
                "Run `omop-graph relationship-classification` to populate it."
            ) from exc

        if not self._relationship_mapping:
            raise RuntimeError(
                "RelationshipMapping table is empty. "
                "Run `omop-graph relationship-classification` to populate it."
            )

        # Embedding-specific private args
        self._emb_config = emb_config
        self._emb = None

    @property
    def emb(self) -> "EmbeddingWriterInterface | EmbeddingReaderInterface":
        """Namespace for all embedding operations.

        Returns an ``EmbeddingWriterInterface`` when ``self._emb_config.write`` is True,
        otherwise an ``EmbeddingReaderInterface`` (read-only).

        The interface/reader is created lazily on first access. Backend resolution follows
        ``omop_emb`` rules: explicit ``backend_type`` first, then ``OMOP_EMB_BACKEND``.
        """
        if self._emb is not None:
            return self._emb

        try:
            from omop_emb.interface import (
                EmbeddingWriterInterface,
                EmbeddingReaderInterface,
            )
            from omop_emb.config import OmopEmbConfig
            from omop_emb.backends.base_backend import resolve_backend

            if self._emb_config is None:
                raise ValueError(
                    "Embedding configuration is not set. Please provide an EmbeddingConfiguration when initializing the KnowledgeGraph to use embedding features."
                )

            cfg = OmopEmbConfig.get_config()
            backend_type = self._emb_config.backend_type or cfg.backend
            faiss_cache_dir = cfg.faiss_cache_dir

            backend = resolve_backend(backend_type)

            if self._emb_config.write:
                # Write-capable interface: the KG builds its own embedding model backend.
                # __post_init__ already guarantees resolved_model is set when write=True.
                self._emb = EmbeddingWriterInterface(
                    backend=backend,
                    metric_type=self._emb_config.metric_type,
                    resolved_model=self._emb_config.resolved_model,  # ty: ignore[invalid-argument-type]
                    omop_cdm_engine=self.cdm_engine,
                )
            else:
                # Read-only interface: only ever needs model identity, never live credentials.
                self._emb = EmbeddingReaderInterface(
                    model=self._emb_config.effective_model_name,
                    backend=backend,
                    metric_type=self._emb_config.metric_type,
                    omop_cdm_engine=self.cdm_engine,
                    provider_type=self._emb_config.effective_provider_type,
                    faiss_cache_dir=faiss_cache_dir,
                )
            logger.debug(
                "Constructed %s embedding interface for model %r (write=%s).",
                "write-capable" if self._emb_config.write else "read-only",
                self._emb_config.effective_model_name,
                self._emb_config.write,
            )
            return self._emb

        except ModuleNotFoundError as e:
            if e.name and e.name.startswith("omop_emb"):
                logger.info(
                    "Embedding functionality is not available because the optional 'omop-emb' package is not installed."
                )
                raise MissingExtensionError() from e
            raise
        except ImportError as e:
            logger.info(
                "Embedding functionality failed to initialize due to an import error in the optional embedding stack."
            )
            raise e

    @property
    def embedding_configuration(self) -> Optional[KnowledgeGraphEmbeddingConfiguration]:
        """Returns the current embedding configuration, if set."""
        return self._emb_config

    @property
    def compute_missing_embeddings(self) -> bool:
        """Indicates whether on-the-fly computation of missing concept embeddings is enabled."""
        return (
            self._emb_config.compute_missing_embeddings if self._emb_config else False
        )

    def concept_view(self, concept_id: int) -> ConceptView:
        """
        Retrieve a single concept view by ID.

        Parameters
        ----------
        concept_id : int
            The OMOP Concept ID.

        Returns
        -------
        ConceptView
            The immutable view of the concept.
        """
        with self.session_factory() as session:
            row = session.execute(q_concept_view(concept_id)).one()
        return ConceptView.from_row(row)

    def concept_views(
        self, concept_ids: tuple[int, ...], sort: bool = True
    ) -> tuple[ConceptView, ...]:
        """
        Retrieve multiple concept views in a batch.

        Parameters
        ----------
        concept_ids : tuple[int, ...]
            A tuple of OMOP Concept IDs.

        Returns
        -------
        tuple[ConceptView, ...]
            A tuple of concept views.
        """
        with self.session_factory() as session:
            concept_views = tuple(
                ConceptView.from_row(row)
                for row in session.execute(q_concept_views(concept_ids, sort=sort))
            )
        return concept_views

    def concept_id_by_code(self, vocabulary_id: str, concept_code: str) -> int:
        """
        Look up a Concept ID using the vocabulary ID and concept code.

        Parameters
        ----------
        vocabulary_id : str
            The vocabulary ID (e.g., 'SNOMED', 'RxNorm').
        concept_code : str
            The source code within that vocabulary.

        Returns
        -------
        int
            The resolved OMOP Concept ID.
        """
        with self.session_factory() as session:
            concept_id = int(
                session.execute(
                    q_concept_id_by_code(vocabulary_id, concept_code)
                ).scalar_one()
            )
        return concept_id

    def concept_lookup(
        self,
        query_term: str,
        match_kind: LabelMatchKind,
        synonym: bool = False,
        search_constraint: Optional[SearchConstraintConcept] = None,
        sort: bool = True,
    ) -> tuple[LabelMatch, ...]:
        """
        Resolve a query to concept_id(s).

        Parameters
        ----------
        query_term : str
            The term to search for.
        match_kind : LabelMatchKind
            The kind of match to perform (exact, fulltext, partial).
        synonym : bool
            If True, searches in Concept_Synonym instead of Concept.
        search_constraint : SearchConstraintConcept, optional
            Additional filters for domain/vocabulary.

        """
        input_query_term = self._normalise_query_term(query_term)
        if not input_query_term:
            return ()

        if match_kind == LabelMatchKind.EXACT:
            fn = q_concept_name_match
        elif match_kind == LabelMatchKind.PARTIAL:
            fn = q_concept_name_ilike
        elif match_kind == LabelMatchKind.FTS:
            fn = functools.partial(q_concept_name_fulltext, engine=self.cdm_engine)
        else:
            raise ValueError(f"Unsupported search mode: {match_kind}")
        try:
            cn = fn(
                input_query_term,
                search_constraint=search_constraint,
                synonym=synonym,
                sort=sort,
            )
        except FullTextError as e:
            if match_kind == LabelMatchKind.FTS:
                logger.info(e)
                return ()
            raise

        with self.session_factory() as session:
            matches = tuple(
                LabelMatch(
                    input_query=input_query_term,
                    matched_concept_label=name,
                    matched_concept_id=int(cid),
                    match_kind=match_kind,
                    is_standard=is_standard,
                    is_active=is_active,
                    synonym=synonym,
                )
                for cid, name, is_standard, is_active in session.execute(cn)
            )
        return matches

    def concept_ids_by_label(self, label: str) -> Tuple[int, ...]:
        """
        Find concept IDs that match the label exactly (case-insensitive).
        """
        label = self._normalise_query_term(label)
        with self.session_factory() as session:
            rows = session.execute(q_concept_name_match(label)).scalars()
        return tuple(rows)

    def predicate(self, relationship_id: str) -> Predicate:
        """
        Retrieve a Predicate object by its relationship ID.

        Parameters
        ----------
        relationship_id : str
            The OMOP relationship ID (e.g., 'maps to').

        Returns
        -------
        Predicate
            The predicate definition.
        """
        with self.session_factory() as session:
            row = session.execute(q_predicate_row_with_ancestry(relationship_id)).one()
        return Predicate(
            relationship_id=row.relationship_id,
            name=row.relationship_name,
            reverse_id=row.reverse_relationship_id,
            is_hierarchical=bool(row.is_hierarchical),
            anc_up=bool(int(row.anc_up)),
            anc_down=bool(int(row.anc_down)),
            predicate_kind=PredicateKind(row.predicate_kind),
            predicate_subkind=row.predicate_subkind,
        )

    def predicate_name(self, relationship_id: str) -> str:
        """
        Retrieve the human-readable name of a relationship.
        """
        # TODO: Not really necessary. The "ID" is mostly human-readable anyways.
        with self.session_factory() as session:
            predicate_name = session.execute(
                q_predicate_name(relationship_id)
            ).scalar_one()
        return predicate_name

    def predicate_kind(self, relationship_id: str) -> PredicateKind:
        """
        Classify the predicate into a semantic kind.
        """
        item = self._relationship_mapping.get(relationship_id)
        if item is None:
            raise AttributeError(f"`{relationship_id}` not in relationship mapping.")
        return item.predicate_kind

    def predicate_kinds(
        self, relationship_ids: tuple[str, ...]
    ) -> Tuple[PredicateKind, ...]:
        """
        Classify a batch of predicates.
        """
        return tuple(self.predicate_kind(rel_id) for rel_id in relationship_ids)

    def relationships(
        self,
        session: Session,
        subjects: tuple[int, ...] | None,
        predicates: tuple[str, ...] | None,
        objects: tuple[int, ...] | None,
        invert: bool = False,
    ) -> Generator[Tuple[int, str, int], None, None]:
        """
        Query relationships between concepts.

        Parameters
        ----------
        subjects : list[CURIE] | None
            List of subject CURIEs.
        predicates : list[str] | None
            List of predicate (relationship) IDs.
        objects : list[CURIE] | None
            List of object CURIEs.
        invert : bool
            If True, swaps subjects and objects in the query and result.

        Yields
        -------
        Tuple[int, str, int]
            Triples of (subject_concept_id, relationship_id, object_concept_id).
            When ``invert=True``, the triple is (object_concept_id, relationship_id, subject_concept_id).
        """
        if invert:
            for s, p, o in self.relationships(
                session=session,
                subjects=objects,
                predicates=predicates,
                objects=subjects,
            ):
                yield o, p, s
            return

        for s, p, o in session.execute(
            q_relationships(
                subjects=subjects,
                predicates=predicates,
                objects=objects,
            )
        ):
            yield s, p, o

    def reverse_predicate_id(self, relationship_id: str) -> Optional[str]:
        """
        Get the reverse relationship ID, if it exists.
        """
        return self.predicate(relationship_id).reverse_id

    def _normalise_query_term(self, query_term: str) -> str:
        """
        Normalize a string for lookup (lowercase, single spaces).
        """
        return re.sub(r"\s+", " ", query_term.strip().lower())

    def edges(
        self,
        concept_ids: Tuple[int, ...] | int,
        direction: Literal["in", "out"],
        predicate_ids: Optional[frozenset[str]] = None,
        predicate_kinds: Optional[frozenset[PredicateKind]] = None,
        active_only: bool = True,
        on: Optional[date] = None,
        within_domain: bool = True,
    ) -> tuple[EdgeView, ...]:
        """
        Convenience method to retrieve all edges from one or multiple concepts.

        Parameters
        ----------
        concept_ids : int, tuple[int, ...]
            The source/target concept ID(s).
        direction : str
            'out' for outgoing, 'in' for incoming.
        predicate_ids : frozenset[str], optional
            Filter by specific relationship IDs.
        predicate_kinds : Set[PredicateKind], optional
            Filter by semantic kind of relationship.
        active_only : bool
            If True, return only valid/active edges.
        on : date, optional
            Check validity on a specific date.
        within_domain : bool
            If True, only return edges where source/target domains match.
        """
        with self.session_factory() as session:
            edges = tuple(
                self.iter_edges(
                    session=session,
                    concept_ids=concept_ids,
                    predicate_ids=predicate_ids,
                    direction=direction,
                    predicate_kinds=predicate_kinds,
                    active_only=active_only,
                    on=on,
                    within_domain=within_domain,
                )
            )
        return edges

    def iter_edges(
        self,
        session: Session,
        concept_ids: int | tuple[int, ...],
        direction: Literal["in", "out"],
        predicate_ids: Optional[frozenset[str]] = None,
        predicate_kinds: Optional[frozenset[PredicateKind]] = None,
        active_only: bool = True,
        on: Optional[date] = None,
        within_domain: bool = True,
    ) -> Generator[EdgeView, None, None]:

        stmt = q_edges(
            concept_ids=concept_ids,
            predicate_ids=predicate_ids,
            direction=direction,
            predicate_kinds=predicate_kinds,
            active_only=active_only,
            on=on,
            within_domain=within_domain,
        )

        for row in session.execute(stmt):
            yield EdgeView.from_query(row)

    def specificity(self, concept_id: int) -> float:
        """
        Compute specificity as the inverse of out-degree.
        Higher is more specific.
        """
        out_edges = self.edges(direction="out", concept_ids=concept_id)
        if not out_edges:
            return 1.0
        return 1.0 / len(out_edges)

    def parents(self, concept_id: int) -> tuple[int, ...]:
        """
        Retrieve parent Concept IDs of concept using Concept_Ancestor table.
        """
        with self.session_factory() as session:
            parents = tuple(session.execute(q_parents(concept_id)).scalars())
        return parents

    def children(self, concept_id) -> tuple[int, ...]:
        """
        Retrieve children Concept IDs of concept using Concept_Ancestor table.
        """
        with self.session_factory() as session:
            children = tuple(session.execute(q_children(concept_id)).scalars())
        return children

    def entities(
        self,
        session: Session,
        domain: str | None = None,
        standard_only: bool = True,
        filter_obsoletes: bool = True,
    ) -> Generator[int, None, None]:

        query = q_entities(
            domain=domain,
            standard_only=standard_only,
            filter_obsoletes=filter_obsoletes,
        )

        for row in session.execute(query):
            yield int(row.concept_id)

    def roots(
        self, domain_id: str | None = None, vocabulary_id: str | None = None
    ) -> tuple[int, ...]:
        """
        Retrieve root concepts (no parents).
        """
        with self.session_factory() as session:
            roots = tuple(
                session.execute(
                    q_roots(domain_id=domain_id, vocabulary_id=vocabulary_id)
                ).scalars()
            )
        return roots

    def leaves(
        self, domain_id: str | None = None, vocabulary_id: str | None = None
    ) -> tuple[int, ...]:
        """
        Retrieve leaf concepts (no children).
        """
        with self.session_factory() as session:
            leaves = tuple(
                session.execute(
                    q_leaves(domain_id=domain_id, vocabulary_id=vocabulary_id)
                ).scalars()
            )
        return leaves

    def singletons(
        self, domain_id: str | None = None, vocabulary_id: str | None = None
    ) -> tuple[int, ...]:
        """
        Retrieve singleton concepts (no parents and no children).
        """
        with self.session_factory() as session:
            return tuple(
                session.execute(
                    q_singletons(domain_id=domain_id, vocabulary_id=vocabulary_id)
                ).scalars()
            )

    def synonyms_for_concept(self, concept_id: int) -> tuple[str, ...]:
        """
        Retrieve all synonyms for a concept.
        """
        with self.session_factory() as session:
            rows = session.execute(q_concept_synonym_filtered(concept_id)).all()
        return tuple(row.name for row in rows)

    def predicates(self) -> tuple[Predicate, ...]:
        """
        Return all predicates known to the knowledge graph.
        """
        with self.session_factory() as session:
            rows = session.execute(q_all_predicates_with_ancestry()).all()
        return tuple(
            Predicate(
                relationship_id=row.relationship_id,
                name=row.relationship_name,
                reverse_id=row.reverse_relationship_id,
                is_hierarchical=bool(int(row.is_hierarchical)),
                anc_up=bool(int(row.anc_up)),
                anc_down=bool(int(row.anc_down)),
                predicate_kind=PredicateKind(row.predicate_kind),
                predicate_subkind=row.predicate_subkind,
            )
            for row in rows
        )

    @functools.cached_property
    def _valid_domains(self) -> frozenset[str]:
        with self.session_factory() as session:
            rows = session.execute(q_concept_domain_ids()).all()
        return frozenset(row.domain_id for row in rows)

    @functools.cached_property
    def _valid_vocabularies(self) -> frozenset[str]:
        with self.session_factory() as session:
            rows = session.execute(q_concept_vocabulary_ids()).all()
        return frozenset(row.vocabulary_id for row in rows)

    def get_potential_ancestor(
        self, child_id: int, parent_id: int
    ) -> Optional[AncestorMatch]:
        """
        Check if an ancestry relationship exists between a child and parent.
        """

        with self.session_factory() as session:
            row = session.execute(
                q_concept_potential_ancestor(child_id, parent_id)
            ).first()

        if row is None:
            return None
        return AncestorMatch(
            ancestor_concept_id=row.ancestor_concept_id,
            descendant_concept_id=row.descendant_concept_id,
            min_levels_of_separation=row.min_levels_of_separation,
        )

    def get_potential_ancestors_batch(
        self,
        child_ids: Tuple[int, ...],
        parent_ids: Tuple[int, ...],
    ) -> Dict[int, Dict[int, AncestorMatch]]:
        """Check which candidate parents are ancestors of one or more children.

        Parameters
        ----------
        child_ids : tuple of int
            A tuple of descendant concept IDs for batch mode.
        parent_ids : tuple of int
            Candidate ancestor concept IDs to check.

        Returns
        -------
        Dict[int, Dict[int, AncestorMatch]]
            When child_id is tuple: maps child_id -> {ancestor_concept_id -> AncestorMatch}.
        """
        if not parent_ids:
            return {}

        with self.session_factory() as session:
            rows = session.execute(
                q_concept_potential_ancestors_batch(child_ids, parent_ids)
            ).all()

        result: Dict[int, Dict[int, AncestorMatch]] = defaultdict(dict)
        for row in rows:
            result[row.descendant_concept_id][row.ancestor_concept_id] = AncestorMatch(
                ancestor_concept_id=row.ancestor_concept_id,
                descendant_concept_id=row.descendant_concept_id,
                min_levels_of_separation=row.min_levels_of_separation,
            )
        return result

    def get_num_ancestors(self, concept_ids: tuple[int, ...]) -> Dict[int, int]:
        """
        Get the count of ancestors for a batch of concepts.
        """
        with self.session_factory() as session:
            rows = session.execute(q_concept_num_ancestors(concept_ids)).all()
        return {row.concept_id: row.num_ancestors for row in rows}

    def check_search_constraints(self, constraints: SearchConstraintConcept) -> None:
        if constraints.domains is not None:
            invalid = [d for d in constraints.domains if d not in self._valid_domains]
            if invalid:
                raise ValueError(
                    f"Invalid domain constraint(s): {invalid}. "
                    f"Available domains: {sorted(self._valid_domains)}"
                )

        if constraints.vocabularies is not None:
            invalid = [
                v for v in constraints.vocabularies if v not in self._valid_vocabularies
            ]
            if invalid:
                raise ValueError(
                    f"Invalid vocabulary constraint(s): {invalid}. "
                    f"Available vocabularies: {sorted(self._valid_vocabularies)}"
                )
