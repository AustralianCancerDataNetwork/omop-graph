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

from sqlalchemy import Engine, Row
from sqlalchemy.orm import Session, sessionmaker
from omop_alchemy.backends import FullTextError
from omop_alchemy.cdm.query import ConceptFilter
from oa_configurator import ResolvedModel

if TYPE_CHECKING:
    from omop_emb import (
        EmbeddingBackend,
        EmbeddingWriterInterface,
        EmbeddingReaderInterface,
    )

# Local Application Imports
from ..extensions.emb import (
    MissingExtensionError,
    EmbeddingMetricType,
)
from ..extensions.omop_alchemy import (
    PredicateKind,
    RelationshipMappingElement,
    load_relationship_mapping,
)
from .base import GraphBackend
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
    q_relationship_mapping_all,
    q_relationship_mapping_row,
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

    A complete configuration: whenever embedding support is used at all
    (read or write), a real backend and a real resolved model are both
    required. The caller resolves the vector store and the model, 
    builds the backend, and passes both in here. Used to enhance the
    knowledge graph with embedding-based grounding and similarity scoring.

    Parameters
    ----------
    metric_type : EmbeddingMetricType
        The similarity/distance metric to use for embedding comparisons (e.g., cosine, euclidean).
        This is required to ensure that the correct type of index is used in the backend and that
        similarity computations are consistent.
    backend : omop_emb.EmbeddingBackend
        An already-constructed embedding backend, e.g. via
        ``omop_emb.backends.resolve_backend_from_resolved``.
    resolved_model : oa_configurator.ResolvedModel
        A model resolved via ``oa_configurator.Resolver.resolve_model()``, carrying real
        provider connection details. ``model_name``/``provider_type`` (see below) are
        read directly off it; used to build the embedding model backend via
        ``omop_llm.build_model_backend_from_resolved`` when ``write=True``.
    write : bool
        If True, the KG holds a write-capable interface that can generate and persist
        embeddings. If False (default), the KG only holds a read-only interface over
        already-computed embeddings.
    compute_missing_embeddings : bool
        If True, the system will compute embeddings on-the-fly for any concept that is not yet present
        in the embedding store. Requires ``write=True`` as it needs a write-capable interface.
        Default: False
    faiss_cache_dir : str, optional
        Directory to cache FAISS index files, for the read-only path only. Passed
        straight through to ``EmbeddingReaderInterface``.
    """

    metric_type: EmbeddingMetricType
    backend: "EmbeddingBackend"
    resolved_model: ResolvedModel
    write: bool = False
    compute_missing_embeddings: bool = False
    faiss_cache_dir: Optional[str] = None

    def __post_init__(self) -> None:
        if self.compute_missing_embeddings and not self.write:
            raise ValueError(
                "compute_missing_embeddings=True requires write=True "
                "(on-the-fly embedding computation needs a write-capable interface)."
            )

    @property
    def model_name(self) -> str:
        """The model name to use, read directly off ``resolved_model``."""
        return self.resolved_model.model

    @property
    def provider_type(self) -> str:
        """The provider key to use, read directly off ``resolved_model``."""
        return self.resolved_model.provider.provider


def _relationship_mapping_lookup(session: Session) -> dict[str, Row]:
    """RelationshipMapping rows keyed by relationship_id.

    RelationshipMapping is an omop-graph extension table, not vocab-role, so
    it never lives on a split ``vocab_engine``. This always runs against
    the primary connection.
    """
    return {
        row.relationship_id: row
        for row in session.execute(q_relationship_mapping_all()).all()
    }


def _predicate_from_rows(ancestry_row: Row, mapping_row: Row) -> Predicate:
    """Build a Predicate from a Relationship-ancestry row and a RelationshipMapping row.

    The two rows come from the same query in a same-connection deployment
    (pass the row twice), or from two separately-fetched engines in a
    split-connection one. This is the one place that shape difference
    collapses back into a single code path.
    """
    return Predicate(
        relationship_id=ancestry_row.relationship_id,
        name=ancestry_row.relationship_name,
        reverse_id=ancestry_row.reverse_relationship_id,
        is_hierarchical=bool(ancestry_row.is_hierarchical),
        anc_up=bool(ancestry_row.anc_up),
        anc_down=bool(ancestry_row.anc_down),
        predicate_kind=PredicateKind(mapping_row.predicate_kind),
        predicate_subkind=mapping_row.predicate_subkind,
    )


class KnowledgeGraph(GraphBackend):
    """
    The main entry point for interacting with the OMOP Graph.

    This class wraps a SQLAlchemy session and provides high-level methods
    to query concepts, relationships, and metadata.

    Parameters
    ----------
    cdm_engine : Engine
        The SQLAlchemy engine for the OMOP CDM database.
    vocab_engine : Engine, optional
        A separate engine for the vocabulary connection, for a deployment
        where ``vocab_connection`` names a physically different server than
        ``connection``. Omit (the common case) when vocabulary tables sit on
        the same connection as everything else: same-connection queries
        stay a single eager join. When given and different from
        ``cdm_engine``, the three queries that join a vocab-role table
        (Concept/Concept_Relationship/Relationship) against
        RelationshipMapping (not vocab-role, since it's an omop-graph
        extension table) fetch each side from its own engine and merge in
        Python, since a SQL join cannot span two physical connections.
    """

    def __init__(
        self,
        cdm_engine: Engine,
        vocab_engine: Optional[Engine] = None,
        emb_config: Optional[KnowledgeGraphEmbeddingConfiguration] = None,
    ):
        self.cdm_engine = cdm_engine
        self.session_factory = sessionmaker(bind=self.cdm_engine, future=True)

        self.vocab_engine = vocab_engine if vocab_engine is not None else cdm_engine
        self._vocab_split = self.vocab_engine is not self.cdm_engine
        self.vocab_session_factory = sessionmaker(bind=self.vocab_engine, future=True)

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

        The interface/reader is created lazily on first access, from the
        already-constructed backend supplied via
        ``KnowledgeGraphEmbeddingConfiguration.backend``; omop-graph itself
        never resolves omop-emb's config.
        """
        if self._emb is not None:
            return self._emb

        try:
            from omop_emb.interface import (
                EmbeddingWriterInterface,
                EmbeddingReaderInterface,
            )

            if self._emb_config is None:
                raise ValueError(
                    "Embedding configuration is not set. Please provide an EmbeddingConfiguration when initializing the KnowledgeGraph to use embedding features."
                )

            if self._emb_config.write:
                # Write-capable interface: the KG builds its own embedding model backend.
                self._emb = EmbeddingWriterInterface(
                    backend=self._emb_config.backend,
                    metric_type=self._emb_config.metric_type,
                    resolved_model=self._emb_config.resolved_model,
                    omop_cdm_engine=self.cdm_engine,
                )
            else:
                # Read-only interface: only ever needs model identity, never live credentials.
                self._emb = EmbeddingReaderInterface(
                    model=self._emb_config.model_name,
                    backend=self._emb_config.backend,
                    metric_type=self._emb_config.metric_type,
                    omop_cdm_engine=self.cdm_engine,
                    provider_type=self._emb_config.provider_type,
                    faiss_cache_dir=self._emb_config.faiss_cache_dir,
                )
            logger.debug(
                "Constructed %s embedding interface for model %r (write=%s).",
                "write-capable" if self._emb_config.write else "read-only",
                self._emb_config.model_name,
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
        search_constraint: Optional[ConceptFilter] = None,
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
        search_constraint : ConceptFilter, optional
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
        if self._vocab_split:
            with self.vocab_session_factory() as vsession:
                ancestry_row = vsession.execute(
                    q_predicate_row_with_ancestry(
                        relationship_id, include_classification=False
                    )
                ).one()
            with self.session_factory() as session:
                mapping_row = session.execute(
                    q_relationship_mapping_row(relationship_id)
                ).one()
            return _predicate_from_rows(ancestry_row, mapping_row)

        with self.session_factory() as session:
            row = session.execute(q_predicate_row_with_ancestry(relationship_id)).one()
        return _predicate_from_rows(row, row)

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

        if self._vocab_split:
            with self.vocab_session_factory() as vsession:
                vocab_rows = vsession.execute(
                    q_edges(
                        concept_ids=concept_ids,
                        predicate_ids=predicate_ids,
                        direction=direction,
                        active_only=active_only,
                        on=on,
                        within_domain=within_domain,
                        include_classification=False,
                    )
                ).all()
            mapping_by_id = _relationship_mapping_lookup(session)
            for vrow in vocab_rows:
                mapping = mapping_by_id.get(vrow.predicate_id)
                if mapping is None:
                    continue
                if predicate_kinds and PredicateKind(mapping.predicate_kind) not in predicate_kinds:
                    continue
                data = dict(vrow._mapping)
                data["predicate_kind"] = PredicateKind(mapping.predicate_kind)
                data["predicate_subkind"] = mapping.predicate_subkind
                yield EdgeView(**data)
            return

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
        if self._vocab_split:
            with self.vocab_session_factory() as vsession:
                ancestry_rows = vsession.execute(
                    q_all_predicates_with_ancestry(include_classification=False)
                ).all()
            with self.session_factory() as session:
                mapping_by_id = _relationship_mapping_lookup(session)
            return tuple(
                _predicate_from_rows(row, mapping_by_id[row.relationship_id])
                for row in ancestry_rows
                if row.relationship_id in mapping_by_id
            )

        with self.session_factory() as session:
            rows = session.execute(q_all_predicates_with_ancestry()).all()
        return tuple(_predicate_from_rows(row, row) for row in rows)

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

    def check_search_constraints(self, constraints: ConceptFilter) -> None:
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
