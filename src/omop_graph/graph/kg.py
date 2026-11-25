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
from collections import defaultdict
from collections.abc import Callable, Generator, Iterable
from dataclasses import dataclass, replace
from itertools import batched
from typing import TYPE_CHECKING, ClassVar, Literal

from oa_configurator import ResolvedModel
from omop_alchemy.backends import BackendNotSupportedError, FullTextError
from omop_alchemy.cdm.model.vocabulary import Concept
from omop_alchemy.cdm.query import ConceptFilter
from sqlalchemy import Row
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from sqlalchemy.sql import Select

if TYPE_CHECKING:
    from omop_emb import (
        EmbeddingBackend,
        EmbeddingReaderInterface,
        EmbeddingStoreReader,
        EmbeddingWriterInterface,
        MetricType,
    )

# Local Application Imports
from ..extensions.emb import MissingExtensionError
from ..extensions.omop_alchemy import (
    HierarchyDirection,
    PredicateKind,
    PredicateSubkind,
    RelationshipMappingElement,
    load_relationship_mapping,
)
from ._streaming import stream
from .base import GraphBackend
from .edges import IDENTITY_FILTER, IS_A_FILTER, EdgeFilter, EdgeView, Predicate
from .nodes import (
    AncestorMatch,
    ConceptView,
    LabelMatch,
    LabelMatchKind,
    MappingView,
    SynonymView,
    VocabularyView,
)
from .queries import (
    DEFAULT_CONCEPT_FILTER,
    q_all_predicates,
    q_children,
    q_closure,
    q_concept_domain_ids,
    q_concept_name_fulltext,
    q_concept_name_ilike,
    q_concept_name_match,
    q_concept_num_ancestors,
    q_concept_potential_ancestor,
    q_concept_potential_ancestors_batch,
    q_concept_view,
    q_concept_views,
    q_concept_vocabulary_ids,
    q_concepts_by_code,
    q_domains,
    q_edges,
    q_entities,
    q_leaves,
    q_mappings,
    q_obsolete_concepts,
    q_parents,
    q_predicate_name,
    q_predicate_row,
    q_relationships,
    q_roots,
    q_singletons,
    q_synonym_language_ids,
    q_synonyms,
    q_vocabularies,
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
    metric_type : MetricType
        The similarity/distance metric to use for embedding comparisons (e.g., cosine, euclidean).
        This is required to ensure that the correct type of index is used in the backend and that
        similarity computations are consistent.
    backend : omop_emb.EmbeddingBackend or omop_emb.EmbeddingStoreReader
        An already-constructed embedding backend. A write-capable
        ``EmbeddingBackend`` (e.g. via ``omop_emb.backends.open_vector_store_writer``)
        is required when ``write=True``; the narrower read-only
        ``EmbeddingStoreReader`` (via ``open_vector_store_reader``) suffices
        when ``write=False``.
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

    metric_type: MetricType
    backend: EmbeddingBackend | EmbeddingStoreReader
    resolved_model: ResolvedModel
    write: bool = False
    compute_missing_embeddings: bool = False
    faiss_cache_dir: str | None = None

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


def _predicate_from_rows(row: Row, mapping: RelationshipMappingElement) -> Predicate:
    """Build a Predicate from a relationship row and its classification."""
    return Predicate(
        relationship_id=row.relationship_id,
        name=row.relationship_name,
        reverse_id=row.reverse_relationship_id,
        predicate_kind=mapping.predicate_kind,
        predicate_subkind=mapping.predicate_subkind,
        direction=mapping.direction,
    )


class KnowledgeGraph(GraphBackend):
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

    LOOKUP_BATCH_SIZE: ClassVar[int] = 5_000
    """Most concept ids sent in one ``IN`` list."""

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

    @property
    def emb(self) -> EmbeddingWriterInterface | EmbeddingReaderInterface:
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
                EmbeddingReaderInterface,
                EmbeddingWriterInterface,
            )

            if self._emb_config is None:
                raise ValueError(
                    "Embedding configuration is not set. Please provide an EmbeddingConfiguration when initializing the KnowledgeGraph to use embedding features."
                )

            if self._emb_config.write:
                from omop_emb import EmbeddingBackend

                if not isinstance(self._emb_config.backend, EmbeddingBackend):
                    raise ValueError(
                        "write=True requires a write-capable EmbeddingBackend, "
                        f"got {type(self._emb_config.backend).__name__}."
                    )
                # Write-capable interface: the KG builds its own embedding model backend.
                self._emb = EmbeddingWriterInterface(
                    backend=self._emb_config.backend,
                    metric_type=self._emb_config.metric_type,
                    resolved_model=self._emb_config.resolved_model,
                    cdm_session_factory=self.session_factory,
                )
            else:
                # Read-only interface: only ever needs model identity, never live credentials.
                self._emb = EmbeddingReaderInterface(
                    model=self._emb_config.model_name,
                    backend=self._emb_config.backend,
                    metric_type=self._emb_config.metric_type,
                    cdm_session_factory=self.session_factory,
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
        except ImportError:
            logger.info(
                "Embedding functionality failed to initialize due to an import error in the optional embedding stack."
            )
            raise

    @property
    def embedding_configuration(self) -> KnowledgeGraphEmbeddingConfiguration | None:
        """Returns the current embedding configuration, if set."""
        return self._emb_config

    @property
    def compute_missing_embeddings(self) -> bool:
        """Indicates whether on-the-fly computation of missing concept embeddings is enabled."""
        return self._emb_config.compute_missing_embeddings if self._emb_config else False

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

    def concepts_by_code(self, vocabulary_id: str, concept_code: str) -> dict[int, bool]:
        """
        Look up every concept with a vocabulary ID and concept code.

        OMOP does not keep a code unique within its vocabulary, so there can be several.

        Parameters
        ----------
        vocabulary_id : str
            The vocabulary ID (e.g., 'SNOMED', 'RxNorm').
        concept_code : str
            The source code within that vocabulary.

        Returns
        -------
        dict[int, bool]
            Whether each matching concept is valid, by concept id in ascending order.
            Empty when no concept has the code.
        """
        with self.session_factory() as session:
            rows = session.execute(q_concepts_by_code(vocabulary_id, concept_code))
            return {int(row.concept_id): bool(row.is_valid) for row in rows}

    def concept_lookup(
        self,
        query_term: str,
        match_kind: LabelMatchKind,
        synonym: bool = False,
        search_constraint: ConceptFilter | None = None,
        sort: bool = True,
        limit: int | None = None,
    ) -> tuple[LabelMatch, ...]:
        """
        Resolve a query to concept_id(s).

        Parameters
        ----------
        query_term : str
            The term to search for.
        match_kind : LabelMatchKind
            The kind of match to perform (exact, starts with, fulltext, partial).
        synonym : bool
            If True, searches in Concept_Synonym instead of Concept.
        search_constraint : ConceptFilter, optional
            Additional filters for domain/vocabulary.
        sort : bool
            Whether to order the matches by quality.
        limit : int, optional
            The most matches to read.
        """
        input_query_term = self._normalise_query_term(query_term)
        if not input_query_term:
            return ()

        with self.session_factory() as session:
            if match_kind == LabelMatchKind.EXACT:
                fn = q_concept_name_match
            elif match_kind == LabelMatchKind.PARTIAL:
                fn = q_concept_name_ilike
            elif match_kind == LabelMatchKind.STARTS_WITH:
                fn = functools.partial(q_concept_name_ilike, anchored=True)
            elif match_kind == LabelMatchKind.FTS:
                fn = functools.partial(q_concept_name_fulltext, engine=session.get_bind(Concept))
            else:
                raise ValueError(f"Unsupported search mode: {match_kind}")
            try:
                cn = fn(
                    input_query_term,
                    search_constraint=search_constraint,
                    synonym=synonym,
                    sort=sort,
                )
            except (FullTextError, BackendNotSupportedError) as e:
                if match_kind == LabelMatchKind.FTS:
                    logger.info(e)
                    return ()
                raise

            if limit is not None:
                cn = cn.limit(limit)
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

    def concept_ids_by_label(self, label: str) -> tuple[int, ...]:
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
            row = session.execute(q_predicate_row(relationship_id)).one()
        return _predicate_from_rows(row, self._mapping_for(relationship_id))

    def predicate_name(self, relationship_id: str) -> str:
        """
        Retrieve the human-readable name of a relationship.
        """
        # TODO: Not really necessary. The "ID" is mostly human-readable anyways.
        with self.session_factory() as session:
            predicate_name = session.execute(q_predicate_name(relationship_id)).scalar_one()
        return predicate_name

    def predicate_kind(self, relationship_id: str) -> PredicateKind:
        """
        Classify the predicate into a semantic kind.
        """
        return self._mapping_for(relationship_id).predicate_kind

    def _mapping_for(self, relationship_id: str) -> RelationshipMappingElement:
        """Classification of *relationship_id*, raising AttributeError if it has none."""
        item = self._relationship_mapping.get(relationship_id)
        if item is None:
            raise AttributeError(f"`{relationship_id}` not in relationship mapping.")
        return item

    def predicate_kinds(self, relationship_ids: tuple[str, ...]) -> tuple[PredicateKind, ...]:
        """
        Classify a batch of predicates.
        """
        return tuple(self.predicate_kind(rel_id) for rel_id in relationship_ids)

    def relationships(
        self,
        subjects: tuple[int, ...] | None,
        objects: tuple[int, ...] | None,
        edge_filter: EdgeFilter | None = None,
        invert: bool = False,
    ) -> Generator[tuple[int, str, int], None, None]:
        """
        Query relationships between concepts.

        Parameters
        ----------
        subjects : tuple[int, ...] | None
            Subject concept ids.
        objects : tuple[int, ...] | None
            Object concept ids.
        edge_filter : EdgeFilter, optional
            Which relationships to return. Defaults to ``EdgeFilter()``. Relationships
            without a classification are skipped, as in :meth:`iter_edges`.
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
                subjects=objects,
                objects=subjects,
                edge_filter=edge_filter,
            ):
                yield o, p, s
            return

        resolved = self._resolve_edge_filter(edge_filter)
        if resolved is None:
            return

        yield from self._classified_triples(
            q_relationships(subjects=subjects, objects=objects, edge_filter=resolved)
        )

    def all_relationships(
        self, edge_filter: EdgeFilter | None = None
    ) -> Generator[tuple[int, str, int], None, None]:
        """
        Yield every relationship selected by *edge_filter*, read through a server-side cursor.

        Meant for scans of the whole table. The triples are as in :meth:`relationships`.
        """
        resolved = self._resolve_edge_filter(edge_filter)
        if resolved is None:
            return
        yield from self._classified_triples(stream(q_relationships(None, None, resolved)))

    def mappings(
        self,
        concept_ids: Iterable[int] | None = None,
        edge_filter: EdgeFilter | None = None,
        source: str | None = None,
    ) -> Generator[MappingView, None, None]:
        """
        Yield the edges starting from concepts, with the vocabularies of both ends.

        Edges from a concept to itself and edges without a classification are left out.
        Without *concept_ids* every edge is read through a server-side cursor.

        Parameters
        ----------
        concept_ids : Iterable[int] | None
            Concepts to start from, or None for all.
        edge_filter : EdgeFilter, optional
            Which edges to return, as in :meth:`relationships`.
        source : str, optional
            Only edges with this vocabulary on either side.
        """
        resolved = self._resolve_edge_filter(edge_filter)
        if resolved is None:
            return
        if concept_ids is None:
            statements = [stream(q_mappings(None, resolved, source))]
        else:
            ids = tuple(dict.fromkeys(concept_ids))
            statements = [
                q_mappings(batch, resolved, source)
                for batch in batched(ids, self.LOOKUP_BATCH_SIZE)
            ]
        with self.session_factory() as session:
            for stmt in statements:
                for row in session.execute(stmt):
                    if row.predicate_id in self._relationship_mapping:
                        yield MappingView(**row._mapping)

    def _classified_triples(self, stmt: Select) -> Generator[tuple[int, str, int], None, None]:
        """Run *stmt*, a relationship triple query, and keep the triples of classified relationships."""
        with self.session_factory() as session:
            for s, p, o in session.execute(stmt):
                if p in self._relationship_mapping:
                    yield s, p, o

    def reverse_predicate_id(self, relationship_id: str) -> str | None:
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
        concept_ids: tuple[int, ...] | int,
        direction: Literal["in", "out"],
        edge_filter: EdgeFilter | None = None,
    ) -> tuple[EdgeView, ...]:
        """
        Convenience method to retrieve all edges from one or multiple concepts.

        Parameters
        ----------
        concept_ids : int, tuple[int, ...]
            The source/target concept ID(s).
        direction : str
            'out' for outgoing, 'in' for incoming.
        edge_filter : EdgeFilter, optional
            Which edges to return. Defaults to ``EdgeFilter()``.
        """
        with self.session_factory() as session:
            edges = tuple(
                self.iter_edges(
                    session=session,
                    concept_ids=concept_ids,
                    direction=direction,
                    edge_filter=edge_filter,
                )
            )
        return edges

    def iter_edges(
        self,
        session: Session,
        concept_ids: int | tuple[int, ...],
        direction: Literal["in", "out"],
        edge_filter: EdgeFilter | None = None,
        ordered: bool = False,
    ) -> Generator[EdgeView, None, None]:
        """Yield edges of *concept_ids* on *session*, classified from the relationship mapping.

        Edges whose relationship has no classification are skipped. If
        ``ordered=True``, edges are ordered by source concept, target concept,
        and relationship ID, making results reproducible. If ``ordered=False``,
        row order is unspecified and may differ between runs and backends.
        """
        resolved = self._resolve_edge_filter(edge_filter)
        if resolved is None:
            return

        stmt = q_edges(
            concept_ids=concept_ids,
            direction=direction,
            edge_filter=resolved,
            ordered=ordered,
        )
        for row in session.execute(stmt):
            mapping = self._relationship_mapping.get(row.predicate_id)
            if mapping is None:
                continue
            yield EdgeView(
                **row._mapping,
                predicate_kind=mapping.predicate_kind,
                predicate_subkind=mapping.predicate_subkind,
            )

    def _resolve_edge_filter(self, edge_filter: EdgeFilter | None) -> EdgeFilter | None:
        """Fold the kind and subkind fields of *edge_filter* into ``predicate_ids``.

        Returns None when they select no relationship, so the caller has nothing
        to query. An empty ``predicate_ids`` selects none, where None selects all.

        Raises
        ------
        ValueError
            If ``predicate_subkinds`` names a subkind no relationship is classified under.
        """
        edge_filter = edge_filter or EdgeFilter()
        if edge_filter.predicate_ids is not None and not edge_filter.predicate_ids:
            return None
        kinds, subkinds = edge_filter.predicate_kinds, edge_filter.predicate_subkinds
        if not kinds and not subkinds:
            return edge_filter

        if subkinds:
            known = {
                PredicateSubkind(m.predicate_kind, m.predicate_subkind)
                for m in self._relationship_mapping.values()
            }
            if unknown := subkinds - known:
                raise ValueError(
                    f"Unknown predicate subkinds {sorted(map(str, unknown))}; "
                    f"valid subkinds are {sorted(map(str, known))}."
                )

        selected = frozenset(
            relationship_id
            for relationship_id, m in self._relationship_mapping.items()
            if (not kinds or m.predicate_kind in kinds)
            and (
                not subkinds or PredicateSubkind(m.predicate_kind, m.predicate_subkind) in subkinds
            )
        )
        if edge_filter.predicate_ids is not None:
            selected &= edge_filter.predicate_ids
        if not selected:
            return None
        return replace(
            edge_filter, predicate_ids=selected, predicate_kinds=None, predicate_subkinds=None
        )

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

    def closure(
        self,
        concept_ids: Iterable[int],
        direction: HierarchyDirection = HierarchyDirection.UP,
        edge_filter: EdgeFilter = IS_A_FILTER,
        concept_filter: ConceptFilter | None = None,
    ) -> dict[int, frozenset[int]]:
        """
        Find the concepts reached by following edges repeatedly from each concept.

        By default this is the strict ``Is a`` ancestors, which unlike :meth:`parents`
        (``concept_ancestor``) holds no cohort, therapeutic or composition ancestry. Other
        edge filters give the closure over other relationships, which can be very large.

        Parameters
        ----------
        concept_ids : Iterable[int]
            Concepts to start from.
        direction : HierarchyDirection
            ``UP`` follows an edge from its subject to its object, as to ancestors, and
            ``DOWN`` from its object to its subject, as to descendants.
        edge_filter : EdgeFilter
            The edges to follow. Relationships without a classification are not followed.
        concept_filter : ConceptFilter, optional
            Report only the concepts it selects. The walk still passes through the others,
            so a concept above an excluded one is still reached. ``concept_ids`` must be unset.

        Returns
        -------
        dict[int, frozenset[int]]
            Every requested concept mapped to the concepts reached from it, never itself.
        """
        starts = tuple(dict.fromkeys(concept_ids))
        reached: dict[int, set[int]] = {concept_id: set() for concept_id in starts}
        resolved = self._resolve_edge_filter(edge_filter)
        if resolved is not None:
            followed = frozenset(self._relationship_mapping)
            if resolved.predicate_ids is not None:
                followed &= resolved.predicate_ids
            resolved = replace(resolved, predicate_ids=followed)
        if resolved is not None and resolved.predicate_ids:
            with self.session_factory() as session:
                for batch in batched(starts, self.LOOKUP_BATCH_SIZE):
                    for origin, other in session.execute(q_closure(batch, direction, resolved)):
                        reached[origin].add(other)
        if concept_filter is not None:
            in_scope = self._within(set().union(*reached.values()), concept_filter)
            reached = {concept_id: others & in_scope for concept_id, others in reached.items()}
        return {concept_id: frozenset(others) for concept_id, others in reached.items()}

    def standardise(
        self, concept_ids: Iterable[int], edge_filter: EdgeFilter = IDENTITY_FILTER
    ) -> dict[int, tuple[int, ...]]:
        """
        Find the standard concepts each concept stands for.

        A standard or classification concept stands for itself. Any other concept stands
        for the standard concepts at the end of its identity edges, such as ``Maps to``
        and ``Concept replaced by``, which may be several. Edges to non-standard concepts
        are not followed further.

        Parameters
        ----------
        concept_ids : Iterable[int]
            Concepts to standardise.
        edge_filter : EdgeFilter
            The edges that map a concept to its standard concepts.

        Returns
        -------
        dict[int, tuple[int, ...]]
            Every requested concept mapped to its standard concepts in ascending order,
            empty when it has none or does not exist.
        """
        ids = tuple(dict.fromkeys(concept_ids))
        standard = self._standard_among(ids)
        mapped: dict[int, set[int]] = {concept_id: set() for concept_id in ids}
        sources = tuple(concept_id for concept_id in ids if concept_id not in standard)
        for batch in batched(sources, self.LOOKUP_BATCH_SIZE):
            for subject_id, _, object_id in self.relationships(batch, None, edge_filter):
                mapped[subject_id].add(object_id)
        standard_targets = self._standard_among(set().union(*mapped.values()))
        return {
            concept_id: (
                (concept_id,)
                if concept_id in standard
                else tuple(sorted(mapped[concept_id] & standard_targets))
            )
            for concept_id in ids
        }

    def _standard_among(self, concept_ids: Iterable[int]) -> set[int]:
        """Select the standard or classification concepts among *concept_ids*."""
        found: set[int] = set()
        for batch in batched(set(concept_ids), self.LOOKUP_BATCH_SIZE):
            found.update(
                view.concept_id
                for view in self.concept_views(batch, sort=False)
                if view.standard_concept
            )
        return found

    def _within(self, concept_ids: Iterable[int], concept_filter: ConceptFilter) -> set[int]:
        """Select the concepts among *concept_ids* that *concept_filter* selects."""
        found: set[int] = set()
        with self.session_factory() as session:
            for batch in batched(set(concept_ids), self.LOOKUP_BATCH_SIZE):
                stmt = q_entities(replace(concept_filter, concept_ids=tuple(batch)))
                found.update(session.execute(stmt).scalars())
        return found

    def entailed_relationships(
        self,
        subjects: tuple[int, ...] | None,
        objects: tuple[int, ...] | None,
        edge_filter: EdgeFilter | None = None,
        concept_filter: ConceptFilter | None = None,
    ) -> Generator[tuple[int, str, int], None, None]:
        """
        Query relationships, including those entailed by the strict ``Is a`` hierarchy.

        A triple (S, R, E) is entailed when (S, R, D) is asserted and E is D or an
        ``Is a`` ancestor of D. Relationships of ancestors are not inherited downwards.
        Each triple is yielded once.

        Parameters
        ----------
        subjects : tuple[int, ...] | None
            Subject concept ids.
        objects : tuple[int, ...] | None
            Object concept ids. At least one of *subjects* and *objects* is required.
        edge_filter : EdgeFilter, optional
            Which asserted relationships to start from, as in :meth:`relationships`.
        concept_filter : ConceptFilter, optional
            Which ``Is a`` ancestors, or descendants for incoming relationships, count as
            entailed objects, as in :meth:`closure`. Asserted edges are unaffected.

        Yields
        ------
        tuple[int, str, int]
            Triples of (subject_concept_id, relationship_id, object_concept_id).

        Raises
        ------
        ValueError
            If both *subjects* and *objects* are None.
        """
        if subjects is None and objects is None:
            raise ValueError("Entailed relationships need subjects or objects")
        if subjects is not None:
            triples = self._entailed_from_subjects(subjects, objects, edge_filter, concept_filter)
        else:
            assert objects is not None
            triples = self._entailed_into_objects(objects, edge_filter, concept_filter)
        seen: set[tuple[int, str, int]] = set()
        for triple in triples:
            if triple not in seen:
                seen.add(triple)
                yield triple

    def _entailed_from_subjects(
        self,
        subjects: tuple[int, ...],
        objects: tuple[int, ...] | None,
        edge_filter: EdgeFilter | None,
        concept_filter: ConceptFilter | None,
    ) -> Generator[tuple[int, str, int], None, None]:
        """Generalise the filler of every asserted edge of *subjects* up the ``Is a`` hierarchy."""
        wanted = None if objects is None else frozenset(objects)
        asserted: list[tuple[int, str, int]] = []
        for batch in batched(subjects, self.LOOKUP_BATCH_SIZE):
            asserted.extend(self.relationships(batch, None, edge_filter))
        ancestors = self.closure(
            {filler for _, _, filler in asserted}, concept_filter=concept_filter
        )
        for subject_id, relationship_id, filler in asserted:
            for reached in (filler, *ancestors[filler]):
                if wanted is None or reached in wanted:
                    yield subject_id, relationship_id, reached

    def _entailed_into_objects(
        self,
        objects: tuple[int, ...],
        edge_filter: EdgeFilter | None,
        concept_filter: ConceptFilter | None,
    ) -> Generator[tuple[int, str, int], None, None]:
        """Find the edges into *objects* and into their ``Is a`` descendants."""
        descendants = self.closure(objects, HierarchyDirection.DOWN, concept_filter=concept_filter)
        targets_by_filler: dict[int, list[int]] = defaultdict(list)
        for target, below in descendants.items():
            for filler in (target, *below):
                targets_by_filler[filler].append(target)
        for batch in batched(targets_by_filler, self.LOOKUP_BATCH_SIZE):
            for subject_id, relationship_id, filler in self.relationships(None, batch, edge_filter):
                for target in targets_by_filler[filler]:
                    yield subject_id, relationship_id, target

    def entities(
        self, concept_filter: ConceptFilter = DEFAULT_CONCEPT_FILTER
    ) -> Generator[int, None, None]:
        """
        Yield the ids of the concepts selected by *concept_filter*.

        Defaults to active standard and classification concepts.
        """
        with self.session_factory() as session:
            for row in session.execute(q_entities(concept_filter)):
                yield int(row.concept_id)

    def roots(
        self,
        concept_filter: ConceptFilter = DEFAULT_CONCEPT_FILTER,
        edge_filter: EdgeFilter | None = None,
    ) -> tuple[int, ...]:
        """
        Retrieve root concepts (no parents) among those selected by *concept_filter*.

        Without *edge_filter* the parents are the closure's level 1 ancestors. With it, a
        parent is the target of an outgoing edge it selects.
        """
        return self._enumerate(q_roots, concept_filter, edge_filter)

    def leaves(
        self,
        concept_filter: ConceptFilter = DEFAULT_CONCEPT_FILTER,
        edge_filter: EdgeFilter | None = None,
    ) -> tuple[int, ...]:
        """
        Retrieve leaf concepts (no children) among those selected by *concept_filter*.

        Without *edge_filter* the children are the closure's level 1 descendants. With it,
        a child is the source of an incoming edge it selects.
        """
        return self._enumerate(q_leaves, concept_filter, edge_filter)

    def singletons(
        self,
        concept_filter: ConceptFilter = DEFAULT_CONCEPT_FILTER,
        edge_filter: EdgeFilter | None = None,
    ) -> tuple[int, ...]:
        """
        Retrieve singleton concepts (no parents and no children) among those selected by
        *concept_filter*. See :meth:`roots` for *edge_filter*.
        """
        return self._enumerate(q_singletons, concept_filter, edge_filter)

    def _enumerate(
        self,
        build: Callable[[ConceptFilter, EdgeFilter | None], Select],
        concept_filter: ConceptFilter,
        edge_filter: EdgeFilter | None,
    ) -> tuple[int, ...]:
        """Run *build* with *edge_filter* resolved, or list every concept when it selects no edge."""
        if edge_filter is None:
            stmt = build(concept_filter, None)
        else:
            resolved = self._resolve_edge_filter(edge_filter)
            stmt = build(concept_filter, resolved) if resolved else q_entities(concept_filter)
        with self.session_factory() as session:
            return tuple(session.execute(stmt).scalars())

    def obsolete_concepts(self, include_updated: bool = True) -> Generator[int, None, None]:
        """
        Yield the ids of deprecated concepts.

        Parameters
        ----------
        include_updated : bool
            If True, include the concepts updated into others (``invalid_reason`` ``U``)
            besides the deleted ones (``D``).
        """
        with self.session_factory() as session:
            yield from session.execute(stream(q_obsolete_concepts(include_updated))).scalars()

    def vocabularies(self) -> tuple[VocabularyView, ...]:
        """Return every vocabulary."""
        with self.session_factory() as session:
            return tuple(
                VocabularyView(**row._mapping) for row in session.execute(q_vocabularies())
            )

    def domain_concept_ids(self) -> dict[str, int]:
        """Return the concept id of each domain, keyed by ``domain_id``."""
        with self.session_factory() as session:
            return {row.domain_id: row.domain_concept_id for row in session.execute(q_domains())}

    def synonyms(self, concept_ids: tuple[int, ...]) -> tuple[SynonymView, ...]:
        """Return the synonyms of *concept_ids*, ordered by concept and name."""
        with self.session_factory() as session:
            return tuple(
                SynonymView(**row._mapping) for row in session.execute(q_synonyms(concept_ids))
            )

    def synonym_language_ids(self) -> frozenset[int]:
        """Return the language concept ids that at least one synonym uses."""
        with self.session_factory() as session:
            return frozenset(session.execute(q_synonym_language_ids()).scalars())

    def synonyms_for_concept(self, concept_id: int) -> tuple[str, ...]:
        """
        Retrieve all synonyms for a concept.
        """
        return tuple(synonym.name for synonym in self.synonyms((concept_id,)))

    def predicates(self) -> tuple[Predicate, ...]:
        """
        Return all predicates known to the knowledge graph.
        """
        with self.session_factory() as session:
            rows = session.execute(q_all_predicates()).all()
        return tuple(
            _predicate_from_rows(row, self._relationship_mapping[row.relationship_id])
            for row in rows
            if row.relationship_id in self._relationship_mapping
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

    def get_potential_ancestor(self, child_id: int, parent_id: int) -> AncestorMatch | None:
        """
        Check if an ancestry relationship exists between a child and parent.
        """

        with self.session_factory() as session:
            row = session.execute(q_concept_potential_ancestor(child_id, parent_id)).first()

        if row is None:
            return None
        return AncestorMatch(
            ancestor_concept_id=row.ancestor_concept_id,
            descendant_concept_id=row.descendant_concept_id,
            min_levels_of_separation=row.min_levels_of_separation,
        )

    def get_potential_ancestors_batch(
        self,
        child_ids: tuple[int, ...],
        parent_ids: tuple[int, ...],
    ) -> dict[int, dict[int, AncestorMatch]]:
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

        rows: list[Row] = []
        with self.session_factory() as session:
            for batch in batched(child_ids, self.LOOKUP_BATCH_SIZE):
                rows.extend(session.execute(q_concept_potential_ancestors_batch(batch, parent_ids)))

        result: dict[int, dict[int, AncestorMatch]] = defaultdict(dict)
        for row in rows:
            result[row.descendant_concept_id][row.ancestor_concept_id] = AncestorMatch(
                ancestor_concept_id=row.ancestor_concept_id,
                descendant_concept_id=row.descendant_concept_id,
                min_levels_of_separation=row.min_levels_of_separation,
            )
        return result

    def get_num_ancestors(self, concept_ids: tuple[int, ...]) -> dict[int, int]:
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
            invalid = [v for v in constraints.vocabularies if v not in self._valid_vocabularies]
            if invalid:
                raise ValueError(
                    f"Invalid vocabulary constraint(s): {invalid}. "
                    f"Available vocabularies: {sorted(self._valid_vocabularies)}"
                )
