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

import logging
import re
from collections import defaultdict
from datetime import date
from functools import lru_cache
from typing import Dict, Iterable, Optional, Set, Tuple, Union

from sqlalchemy.exc import InvalidRequestError, PendingRollbackError
from sqlalchemy.orm import Session

# Local Application Imports
from omop_alchemy.cdm.base.embeddings import _MODEL_CACHE
from .base import GraphBackend
from .constraints import SearchConstraintConcept
from .edges import EdgeView, Predicate, PredicateKind, PredicateSummary, is_active
from .nodes import (
    AncestorMatch,
    ConceptView,
    LabelMatch,
    LabelMatchGroupView,
    LabelMatchKind,
)
from .queries import (
    q_all_predicates,
    q_all_predicates_with_ancestry,
    q_concept_domain_ids,
    q_concept_id_by_code,
    q_concept_name_fulltext,
    q_concept_name_ilike,
    q_concept_name_match,
    q_concept_num_ancestors,
    q_concept_potential_ancestor,
    q_concept_synonym_filtered,
    q_concept_synonym_fulltext,
    q_concept_synonym_ilike,
    q_concept_synonym_match,
    q_concept_view,
    q_concept_views,
    q_concept_vocabulary_ids,
    q_incoming_edges,
    q_incoming_edges_batch,
    q_leaves,
    q_outgoing_edges,
    q_outgoing_edges_batch,
    q_parents,
    q_predicate_name,
    q_predicate_row,
    q_predicate_row_with_ancestry,
    q_roots,
    q_singletons,
    q_embedding_model_table_name,
    q_embedding_cosine_similarity,
)

logger = logging.getLogger(__name__)


def _pred_id(pred: Union[Predicate, str, None]) -> Optional[str]:
    """
    Helper to extract the relationship ID string from various input types.
    """
    if pred is None:
        return None
    if isinstance(pred, Predicate):
        return pred.relationship_id
    if isinstance(pred, str):
        return pred
    raise TypeError(f"Unsupported predicate type: {type(pred)}")


class KnowledgeGraph(GraphBackend):
    """
    The main entry point for interacting with the OMOP Graph.

    This class wraps a SQLAlchemy session and provides high-level methods
    to query concepts, relationships, and metadata.

    Parameters
    ----------
    session : Session
        The SQLAlchemy session connected to the OMOP database.
    """

    def __init__(self, session: Session):
        self.session = session

    @lru_cache(maxsize=200_000)
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
        row = self.session.execute(q_concept_view(concept_id)).one()
        return ConceptView.from_row(row)

    @lru_cache(maxsize=200_000)
    def concept_views(self, concept_ids: tuple[int, ...]) -> tuple[ConceptView, ...]:
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
        return tuple(
            ConceptView.from_row(row)
            for row in self.session.execute(q_concept_views(concept_ids)).all()
        )

    @lru_cache(maxsize=200_000)
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
        return int(
            self.session.execute(
                q_concept_id_by_code(vocabulary_id, concept_code)
            ).scalar_one()
        )

    @lru_cache(maxsize=200_000)
    def _synonym_lookup_raw(
        self,
        label: str,
        fuzzy: bool = False,
        search_constraint: Optional[SearchConstraintConcept] = None,
    ) -> Tuple[LabelMatch, ...]:
        """
        Resolve a synonym label to concept_id(s).

        Returns matches annotated with LabelMatchKind for downstream explanations.
        """
        input_label = self._normalise_label(label)
        if not input_label:
            return ()

        fn = q_concept_synonym_ilike if fuzzy else q_concept_synonym_match
        cs = fn(input_label, search_constraint=search_constraint)
        syn_rows = self.session.execute(cs).all()

        return tuple(
            LabelMatch(
                input_label=input_label,
                matched_label=name,
                concept_id=int(cid),
                match_kind=LabelMatchKind.SYNONYM,
                is_standard=is_standard,
                is_active=is_active,
            )
            for cid, name, is_standard, is_active in syn_rows
        )

    @lru_cache(maxsize=200_000)
    def _label_lookup_raw(
        self,
        label: str,
        fuzzy: bool = False,
        search_constraint: Optional[SearchConstraintConcept] = None,
    ) -> Tuple[LabelMatch, ...]:
        """
        Resolve a label to concept_id(s), preferring Concept.concept_name matches.
        Returns matches annotated with LabelMatchKind for downstream explanations.
        """
        input_label = self._normalise_label(label)
        if not input_label:
            return ()

        fn = q_concept_name_ilike if fuzzy else q_concept_name_match
        cn = fn(input_label, search_constraint=search_constraint)

        direct_rows = self.session.execute(cn).all()

        return tuple(
            LabelMatch(
                input_label=input_label,
                matched_label=name,
                concept_id=int(cid),
                match_kind=LabelMatchKind.DIRECT,
                is_standard=is_standard,
                is_active=is_active,
            )
            for cid, name, is_standard, is_active in direct_rows
        )

    @lru_cache(maxsize=200_000)
    def _fulltext_lookup_raw(
        self,
        label: str,
        fuzzy: bool = False,
        search_constraint: Optional[SearchConstraintConcept] = None,
    ) -> Tuple[LabelMatch, ...]:
        """
        Resolve a label using fulltext search (bag of words, ignoring word order).
        """
        q = q_concept_synonym_fulltext if fuzzy else q_concept_name_fulltext
        rows = self.session.execute(q(label, search_constraint=search_constraint)).all()

        return tuple(
            LabelMatch(
                input_label=label,
                matched_label=name,
                concept_id=int(cid),
                match_kind=LabelMatchKind.FULLTEXT,
                is_standard=is_standard,
                is_active=is_active,
            )
            for cid, name, is_standard, is_active in rows
        )

    def synonym_lookup(
        self,
        label: str,
        fuzzy: bool = False,
        sort: bool = True,
        search_constraint: Optional[SearchConstraintConcept] = None,
    ) -> LabelMatchGroupView:
        """
        Resolve a synonym label to grouped concept matches.

        Parameters
        ----------
        label : str
            The term to search for.
        fuzzy : bool
            If True, performs an ILIKE search.
        sort : bool
            If True, sorts the results.
        search_constraint : SearchConstraintConcept, optional
            Filters for domain/vocab.

        Returns
        -------
        LabelMatchGroupView
            Grouped matches.
        """
        raw = self._synonym_lookup_raw(
            label, fuzzy=fuzzy, search_constraint=search_constraint
        )
        if sort:
            raw = sorted(raw)
        return LabelMatchGroupView.from_matches(raw)

    def label_lookup(
        self,
        label: str,
        fuzzy: bool = False,
        sort: bool = True,
        search_constraint: Optional[SearchConstraintConcept] = None,
    ) -> LabelMatchGroupView:
        """
        Resolve a label to grouped concept matches, preferring direct name matches.

        Parameters
        ----------
        label : str
            The term to search for.
        fuzzy : bool
            If True, performs an ILIKE search.
        sort : bool
            If True, sorts the results.
        search_constraint : SearchConstraintConcept, optional
            Filters for domain/vocab.

        Returns
        -------
        LabelMatchGroupView
            Grouped matches.
        """
        raw = self._label_lookup_raw(
            label, fuzzy=fuzzy, search_constraint=search_constraint
        )
        if sort:
            raw = sorted(raw)
        return LabelMatchGroupView.from_matches(raw)

    def fulltext_lookup(
        self,
        label: str,
        sort: bool = True,
        fuzzy: bool = False,
        search_constraint: Optional[SearchConstraintConcept] = None,
    ) -> LabelMatchGroupView:
        """
        Resolve a label using fulltext search (bag of words, ignoring word order).

        Parameters
        ----------
        label : str
            The term to search for.
        sort : bool
            If True, sorts the results.
        fuzzy : bool
            If True, searches synonyms instead of just concept names.
        search_constraint : SearchConstraintConcept, optional
            Filters for domain/vocab.

        Returns
        -------
        LabelMatchGroupView
            Grouped matches.
        """
        raw = self._fulltext_lookup_raw(
            label, fuzzy=fuzzy, search_constraint=search_constraint
        )
        if sort:
            raw = sorted(raw)

        return LabelMatchGroupView.from_matches(raw)

    @lru_cache(maxsize=200_000)
    def concept_ids_by_label(self, label: str) -> Tuple[int, ...]:
        """
        Find concept IDs that match the label exactly (case-insensitive).
        """
        rows = self.session.execute(q_concept_name_match(label)).scalars()
        return tuple(rows)

    @lru_cache(maxsize=10_000)
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
        row = self.session.execute(q_predicate_row_with_ancestry(relationship_id)).one()

        return Predicate(
            relationship_id=row.relationship_id,
            name=row.relationship_name,
            reverse_id=row.reverse_relationship_id,
            is_hierarchical=bool(row.is_hierarchical),
            anc_up=bool(int(row.anc_up)),
            anc_down=bool(int(row.anc_up)),
        )

    @lru_cache(maxsize=10_000)
    def predicate_name(self, relationship_id: str) -> str:
        """
        Retrieve the human-readable name of a relationship.
        """
        return self.session.execute(q_predicate_name(relationship_id)).scalar_one()

    def predicate_kind(self, relationship_id: str) -> PredicateKind:
        """
        Classify the predicate into a semantic kind.
        """
        return self.predicate(relationship_id).classify_predicate()

    def predicate_kinds(
        self, relationship_ids: tuple[str, ...]
    ) -> Tuple[PredicateKind, ...]:
        """
        Classify a batch of predicates.
        """
        return tuple(self.predicate_kind(rel_id) for rel_id in relationship_ids)

    def reverse_predicate_id(self, relationship_id: str) -> Optional[str]:
        """
        Get the reverse relationship ID, if it exists.
        """
        return self.predicate(relationship_id).reverse_id

    def _normalise_label(self, s: str) -> str:
        """
        Normalize a string for lookup (lowercase, single spaces).
        """
        return re.sub(r"\s+", " ", s.strip().lower())

    @lru_cache(maxsize=500_000)
    def outgoing_edges(
        self,
        concept_id: int,
        relationship_id: str | None = None,
    ) -> tuple[EdgeView, ...]:
        """
        Retrieve all outgoing edges from a concept.
        """
        stmt = q_outgoing_edges(concept_id, relationship_id)
        return tuple(EdgeView(*row) for row in self.session.execute(stmt).all())

    @lru_cache(maxsize=500_000)
    def outgoing_edges_batch(
        self,
        concept_ids: tuple[int, ...],
        relationship_id: str | None = None,
    ) -> tuple[EdgeView, ...]:
        """
        Retrieve all outgoing edges for a batch of concepts.
        """
        stmt = q_outgoing_edges_batch(concept_ids, relationship_id)
        return tuple(EdgeView(*row) for row in self.session.execute(stmt).all())

    def specificity(self, concept_id: int) -> float:
        """
        Compute specificity as the inverse of out-degree.
        Higher is more specific.
        """
        out_edges = self.outgoing_edges(concept_id)
        if not out_edges:
            return 1.0
        return 1.0 / len(out_edges)

    def _same_domain(self, e: EdgeView) -> bool:
        """Check if subject and object of an edge belong to the same domain."""
        subj = self.concept_view(e.subject_id)
        obj = self.concept_view(e.object_id)
        return subj.domain_id == obj.domain_id

    @lru_cache(maxsize=500_000)
    def incoming_edges(
        self,
        concept_id: int,
        relationship_id: str | None = None,
    ) -> tuple[EdgeView, ...]:
        """
        Retrieve all incoming edges to a concept.
        """
        stmt = q_incoming_edges(concept_id, relationship_id)
        return tuple(EdgeView(*row) for row in self.session.execute(stmt).all())

    @lru_cache(maxsize=500_000)
    def incoming_edges_batch(
        self,
        concept_ids: tuple[int, ...],
        relationship_id: str | None = None,
    ) -> tuple[EdgeView, ...]:
        """
        Retrieve all incoming edges for a batch of concepts.
        """
        stmt = q_incoming_edges_batch(concept_ids, relationship_id)
        return tuple(EdgeView(*row) for row in self.session.execute(stmt).all())

    def iter_edges(
        self,
        concept_id: int,
        *,
        direction: str = "out",
        predicate: Union[str, Predicate, None] = None,
        predicate_kinds: Optional[Set[PredicateKind]] = None,
        active_only: bool = True,
        on: Optional[date] = None,
        within_domain: bool = True,
    ) -> Iterable[EdgeView]:
        """
        Iterate over edges for a concept with filtering.

        Parameters
        ----------
        concept_id : int
            The source/target concept ID.
        direction : str
            'out' for outgoing, 'in' for incoming.
        predicate : str | Predicate, optional
            Filter by specific relationship ID.
        predicate_kinds : Set[PredicateKind], optional
            Filter by semantic kind of relationship.
        active_only : bool
            If True, return only valid/active edges.
        on : date, optional
            Check validity on a specific date.
        within_domain : bool
            If True, only return edges where source/target domains match.

        Yields
        -------
        EdgeView
            Edges matching criteria.
        """
        pred_id = _pred_id(predicate)

        edges = (
            self.outgoing_edges(concept_id, pred_id)
            if direction == "out"
            else self.incoming_edges(concept_id, pred_id)
        )

        for e in edges:
            if active_only and not is_active(
                e.valid_start_date,
                e.valid_end_date,
                e.invalid_reason,
                on=on,
            ):
                continue

            if within_domain and not self._same_domain(e):
                continue

            if predicate_kinds and (
                self.predicate_kind(e.predicate_id) not in predicate_kinds
            ):
                continue

            yield e

    def iter_edges_batch(
        self,
        concept_ids: tuple[int, ...],
        *,
        direction: str = "out",
        predicate: Union[str, Predicate, None] = None,
        predicate_kinds: Union[Set[PredicateKind], frozenset[PredicateKind], None] = None,
        active_only: bool = True,
        on: Optional[date] = None,
        within_domain: bool = True,
    ) -> Iterable[EdgeView]:
        """
        Iterate over edges for a batch of concepts with filtering.
        """
        if not concept_ids:
            return []

        pred_id = _pred_id(predicate)

        # 1. Fetch ALL raw edges for this batch from DB
        if direction == "out":
            edges = self.outgoing_edges_batch(concept_ids, pred_id)
        else:
            edges = self.incoming_edges_batch(concept_ids, pred_id)

        # 2. Filter them in memory (Python is fast enough for this)
        for e in edges:
            if active_only and not is_active(
                e.valid_start_date,
                e.valid_end_date,
                e.invalid_reason,
                on=on,
            ):
                continue

            if within_domain and not self._same_domain(e):
                continue

            if predicate_kinds and (
                self.predicate_kind(e.predicate_id) not in predicate_kinds
            ):
                continue

            yield e

    @lru_cache(maxsize=500_000)
    def parents(self, concept_id: int) -> tuple[int, ...]:
        """
        Retrieve parent Concept IDs (based on 'is_a' or 'subsumes' hierarchy).
        """
        return tuple(self.session.execute(q_parents(concept_id)).scalars())

    @lru_cache(maxsize=20_000)
    def roots(
        self, domain_id: str | None = None, vocabulary_id: str | None = None
    ) -> tuple[int, ...]:
        """
        Retrieve root concepts (no parents).
        """
        return tuple(
            self.session.execute(
                q_roots(domain_id=domain_id, vocabulary_id=vocabulary_id)
            ).scalars()
        )

    @lru_cache(maxsize=20_000)
    def leaves(
        self, domain_id: str | None = None, vocabulary_id: str | None = None
    ) -> tuple[int, ...]:
        """
        Retrieve leaf concepts (no children).
        """
        return tuple(
            self.session.execute(
                q_leaves(domain_id=domain_id, vocabulary_id=vocabulary_id)
            ).scalars()
        )

    @lru_cache(maxsize=20_000)
    def singletons(
        self, domain_id: str | None = None, vocabulary_id: str | None = None
    ) -> tuple[int, ...]:
        """
        Retrieve singleton concepts (no parents and no children).
        """
        return tuple(
            self.session.execute(
                q_singletons(domain_id=domain_id, vocabulary_id=vocabulary_id)
            ).scalars()
        )

    @lru_cache(maxsize=50_000)
    def synonyms_for_concept(self, concept_id: int) -> tuple[str, ...]:
        """
        Retrieve all synonyms for a concept.
        """
        rows = self.session.execute(q_concept_synonym_filtered(concept_id)).all()
        return tuple(row.concept_synonym_name for row in rows)

    def rollback_session(self) -> None:
        """
        Safely rollback the session if in a pending state.
        """
        try:
            self.session.rollback()
        except (PendingRollbackError, InvalidRequestError):
            pass

    def predicates(self) -> tuple[Predicate, ...]:
        """
        Return all predicates known to the knowledge graph.
        """
        rows = self.session.execute(q_all_predicates_with_ancestry()).all()
        return tuple(
            Predicate(
                relationship_id=row.relationship_id,
                name=row.relationship_name,
                reverse_id=row.reverse_relationship_id,
                is_hierarchical=bool(int(row.is_hierarchical)),
                anc_up=bool(int(row.anc_up)),
                anc_down=bool(int(row.anc_down)),
            )
            for row in rows
        )

    def predicate_summary(self) -> PredicateSummary:
        """
        Generate a summary of predicates grouped by kind.
        """
        groups: dict[PredicateKind, list[Predicate]] = defaultdict(list)

        for pred in self.predicates():
            kind = pred.classify_predicate()
            groups[kind].append(pred)

        return PredicateSummary(groups={k: tuple(v) for k, v in groups.items()})

    def get_all_concept_domain_ids(self) -> tuple[str, ...]:
        """
        Retrieve all distinct Domain IDs present in the concept table.
        """
        rows = self.session.execute(q_concept_domain_ids()).all()
        return tuple(row.domain_id for row in rows)

    def get_all_concept_vocabulary_ids(self) -> tuple[str, ...]:
        """
        Retrieve all distinct Vocabulary IDs present in the concept table.
        """
        rows = self.session.execute(q_concept_vocabulary_ids()).all()
        return tuple(row.vocabulary_id for row in rows)

    def get_potential_ancestor(
        self, child_id: int, parent_id: int
    ) -> Optional[AncestorMatch]:
        """
        Check if an ancestry relationship exists between a child and parent.
        """
        rows = self.session.execute(
            q_concept_potential_ancestor(child_id, parent_id)
        ).all()

        if not rows:
            return None
        else:
            if len(rows) > 1:
                logger.warning(
                    f"Multiple potential ancestor rows found for child_id={child_id} "
                    f"and parent_id={parent_id}. This should not happen. Returning the first match."
                )
            return AncestorMatch(
                ancestor_concept_id=rows[0].ancestor_concept_id,
                descendant_concept_id=rows[0].descendant_concept_id,
                min_levels_of_separation=rows[0].min_levels_of_separation,
            )

    def get_num_ancestors(self, concept_ids: tuple[int, ...]) -> Dict[int, int]:
        """
        Get the count of ancestors for a batch of concepts.
        """
        rows = self.session.execute(q_concept_num_ancestors(concept_ids)).all()
        return {row.concept_id: row.num_ancestors for row in rows}
    
    def get_embedding_model_table_name(self, model_name: str) -> Optional[str]:
        """
        Check if an embedding model exists in the database.

        Parameters
        ----------
        model_name : str
            The name of the embedding model to check.

        Returns
        -------
        Optional[str]
            The table name of the embedding model if it exists, None otherwise.
        """
        query = self.session.execute(q_embedding_model_table_name(model_name)).all()
        table_names = [row.table_name for row in query]
        if len(table_names) > 1:
            raise RuntimeError(f"Multiple embedding model tables found for model_name='{model_name}'. This should not happen. Returning the first match.")
        return table_names[0] if table_names else None
    
    def is_embedding_model_registered(self, model_name: str) -> bool:
        """
        Check if an embedding model is registered in the database.

        Parameters
        ----------
        model_name : str
            The name of the embedding model to check.

        Returns
        -------
        bool
            True if the model is registered, False otherwise.
        """
        return self.get_embedding_model_table_name(model_name) is not None
    
    def get_embedding_similarities(
        self,
        embedding_model_name: str,
        text_embedding: list[float],
        concept_ids: Optional[Tuple[int, ...]] = None
    ):
        
        embedding_table = _MODEL_CACHE.get(embedding_model_name)
        if embedding_table is None:
            raise ValueError(f"Embedding model '{embedding_model_name}' not found in cache. Make sure to initialize embedding tables at startup.")
        
        query = q_embedding_cosine_similarity(
            embedding_table=embedding_table,
            text_embedding=text_embedding,
            concept_ids=concept_ids
        )

        return {row.concept_id: row.similarity for row in self.session.execute(query).all()}
        



    def clear_caches(self) -> None:
        """
        Clear all LRU caches associated with the graph.
        """
        self.concept_view.cache_clear()
        self.concept_id_by_code.cache_clear()
        self._label_lookup_raw.cache_clear()
        self._synonym_lookup_raw.cache_clear()
        self.concept_ids_by_label.cache_clear()
        self.predicate.cache_clear()
        self.predicate_name.cache_clear()
        self.parents.cache_clear()
        self.outgoing_edges.cache_clear()
        self.incoming_edges.cache_clear()