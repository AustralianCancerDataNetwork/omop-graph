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

# IMPORTANT: The lru_cache has access to self in each cache. We need to avoid this if we use it
# TODO: Get rid of the LRU cache and instead optimise the queries!

from __future__ import annotations

import logging
import re
from collections import defaultdict
from datetime import date
from functools import lru_cache
from typing import Dict, Iterable, Optional, Set, Tuple, Union, Literal, Generator

from sqlalchemy.exc import InvalidRequestError, PendingRollbackError
from sqlalchemy.orm import Session, sessionmaker


# Local Application Imports
from ..extensions.emb import MissingExtensionError
from ..extensions.omop_alchemy import ClassIDEnum, RelationshipCache, validate_mapping_table
from .base import GraphBackend
from .constraints import SearchConstraintConcept
from .edges import EdgeView, Predicate, is_active
from .nodes import (
    AncestorMatch,
    ConceptView,
    LabelMatch,
    LabelMatchGroupView,
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
    q_concept_synonym_filtered,
    q_concept_synonym_fulltext,
    q_concept_synonym_ilike,
    q_concept_synonym_match,
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
    q_relationships
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

    def __init__(self, session_factory: sessionmaker):
        self.session_factory = session_factory

        # Populate the relationshipcache 
        with self.session_factory() as session:
            RelationshipCache.load(session)

    @property
    def emb(self):
        """Namespace for all embedding operations."""
        try:
            from omop_emb.accessor import EmbeddingAccessor
            return EmbeddingAccessor()
        except ImportError:
            raise MissingExtensionError(
                "Accessing '.emb' requires the 'omop-emb' package. "
                "Install via: pip install omop-graph[embeddings]"
            )

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
        with self.session_factory() as session:
            row = session.execute(q_concept_view(concept_id)).one()
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
        with self.session_factory() as session:
            concept_views = tuple(
                ConceptView.from_row(row)
                for row in session.execute(q_concept_views(concept_ids))
            )
        return concept_views

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
        with self.session_factory() as session:
            concept_id = int(
                session.execute(
                    q_concept_id_by_code(vocabulary_id, concept_code)
                ).scalar_one()
            )
        return concept_id

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

        with self.session_factory() as session:
            matches = tuple(
                LabelMatch(
                    input_label=input_label,
                    matched_label=name,
                    concept_id=int(cid),
                    match_kind=LabelMatchKind.SYNONYM,
                    is_standard=is_standard,
                    is_active=is_active,
                )
                for cid, name, is_standard, is_active in session.execute(cs)
            )
        return matches

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

        with self.session_factory() as session:
            matches = tuple(
                LabelMatch(
                    input_label=input_label,
                    matched_label=name,
                    concept_id=int(cid),
                    match_kind=LabelMatchKind.DIRECT,
                    is_standard=is_standard,
                    is_active=is_active,
                )
                for cid, name, is_standard, is_active in session.execute(cn)
            )
        return matches

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

        with self.session_factory() as session:
            matches = tuple(
                LabelMatch(
                    input_label=label,
                    matched_label=name,
                    concept_id=int(cid),
                    match_kind=LabelMatchKind.FULLTEXT,
                    is_standard=is_standard,
                    is_active=is_active,
                )
                for cid, name, is_standard, is_active in session.execute(q(label, search_constraint=search_constraint))
            )
        return matches

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

        with self.session_factory() as session:
            rows = session.execute(q_concept_name_match(label)).scalars()
        return tuple(rows)

    @lru_cache(maxsize=10_000)
    @validate_mapping_table
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
            anc_down=bool(int(row.anc_up)),
            class_id=ClassIDEnum(row.class_id),
            subclass_id=row.subclass_id
        )

    @lru_cache(maxsize=10_000)
    def predicate_name(self, relationship_id: str) -> str:
        """
        Retrieve the human-readable name of a relationship.
        """
        # TODO: Not really necessary. The "ID" is mostly human-readable anyways.
        with self.session_factory() as session:
            predicate_name = session.execute(q_predicate_name(relationship_id)).scalar_one()
        return predicate_name

    def predicate_kind(self, relationship_id: str) -> ClassIDEnum:
        """
        Classify the predicate into a semantic kind.
        """
        try:
            return RelationshipCache.get(relationship_id).class_id
        except AttributeError as e:
            raise AttributeError(e)
    
    def predicate_kinds(
        self, relationship_ids: tuple[str, ...]
    ) -> Tuple[ClassIDEnum, ...]:
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
        Tuple[CURIE, PRED_CURIE, CURIE]
            Triples (subject, predicate, object).
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
        


        for s,p,o in session.execute(
            q_relationships(
                subjects=objects,
                predicates=predicates,
                objects=subjects,
            )
        ):
            yield o, p, s


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
       
    @lru_cache(maxsize=1_000_000)
    def edges(
        self,
        concept_ids: Tuple[int, ...] | int,
        direction: Literal["in", "out"],
        predicate_ids: Optional[frozenset[str]] = None,
        predicate_kinds: Optional[frozenset[ClassIDEnum]] = None,
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
        predicate_kinds : Set[ClassIDEnum], optional
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
                    within_domain=within_domain)
                )
        return edges

    @validate_mapping_table
    def iter_edges(
        self,
        session: Session,
        concept_ids: int | tuple[int, ...],
        direction: Literal["in", "out"],
        predicate_ids: Optional[frozenset[str]] = None,
        predicate_kinds: Optional[frozenset[ClassIDEnum]] = None,
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
            within_domain=within_domain
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

    @lru_cache(maxsize=500_000)
    def parents(self, concept_id: int) -> tuple[int, ...]:
        """
        Retrieve parent Concept IDs of concept using Concept_Ancestor table.
        """
        with self.session_factory() as session:
            parents = tuple(session.execute(q_parents(concept_id)).scalars())
        return parents

    @lru_cache(maxsize=500_000)
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
            filter_obsoletes=filter_obsoletes
        )
        
        for row in session.execute(query):
            yield int(row.concept_id)
        

    @lru_cache(maxsize=20_000)
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

    @lru_cache(maxsize=20_000)
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

    @lru_cache(maxsize=20_000)
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

    @lru_cache(maxsize=50_000)
    def synonyms_for_concept(self, concept_id: int) -> tuple[str, ...]:
        """
        Retrieve all synonyms for a concept.
        """
        with self.session_factory() as session:
            rows = session.execute(q_concept_synonym_filtered(concept_id)).all()
        return tuple(row.concept_synonym_name for row in rows)

    def rollback_session(self) -> None:
        """
        Safely rollback the session if in a pending state.
        """
        try:
            with self.session_factory() as session:
                session.rollback()
        except (PendingRollbackError, InvalidRequestError):
            pass
    
    @validate_mapping_table
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
                class_id=ClassIDEnum(row.class_id),
                subclass_id=row.subclass_id
            )
            for row in rows
        )

    def get_all_concept_domain_ids(self) -> tuple[str, ...]:
        """
        Retrieve all distinct Domain IDs present in the concept table.
        """
        with self.session_factory() as session:
            rows = session.execute(q_concept_domain_ids()).all()
        return tuple(row.domain_id for row in rows)

    def get_all_concept_vocabulary_ids(self) -> tuple[str, ...]:
        """
        Retrieve all distinct Vocabulary IDs present in the concept table.
        """
        with self.session_factory() as session:
            rows = session.execute(q_concept_vocabulary_ids()).all()
        return tuple(row.vocabulary_id for row in rows)

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

    def get_num_ancestors(self, concept_ids: tuple[int, ...]) -> Dict[int, int]:
        """
        Get the count of ancestors for a batch of concepts.
        """
        with self.session_factory() as session:
            rows = session.execute(q_concept_num_ancestors(concept_ids)).all()
        return {row.concept_id: row.num_ancestors for row in rows}


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
        self.edges.cache_clear()