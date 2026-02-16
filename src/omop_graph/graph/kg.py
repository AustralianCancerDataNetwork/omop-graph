from __future__ import annotations
from collections import defaultdict
import re
from datetime import date
from functools import lru_cache
from typing import Optional, Iterable, Tuple
from sqlalchemy.orm import Session
from sqlalchemy.exc import PendingRollbackError, InvalidRequestError

from .base import GraphBackend
from .edges import EdgeView, Predicate, PredicateKind, is_active, PredicateSummary
from .nodes import ConceptView, LabelMatch, LabelMatchKind, LabelMatchGroupView, AncestorMatch

from omop_graph.db.session import safe_execute
from .queries import (
    q_concept_view,
    q_concept_views,
    q_concept_id_by_code,
    q_predicate_row,
    q_predicate_row_with_ancestry,
    q_predicate_name,
    q_outgoing_edges,
    q_incoming_edges,
    q_outgoing_edges_batch,
    q_incoming_edges_batch,
    q_parents,
    q_concept_name_match,
    q_concept_name_ilike,
    q_concept_synonym_match,
    q_concept_synonym_ilike,
    q_roots,
    q_leaves,
    q_singletons,
    q_concept_synonym_filtered,
    q_all_predicates,
    q_all_predicates_with_ancestry,
    q_concept_domain_ids,
    q_concept_vocabulary_ids,
    q_concept_potential_ancestor,
    q_concept_num_ancestors,
    q_concept_name_fulltext,
    q_concept_synonym_fulltext,
)
from .constraints import SearchConstraintConcept

import logging
logger = logging.getLogger(__name__)

"""
OMOP-backed graph facade.

Responsibilities:
- SQLAlchemy access
- caching
- predicate semantics
- edge / node retrieval
"""

def _pred_id(pred: Predicate | str | None) -> str | None:
    if pred is None:
        return None
    if isinstance(pred, Predicate):
        return pred.relationship_id
    if isinstance(pred, str):
        return pred
    raise TypeError(f"Unsupported predicate type: {type(pred)}")

class KnowledgeGraph(GraphBackend):

    def __init__(self, session: Session):
        self.session = session

    @lru_cache(maxsize=200_000)
    def concept_view(self, concept_id: int) -> ConceptView:
        row = self.session.execute(
            q_concept_view(concept_id)
        ).one()
        return ConceptView.from_row(row)
    
    @lru_cache(maxsize=200_000)
    def concept_views(self, concept_ids: tuple[int, ...]) -> tuple[ConceptView, ...]:
        return tuple(
            ConceptView.from_row(row)
            for row in self.session.execute(
                q_concept_views(concept_ids)
            ).all()
        )

    @lru_cache(maxsize=200_000)
    def concept_id_by_code(self, vocabulary_id: str, concept_code: str) -> int:
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
        search_constraint: Optional[SearchConstraintConcept] = None
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
        search_constraint: Optional[SearchConstraintConcept] = None
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
        search_constraint: Optional[SearchConstraintConcept] = None
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
        search_constraint: Optional[SearchConstraintConcept] = None
    ) -> LabelMatchGroupView:
        """
        Resolve a synonym label to grouped concept matches.
        """
        raw = self._synonym_lookup_raw(label, fuzzy=fuzzy, search_constraint=search_constraint)
        if sort:
            raw = sorted(raw)
        return LabelMatchGroupView.from_matches(raw)

    def label_lookup(
        self, 
        label: str, 
        fuzzy: bool = False, 
        sort: bool = True,
        search_constraint: Optional[SearchConstraintConcept] = None
    ) -> LabelMatchGroupView:
        """
        Resolve a label to grouped concept matches, preferring direct name matches.
        """
        raw = self._label_lookup_raw(label, fuzzy=fuzzy, search_constraint=search_constraint)
        if sort:
            raw = sorted(raw)
        return LabelMatchGroupView.from_matches(raw)
    
    def fulltext_lookup(
        self,
        label: str,
        sort: bool = True,
        fuzzy: bool = False,
        search_constraint: Optional[SearchConstraintConcept] = None
    ) -> LabelMatchGroupView:
        """
        Resolve a label using fulltext search (bag of words, ignoring word order).
        """
        raw = self._fulltext_lookup_raw(label, fuzzy=fuzzy, search_constraint=search_constraint)
        if sort:
            raw = sorted(raw)

        return LabelMatchGroupView.from_matches(raw)


    @lru_cache(maxsize=200_000)
    def concept_ids_by_label(self, label: str) -> Tuple[int, ...]:
        rows = self.session.execute(
            q_concept_name_match(label)
        ).scalars()

        return tuple(rows)

    @lru_cache(maxsize=10_000)
    def predicate(self, relationship_id: str) -> Predicate:
        row = self.session.execute(
            q_predicate_row_with_ancestry(relationship_id)
        ).one()

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
        return self.session.execute(
            q_predicate_name(relationship_id)
        ).scalar_one()

    def predicate_kind(self, relationship_id: str) -> PredicateKind:
        return self.predicate(relationship_id).classify_predicate()
    
    def predicate_kinds(self, relationship_ids: tuple[str, ...]) -> Tuple[PredicateKind, ...]:
        return tuple(
            self.predicate_kind(rel_id)
            for rel_id in relationship_ids
        )

    def reverse_predicate_id(self, relationship_id: str) -> Optional[str]:
        return self.predicate(relationship_id).reverse_id
    
    def _normalise_label(self, s: str) -> str:
        return re.sub(r"\s+", " ", s.strip().lower())

    @lru_cache(maxsize=500_000)
    def outgoing_edges(
        self,
        concept_id: int,
        relationship_id: str | None = None,
    ) -> tuple[EdgeView, ...]:

        stmt = q_outgoing_edges(concept_id, relationship_id)

        return tuple(
            EdgeView(*row)
            for row in self.session.execute(stmt).all()
        )
    
    @lru_cache(maxsize=500_000)
    def outgoing_edges_batch(
        self,
        concept_ids: tuple[int, ...],
        relationship_id: str | None = None,
    ) -> tuple[EdgeView, ...]:

        stmt = q_outgoing_edges_batch(concept_ids, relationship_id)

        return tuple(
            EdgeView(*row)
            for row in self.session.execute(stmt).all()
        )
    
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
        subj = self.concept_view(e.subject_id)
        obj = self.concept_view(e.object_id)
        return subj.domain_id == obj.domain_id

    @lru_cache(maxsize=500_000)
    def incoming_edges(
        self,
        concept_id: int,
        relationship_id: str | None = None,
    ) -> tuple[EdgeView, ...]:

        stmt = q_incoming_edges(concept_id, relationship_id)

        return tuple(
            EdgeView(*row)
            for row in self.session.execute(stmt).all()
        )
    
    @lru_cache(maxsize=500_000)
    def incoming_edges_batch(
        self,
        concept_ids: tuple[int, ...],
        relationship_id: str | None = None,
    ) -> tuple[EdgeView, ...]:

        stmt = q_incoming_edges_batch(concept_ids, relationship_id)

        return tuple(
            EdgeView(*row)
            for row in self.session.execute(stmt).all()
        )
    
    def iter_edges(
        self,
        concept_id: int,
        *,
        direction: str = "out",
        predicate=None,
        predicate_kinds: set[PredicateKind] | None = None,
        active_only: bool = True,
        on: date | None = None,
        within_domain: bool = True,
    )  -> Iterable[EdgeView]:
        
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
        predicate=None,
        predicate_kinds: set[PredicateKind] | frozenset[PredicateKind] | None = None,
        active_only: bool = True,
        on: date | None = None,
        within_domain: bool = True,
    ) -> Iterable[EdgeView]:
        
        if not concept_ids:
            return []

        pred_id = _pred_id(predicate)

        # 1. Fetch ALL raw edges for this batch from DB
        # We assume you implement these helpers using the SQL you shared earlier
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
        return tuple(
            self.session.execute(
                q_parents(concept_id)
            ).scalars()
        )
    
    @lru_cache(maxsize=20_000)
    def roots(self, domain_id: str | None = None, vocabulary_id: str | None = None) -> tuple[int, ...]:
        return tuple(
            self.session.execute(
                q_roots(domain_id=domain_id, vocabulary_id=vocabulary_id)
            ).scalars()
        )
    
    @lru_cache(maxsize=20_000)
    def leaves(self, domain_id: str | None = None, vocabulary_id: str | None = None) -> tuple[int, ...]:
        return tuple(
            self.session.execute(
                q_leaves(domain_id=domain_id, vocabulary_id=vocabulary_id)
            ).scalars()
        )

    @lru_cache(maxsize=20_000)
    def singletons(self, domain_id: str | None = None, vocabulary_id: str | None = None) -> tuple[int, ...]:
        return tuple(
            self.session.execute(
                q_singletons(domain_id=domain_id, vocabulary_id=vocabulary_id)
            ).scalars()
        )

    @lru_cache(maxsize=50_000)
    def synonyms_for_concept(self, concept_id: int) -> tuple[str, ...]:
        rows = self.session.execute(
            q_concept_synonym_filtered(concept_id)
        ).all()

        return tuple(row.concept_synonym_name for row in rows)

    def rollback_session(self) -> None:
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
        groups: dict[PredicateKind, list[Predicate]] = defaultdict(list)

        for pred in self.predicates():   
            kind = pred.classify_predicate()
            groups[kind].append(pred)

        return PredicateSummary(
            groups={k: tuple(v) for k, v in groups.items()}
        )
    
    def get_all_concept_domain_ids(self) -> tuple[str, ...]:
        rows = self.session.execute(q_concept_domain_ids()).all()
        return tuple(row.domain_id for row in rows)
    
    def get_all_concept_vocabulary_ids(self) -> tuple[str, ...]:
        rows = self.session.execute(q_concept_vocabulary_ids()).all()
        return tuple(row.vocabulary_id for row in rows)
    
    def get_potential_ancestor(self, child_id: int, parent_id: int) -> Optional[AncestorMatch]:
        rows = self.session.execute(q_concept_potential_ancestor(child_id, parent_id)).all()

        if not rows:
            return None
        else:
            if len(rows) > 1:
                logger.warning(f"Multiple potential ancestor rows found for child_id={child_id} and parent_id={parent_id}. This should not happen. Returning the first match.")
            return AncestorMatch( 
                ancestor_concept_id=rows[0].ancestor_concept_id, 
                descendant_concept_id=rows[0].descendant_concept_id,
                min_levels_of_separation=rows[0].min_levels_of_separation
            )

        
    def get_num_ancestors(self, concept_ids: tuple[int, ...]) -> dict[int, int]:
        rows = self.session.execute(q_concept_num_ancestors(concept_ids)).all()
        return {row.concept_id: row.num_ancestors for row in rows}

    def clear_caches(self) -> None:
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