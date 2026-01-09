from __future__ import annotations
from collections import defaultdict
import re
from datetime import date
from functools import lru_cache
from typing import Optional, Iterable, Tuple
from sqlalchemy.orm import Session
from sqlalchemy.exc import PendingRollbackError, InvalidRequestError

from .base import GraphBackend
from .edges import EdgeView, Predicate, PredicateKind, is_active, _pred_id
from .nodes import ConceptView, LabelMatch, LabelMatchKind

from omop_graph.db.session import safe_execute
from .queries import (
    q_concept_view,
    q_concept_id_by_code,
    q_predicate_row,
    q_predicate_name,
    q_outgoing_edges,
    q_incoming_edges,
    q_parents,
    q_concept_name_match,
    q_concept_name_ilike,
    q_concept_synonym_match,
    q_concept_synonym_ilike,
    q_roots,
    q_leaves,
    q_singletons,
    q_concept_synonym_filtered
)

"""
OMOP-backed graph facade.

Responsibilities:
- SQLAlchemy access
- caching
- predicate semantics
- edge / node retrieval
"""


class KnowledgeGraph(GraphBackend):

    def __init__(self, session: Session):
        self.session = session

    @lru_cache(maxsize=200_000)
    def concept_view(self, concept_id: int) -> ConceptView:
        row = self.session.execute(
            q_concept_view(concept_id)
        ).one()
        return ConceptView(*row)

    @lru_cache(maxsize=200_000)
    def concept_id_by_code(self, vocabulary_id: str, concept_code: str) -> int:
        return int(
            self.session.execute(
                q_concept_id_by_code(vocabulary_id, concept_code)
            ).scalar_one()
        )
    
    @lru_cache(maxsize=200_000)
    def synonym_lookup(self, label: str, fuzzy: bool = False) -> Tuple[LabelMatch, ...]:
        """
        Resolve a synonym label to concept_id(s).

        Returns matches annotated with LabelMatchKind for downstream explanations.
        """
        input_label = self._normalise_label(label)
        if not input_label:
            return ()
        
        cs = q_concept_synonym_ilike(input_label) if fuzzy else q_concept_synonym_match(input_label)

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
    def label_lookup(self, label: str, fuzzy: bool = False) -> Tuple[LabelMatch, ...]:
        """
        Resolve a label to concept_id(s), preferring Concept.concept_name matches.
        Returns matches annotated with LabelMatchKind for downstream explanations.
        """
        input_label = self._normalise_label(label)
        if not input_label:
            return ()
        
        cn = q_concept_name_ilike(input_label) if fuzzy else q_concept_name_match(input_label)

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
    def concept_ids_by_label(self, label: str) -> Tuple[int, ...]:
        rows = self.session.execute(
            q_concept_name_match(label)
        ).scalars()

        return tuple(rows)

    @lru_cache(maxsize=10_000)
    def predicate(self, relationship_id: str) -> Predicate:
        row = self.session.execute(
            q_predicate_row(relationship_id)
        ).one()

        return Predicate(
            relationship_id=row.relationship_id,
            name=row.relationship_name,
            reverse_id=row.reverse_relationship_id,
            is_hierarchical=bool(row.is_hierarchical),
            defines_ancestry=bool(row.defines_ancestry),
        )

    @lru_cache(maxsize=10_000)
    def predicate_name(self, relationship_id: str) -> str:
        return self.session.execute(
            q_predicate_name(relationship_id)
        ).scalar_one()

    def predicate_kind(self, relationship_id: str) -> PredicateKind:
        return self.predicate(relationship_id).classify_predicate(kg=self)

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
        ).scalars()

        return tuple(rows)

    def rollback_session(self) -> None:
        try:
            self.session.rollback()
        except (PendingRollbackError, InvalidRequestError):
            pass

    def clear_caches(self) -> None:
        self.concept_view.cache_clear()
        self.concept_id_by_code.cache_clear()
        self.label_lookup.cache_clear()
        self.concept_ids_by_label.cache_clear()
        self.predicate.cache_clear()
        self.predicate_name.cache_clear()
        self.parents.cache_clear()
        self.outgoing_edges.cache_clear()
        self.incoming_edges.cache_clear()