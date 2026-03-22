"""
Batch orchestration for semantic grounding.

This module provides ``GroundingBatchRunner``, a staged wrapper around the
existing grounding semantics. It is designed for workloads where many query
texts are grounded under the same constraint set and we want to avoid
repeating the most expensive lookup-heavy steps:

1. batch query embedding generation
2. resolver hit collection per query
3. candidate -> standard concept standardization cached across the batch
4. ancestor validation against the parent pool in bulk
5. grounded concept embedding backfill in bulk
6. final per-query scoring using the existing ranking function
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping, Optional, Sequence

import numpy as np
from omop_alchemy.cdm.model.vocabulary import Concept_Ancestor
from sqlalchemy import and_, select

from omop_graph.extensions.emb import MissingExtensionError
from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.graph.paths import StandardConcept, get_unique_standard_concepts
from omop_graph.graph.scoring import StandardConceptWithScore, _score_standard_concept
from omop_graph.reasoning.grounding import (
    GroundingConstraints,
    _grounded_concept_matches_search_constraint,
)
from omop_graph.reasoning.resolvers import CandidateHit, ResolverConfidence, ResolverPipeline
from omop_llm import LLMClient

if TYPE_CHECKING:
    from omop_emb import EmbeddingService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GroundingBatchResult:
    """Batch-grounding output for one input text."""

    text: str
    resolved_hits: tuple[CandidateHit, ...]
    grounded_candidates: tuple[StandardConcept, ...]
    ranked_concepts: tuple[StandardConceptWithScore, ...]
    text_embedding: Optional[np.ndarray] = None

    @property
    def best(self) -> Optional[StandardConceptWithScore]:
        return self.ranked_concepts[0] if self.ranked_concepts else None


@dataclass(frozen=True)
class _StandardizationTemplate:
    concept_id: int
    concept_name: str
    separation: int
    hierarchy_cost: float
    mapped_from_non_standard: bool


@dataclass
class GroundingBatchRunner:
    """
    Run grounding over many queries while reusing shared intermediate work.

    The runner intentionally keeps the existing single-query ``ground_term``
    behavior as the semantic reference point. It improves throughput mainly by
    caching candidate standardization and batching ancestor validation and
    grounded-concept embedding backfill.
    """

    resolver_pipeline: ResolverPipeline
    kg: KnowledgeGraph
    constraints: GroundingConstraints
    text_embedding_model: Optional[str]
    embedding_client: Optional[LLMClient] = None
    embedding_service: Optional["EmbeddingService"] = None
    limit_per_resolver: Optional[int] = None
    max_candidates: Optional[int] = None
    embedding_batch_size: Optional[int] = None
    _candidate_name_cache: dict[int, str] = field(default_factory=dict, init=False, repr=False)
    _standardization_cache: dict[int, tuple[_StandardizationTemplate, ...]] = field(default_factory=dict, init=False, repr=False)
    _best_parent_separation_cache: dict[int, int] = field(default_factory=dict, init=False, repr=False)

    def ground_texts(
        self,
        texts: Sequence[str],
        *,
        query_embeddings: Optional[Mapping[str, np.ndarray]] = None,
    ) -> list[GroundingBatchResult]:
        """
        Ground many texts under a shared set of constraints.

        Parameters
        ----------
        texts : sequence[str]
            Input query texts. Duplicate texts are allowed and share work.
        query_embeddings : mapping[str, np.ndarray], optional
            Optional precomputed query embeddings keyed by text. Values may be
            1D or 2D; they are normalized to shape ``(1, d)`` internally.
        """
        ordered_texts = [text for text in texts if text]
        if not ordered_texts:
            return []

        search_constraint = self.constraints.search_constraint
        resolver_search_constraint = self._resolver_search_constraint()
        if search_constraint is not None:
            search_constraint.check(self.kg)
        if (
            resolver_search_constraint is not None
            and resolver_search_constraint is not search_constraint
        ):
            resolver_search_constraint.check(self.kg)

        unique_texts = tuple(dict.fromkeys(ordered_texts))
        embedding_by_text = self._prepare_query_embeddings(
            unique_texts,
            query_embeddings=query_embeddings,
        )
        hits_by_text = self._resolve_hits(
            unique_texts,
            resolver_search_constraint=resolver_search_constraint,
            query_embeddings=embedding_by_text,
        )
        grounded_by_text = self._ground_hits_batch(hits_by_text)
        self._backfill_grounded_concept_embeddings(grounded_by_text)
        num_ancestors = self._batch_num_ancestors(grounded_by_text)

        results_by_text: dict[str, GroundingBatchResult] = {}
        try:
            with self.kg.session_factory() as session:
                for text in unique_texts:
                    grounded = grounded_by_text[text]
                    similarity_scores = self._similarity_scores_for_text(
                        session=session,
                        query_embedding=embedding_by_text.get(text),
                        concept_ids=tuple(sc.concept_id for sc in grounded),
                    )
                    ranked = tuple(
                        sorted(
                            (
                                _score_standard_concept(
                                    kg=self.kg,
                                    text=text,
                                    standard_concept=standard_concept,
                                    num_ancestors=num_ancestors.get(standard_concept.concept_id, 0),
                                    similarity_score=(
                                        similarity_scores.get(standard_concept.concept_id)
                                        if similarity_scores is not None
                                        else None
                                    ),
                                )
                                for standard_concept in grounded
                            ),
                            key=lambda scored: scored.total_score,
                            reverse=True,
                        )
                    )
                    if self.max_candidates is not None:
                        ranked = ranked[: self.max_candidates]
                    results_by_text[text] = GroundingBatchResult(
                        text=text,
                        resolved_hits=hits_by_text[text],
                        grounded_candidates=grounded,
                        ranked_concepts=ranked,
                        text_embedding=embedding_by_text.get(text),
                    )
        except MissingExtensionError:
            logger.info(
                "Embedding-based scoring not available during batch grounding. "
                "Proceeding with lexical scoring only."
            )
            for text in unique_texts:
                grounded = grounded_by_text[text]
                ranked = tuple(
                    sorted(
                        (
                            _score_standard_concept(
                                kg=self.kg,
                                text=text,
                                standard_concept=standard_concept,
                                num_ancestors=num_ancestors.get(standard_concept.concept_id, 0),
                                similarity_score=None,
                            )
                            for standard_concept in grounded
                        ),
                        key=lambda scored: scored.total_score,
                        reverse=True,
                    )
                )
                if self.max_candidates is not None:
                    ranked = ranked[: self.max_candidates]
                results_by_text[text] = GroundingBatchResult(
                    text=text,
                    resolved_hits=hits_by_text[text],
                    grounded_candidates=grounded,
                    ranked_concepts=ranked,
                    text_embedding=embedding_by_text.get(text),
                )

        return [results_by_text[text] for text in ordered_texts]

    def _resolver_search_constraint(self):
        return (
            self.constraints.resolver_search_constraint
            if self.constraints.resolver_search_constraint is not None
            else self.constraints.search_constraint
        )

    def _prepare_query_embeddings(
        self,
        texts: Sequence[str],
        *,
        query_embeddings: Optional[Mapping[str, np.ndarray]],
    ) -> dict[str, np.ndarray]:
        embeddings_by_text: dict[str, np.ndarray] = {}
        if query_embeddings is not None:
            for text, embedding in query_embeddings.items():
                embeddings_by_text[text] = self._normalize_query_embedding(embedding)

        missing_texts = [text for text in texts if text not in embeddings_by_text]
        if not missing_texts:
            return embeddings_by_text
        if self.text_embedding_model is None or self.embedding_client is None:
            return embeddings_by_text

        service = self.embedding_service or self.kg.emb_service
        batch_embeddings = service.embed_texts(
            list(missing_texts),
            embedding_client=self.embedding_client,
            batch_size=self.embedding_batch_size,
        ).astype(np.float32, copy=False)
        for text, embedding in zip(missing_texts, batch_embeddings):
            embeddings_by_text[text] = self._normalize_query_embedding(embedding)
        return embeddings_by_text

    def _resolve_hits(
        self,
        texts: Sequence[str],
        *,
        resolver_search_constraint,
        query_embeddings: Mapping[str, np.ndarray],
    ) -> dict[str, tuple[CandidateHit, ...]]:
        hits_by_text: dict[str, tuple[CandidateHit, ...]] = {}
        for text in texts:
            hits_by_text[text] = tuple(
                self.resolver_pipeline.resolve(
                    self.kg,
                    text,
                    limit_per_resolver=self.limit_per_resolver,
                    constraints=resolver_search_constraint,
                    text_embedding=query_embeddings.get(text),
                    text_embedding_model=self.text_embedding_model,
                    embedding_client=self.embedding_client,
                    embedding_service=self.embedding_service,
                )
            )
        return hits_by_text

    def _ground_hits_batch(
        self,
        hits_by_text: Mapping[str, tuple[CandidateHit, ...]],
    ) -> dict[str, tuple[StandardConcept, ...]]:
        unique_candidate_ids = tuple(
            dict.fromkeys(
                hit.concept_id
                for hits in hits_by_text.values()
                for hit in hits
            )
        )
        if unique_candidate_ids:
            self._ensure_candidate_names(unique_candidate_ids)
            self._populate_standardization_cache(unique_candidate_ids)

        grounded_by_text: dict[str, tuple[StandardConcept, ...]] = {}
        for text, hits in hits_by_text.items():
            grounded: list[StandardConcept] = []
            for hit in hits:
                grounded.extend(self._materialize_standard_concepts(hit))
            grounded = get_unique_standard_concepts(grounded)
            if self.constraints.search_constraint is not None:
                grounded = [
                    standard_concept
                    for standard_concept in grounded
                    if _grounded_concept_matches_search_constraint(
                        self.kg,
                        standard_concept.concept_id,
                        self.constraints.search_constraint,
                    )
                ]
            grounded_by_text[text] = tuple(grounded)
        return grounded_by_text

    def _ensure_candidate_names(self, concept_ids: Sequence[int]) -> None:
        missing_ids = tuple(
            concept_id
            for concept_id in concept_ids
            if concept_id not in self._candidate_name_cache
        )
        if not missing_ids:
            return
        for concept_view in self.kg.concept_views(tuple(missing_ids)):
            self._candidate_name_cache[concept_view.concept_id] = concept_view.concept_name

    def _populate_standardization_cache(self, candidate_ids: Sequence[int]) -> None:
        missing_candidate_ids = tuple(
            candidate_id
            for candidate_id in candidate_ids
            if candidate_id not in self._standardization_cache
        )
        if not missing_candidate_ids:
            return

        candidate_views = {
            concept_view.concept_id: concept_view
            for concept_view in self.kg.concept_views(tuple(missing_candidate_ids))
        }
        constrained = self.constraints.parent_ids is not None

        raw_templates_by_candidate: dict[int, list[tuple[int, str, float, bool]]] = defaultdict(list)
        standard_candidate_ids = [
            candidate_id
            for candidate_id, concept_view in candidate_views.items()
            if concept_view.standard_concept
        ]
        for candidate_id in standard_candidate_ids:
            concept_view = candidate_views[candidate_id]
            raw_templates_by_candidate[candidate_id].append(
                (concept_view.concept_id, concept_view.concept_name, 0.0, False)
            )

        mapped_candidate_ids = tuple(
            candidate_id
            for candidate_id, concept_view in candidate_views.items()
            if not concept_view.standard_concept
        )
        if mapped_candidate_ids:
            with self.kg.session_factory() as session:
                edges = tuple(
                    self.kg.iter_edges(
                        session=session,
                        concept_ids=mapped_candidate_ids,
                        direction="out",
                        predicate_kinds=self.constraints.predicate_kinds,
                    )
                )
            target_ids = tuple(dict.fromkeys(edge.object_id for edge in edges))
            target_views = (
                {
                    concept_view.concept_id: concept_view
                    for concept_view in self.kg.concept_views(target_ids)
                }
                if target_ids
                else {}
            )
            seen_targets: dict[int, set[int]] = defaultdict(set)
            for edge in edges:
                target_view = target_views.get(edge.object_id)
                if target_view is None or not target_view.standard_concept:
                    continue
                if edge.object_id in seen_targets[edge.subject_id]:
                    continue
                seen_targets[edge.subject_id].add(edge.object_id)
                raw_templates_by_candidate[edge.subject_id].append(
                    (target_view.concept_id, target_view.concept_name, 0.0, True)
                )

        standard_target_ids = tuple(
            dict.fromkeys(
                concept_id
                for templates in raw_templates_by_candidate.values()
                for concept_id, _, _, _ in templates
            )
        )
        best_parent_separation = (
            self._get_best_parent_separation(standard_target_ids)
            if constrained
            else {}
        )

        for candidate_id in missing_candidate_ids:
            templates: list[_StandardizationTemplate] = []
            for concept_id, concept_name, hierarchy_cost, mapped_from_non_standard in raw_templates_by_candidate.get(candidate_id, []):
                if constrained:
                    separation = best_parent_separation.get(concept_id)
                    if separation is None or separation > self.constraints.max_depth:
                        continue
                else:
                    separation = 0
                templates.append(
                    _StandardizationTemplate(
                        concept_id=concept_id,
                        concept_name=concept_name,
                        separation=separation,
                        hierarchy_cost=hierarchy_cost,
                        mapped_from_non_standard=mapped_from_non_standard,
                    )
                )
                if self.constraints.parent_ids is None and len(templates) >= 3:
                    break
            self._standardization_cache[candidate_id] = tuple(templates)

    def _get_best_parent_separation(
        self,
        standard_concept_ids: Sequence[int],
    ) -> dict[int, int]:
        missing_ids = tuple(
            concept_id
            for concept_id in standard_concept_ids
            if concept_id not in self._best_parent_separation_cache
        )
        if missing_ids and self.constraints.parent_ids is not None:
            stmt = (
                select(
                    Concept_Ancestor.descendant_concept_id,
                    Concept_Ancestor.min_levels_of_separation,
                )
                .where(
                    and_(
                        Concept_Ancestor.ancestor_concept_id.in_(self.constraints.parent_ids),
                        Concept_Ancestor.descendant_concept_id.in_(missing_ids),
                        Concept_Ancestor.min_levels_of_separation > 1,
                        Concept_Ancestor.min_levels_of_separation <= self.constraints.max_depth,
                    )
                )
            )
            with self.kg.session_factory() as session:
                for descendant_concept_id, min_levels_of_separation in session.execute(stmt):
                    best = self._best_parent_separation_cache.get(int(descendant_concept_id))
                    separation = int(min_levels_of_separation)
                    if best is None or separation < best:
                        self._best_parent_separation_cache[int(descendant_concept_id)] = separation
        return {
            concept_id: self._best_parent_separation_cache[concept_id]
            for concept_id in standard_concept_ids
            if concept_id in self._best_parent_separation_cache
        }

    def _materialize_standard_concepts(self, hit: CandidateHit) -> list[StandardConcept]:
        original_name = self._candidate_name_cache[hit.concept_id]
        materialized: list[StandardConcept] = []
        for template in self._standardization_cache.get(hit.concept_id, ()):
            materialized.append(
                StandardConcept(
                    concept_id=template.concept_id,
                    concept_name=template.concept_name,
                    separation=template.separation,
                    original_id=hit.concept_id,
                    original_name=original_name,
                    matched_label=hit.matched_label,
                    resolver_confidence=(
                        ResolverConfidence.PARTIAL
                        if template.mapped_from_non_standard
                        else hit.resolver_confidence
                    ),
                    hierarchy_cost=template.hierarchy_cost,
                )
            )
        return materialized

    def _batch_num_ancestors(
        self,
        grounded_by_text: Mapping[str, tuple[StandardConcept, ...]],
    ) -> dict[int, int]:
        concept_ids = tuple(
            dict.fromkeys(
                standard_concept.concept_id
                for grounded in grounded_by_text.values()
                for standard_concept in grounded
            )
        )
        return self.kg.get_num_ancestors(concept_ids) if concept_ids else {}

    def _backfill_grounded_concept_embeddings(
        self,
        grounded_by_text: Mapping[str, tuple[StandardConcept, ...]],
    ) -> None:
        if self.text_embedding_model is None or self.embedding_client is None:
            return

        concept_id_to_name = {
            standard_concept.concept_id: standard_concept.concept_name
            for grounded in grounded_by_text.values()
            for standard_concept in grounded
        }
        if not concept_id_to_name:
            return

        try:
            with self.kg.session_factory() as session:
                if not self.kg.emb.is_model_registered(
                    session=session,
                    model_name=self.text_embedding_model,
                ):
                    return
                service = self.embedding_service or self.kg.emb_service
                missing_concept_ids = service.get_missing_concept_ids(
                    session=session,
                    model_name=self.text_embedding_model,
                    concept_ids=tuple(concept_id_to_name),
                )
                if not missing_concept_ids:
                    return
                service.embed_and_upsert_concepts(
                    session=session,
                    model_name=self.text_embedding_model,
                    concept_ids=missing_concept_ids,
                    concept_texts=tuple(
                        concept_id_to_name[concept_id]
                        for concept_id in missing_concept_ids
                    ),
                    embedding_client=self.embedding_client,
                    batch_size=self.embedding_batch_size,
                )
        except MissingExtensionError:
            logger.info(
                "Embedding extension unavailable during batch grounding backfill."
            )

    def _similarity_scores_for_text(
        self,
        *,
        session,
        query_embedding: Optional[np.ndarray],
        concept_ids: tuple[int, ...],
    ) -> Optional[dict[int, float]]:
        if (
            not concept_ids
            or self.text_embedding_model is None
            or query_embedding is None
        ):
            return None
        if not self.kg.emb.is_model_registered(
            session=session,
            model_name=self.text_embedding_model,
        ):
            return None
        scores = self.kg.emb.get_similarities(
            session=session,
            embedding_model_name=self.text_embedding_model,
            text_embedding=query_embedding[0].tolist(),
            concept_ids=concept_ids,
        )
        return {int(concept_id): float(score) for concept_id, score in scores.items()}

    @staticmethod
    def _normalize_query_embedding(embedding: np.ndarray) -> np.ndarray:
        array = np.asarray(embedding, dtype=np.float32)
        if array.ndim == 1:
            return array.reshape(1, -1)
        if array.ndim == 2 and array.shape[0] == 1:
            return array
        raise ValueError(
            f"Expected query embedding to have shape (d,) or (1, d); got {array.shape}."
        )
