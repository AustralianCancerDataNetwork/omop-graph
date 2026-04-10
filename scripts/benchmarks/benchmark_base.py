"""Shared helpers for benchmark scripts.

This module centralizes common benchmark plumbing (case loading, DB/KG setup,
constraints, and metric helpers) so multiple benchmark entry points can share
consistent behavior and report semantics.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import sqlalchemy as sa
from dotenv import load_dotenv
from sqlalchemy.orm import sessionmaker

from omop_graph.extensions.emb import EmbeddingBackendType
from omop_graph.graph.constraints import SearchConstraintConcept
from omop_graph.graph.kg import KnowledgeGraph
from omop_llm import LLMClient


@dataclass(frozen=True)
class BenchmarkCase:
    """One benchmark example and its expected target concept.

    Parameters
    ----------
    id : str
        Stable case identifier used in outputs.
    text : str
        Query text sent to resolver/grounding pipelines.
    bucket : str
        Difficulty bucket used for grouped reporting.
    domain : str
        OMOP domain constraint value or ``NA`` for unconstrained.
    vocabulary : str
        OMOP vocabulary constraint value or ``NA`` for unconstrained.
    expected_concept_id : int, optional
        Gold concept ID. ``None`` indicates out-of-scope/null-grounding case.
    expected_concept_name : str, optional
        Human-readable expected concept label.
    parent_ids : tuple[int, ...], optional
        Optional grounding parent IDs for case-specific hierarchy constraints.
    """

    id: str
    text: str
    bucket: str
    domain: str
    vocabulary: str
    expected_concept_id: Optional[int]
    expected_concept_name: Optional[str] = None
    parent_ids: Optional[Tuple[int, ...]] = None


def load_cases(path: Path) -> List[BenchmarkCase]:
    """Load benchmark cases from JSON into typed dataclass instances."""

    payload = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(payload, list):
        return [BenchmarkCase(**row) for row in payload]

    if isinstance(payload, dict):
        cases: List[BenchmarkCase] = []
        for bucket, bucket_cases in payload.items():
            for row in bucket_cases:
                if "parent_ids" in row and row["parent_ids"] is not None:
                    row = {**row, "parent_ids": tuple(row["parent_ids"])}
                cases.append(BenchmarkCase(bucket=bucket, **row))
        return cases

    raise TypeError(f"Unsupported benchmark case file shape: {type(payload).__name__}")


def build_session_factory(database_url: Optional[str]) -> sessionmaker:
    """Build a SQLAlchemy session factory for the configured OMOP database."""

    load_dotenv()
    resolved_url = database_url or os.getenv("OMOP_DATABASE_URL")
    if not resolved_url:
        raise RuntimeError(
            "No database URL provided. Pass --database-url or set OMOP_DATABASE_URL."
        )

    engine = sa.create_engine(resolved_url, future=True, echo=False)
    return sessionmaker(bind=engine, future=True)


def build_knowledge_graph(database_url: Optional[str]) -> KnowledgeGraph:
    """Create a KnowledgeGraph backed by the live OMOP CDM database."""

    return KnowledgeGraph(session_factory=build_session_factory(database_url))


def build_embedding_knowledge_graph(
    database_url: Optional[str],
    embedding_backend: Optional[EmbeddingBackendType],
    embedding_client: Optional[LLMClient],
    embedding_storage_base_dir: Optional[str],
) -> KnowledgeGraph:
    """Create a KnowledgeGraph with embedding support configured."""

    session_factory = build_session_factory(database_url)
    return KnowledgeGraph(
        session_factory=session_factory,
        emb_backend=embedding_backend,
        emb_base_storage_dir=embedding_storage_base_dir,
        emb_client=embedding_client,
    )


def case_constraints(case: BenchmarkCase) -> Optional[SearchConstraintConcept]:
    """Translate case metadata into OMOP search constraints when available."""

    if case.domain == "NA" and case.vocabulary == "NA":
        return None

    domains = (case.domain,) if case.domain != "NA" else None
    vocabularies = (case.vocabulary,) if case.vocabulary != "NA" else None

    return SearchConstraintConcept(
        domains=domains,
        vocabs=vocabularies,
        require_standard=False,
    )


def percentile(values: List[float], p: int) -> float:
    """Return a simple nearest-rank percentile for numeric values."""

    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int((p / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


def mcnemar(
    a: Sequence[Dict[str, float | int | bool | str]],
    b: Sequence[Dict[str, float | int | bool | str]],
    *,
    field: str = "top1_correct",
) -> Dict[str, float]:
    """Compute continuity-corrected McNemar-style paired comparison."""

    paired = [(float(x[field]), float(y[field])) for x, y in zip(a, b, strict=True)]
    b_only = sum(1 for x, y in paired if x == 0.0 and y == 1.0)
    a_only = sum(1 for x, y in paired if x == 1.0 and y == 0.0)
    denom = b_only + a_only
    chi2 = ((abs(b_only - a_only) - 1.0) ** 2 / denom) if denom > 0 else 0.0
    return {
        "a_only_correct": float(a_only),
        "b_only_correct": float(b_only),
        "mcnemar_chi2_cc": chi2,
    }


def ranking_metrics(
    predictions: Sequence[int],
    expected: Optional[int],
    k: int,
) -> Dict[str, float]:
    """Compute top1/MRR/Recall@K for one ranked prediction list."""

    if expected is None:
        return {"top1_correct": 0.0, "mrr": 0.0, "recall_at_k": 0.0}

    if expected not in predictions:
        return {"top1_correct": 0.0, "mrr": 0.0, "recall_at_k": 0.0}

    rank = predictions.index(expected) + 1
    return {
        "top1_correct": 1.0 if rank == 1 else 0.0,
        "mrr": 1.0 / rank,
        "recall_at_k": 1.0 if rank <= k else 0.0,
    }
