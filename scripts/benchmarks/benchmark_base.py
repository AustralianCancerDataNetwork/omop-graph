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


DEFAULT_VOCABULARIES: Tuple[str, ...] = ("SNOMED", "ICDO3", "HemOnc")


def _normalize_parent_ids(raw_parent_ids: object) -> Optional[Tuple[int, ...]]:
    """Normalize parent_ids from JSON into an optional tuple of ints."""

    if raw_parent_ids is None:
        return None
    if isinstance(raw_parent_ids, int):
        return (raw_parent_ids,)
    if isinstance(raw_parent_ids, (list, tuple)):
        return tuple(int(parent_id) for parent_id in raw_parent_ids)
    raise TypeError(
        "Invalid parent_ids value. Expected int, list[int], tuple[int, ...], or null."
    )


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
    domain : str, optional
        OMOP domain constraint value. ``None`` means unconstrained.
    vocabularies : tuple[str, ...], optional
        OMOP vocabulary constraints. ``None`` or empty falls back to defaults.
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
    domain: Optional[str] = None
    vocabularies: Optional[Tuple[str, ...]] = None
    expected_concept_id: Optional[int] = None
    expected_concept_name: Optional[str] = None
    parent_ids: Optional[Tuple[int, ...]] = None


def load_cases(path: Path) -> List[BenchmarkCase]:
    """Load benchmark cases from JSON into typed dataclass instances."""

    payload = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(payload, list):
        cases: List[BenchmarkCase] = []
        for row in payload:
            if "parent_ids" in row:
                row = {**row, "parent_ids": _normalize_parent_ids(row["parent_ids"])}
            if "vocabularies" not in row and "vocabulary" in row:
                legacy_vocab = row.get("vocabulary")
                if legacy_vocab is not None:
                    row = {**row, "vocabularies": (str(legacy_vocab),)}
                row = {k: v for k, v in row.items() if k != "vocabulary"}
            if "vocabularies" in row and row["vocabularies"] is not None:
                raw_vocabs = row["vocabularies"]
                if isinstance(raw_vocabs, str):
                    row = {**row, "vocabularies": (raw_vocabs,)}
                else:
                    row = {**row, "vocabularies": tuple(raw_vocabs)}
            cases.append(BenchmarkCase(**row))
        return cases

    if isinstance(payload, dict):
        cases: List[BenchmarkCase] = []
        for bucket, bucket_cases in payload.items():
            for row in bucket_cases:
                if "parent_ids" in row:
                    row = {**row, "parent_ids": _normalize_parent_ids(row["parent_ids"])}
                if "vocabularies" not in row and "vocabulary" in row:
                    legacy_vocab = row.get("vocabulary")
                    if legacy_vocab is not None:
                        row = {**row, "vocabularies": (str(legacy_vocab),)}
                    row = {k: v for k, v in row.items() if k != "vocabulary"}
                if "vocabularies" in row and row["vocabularies"] is not None:
                    raw_vocabs = row["vocabularies"]
                    if isinstance(raw_vocabs, str):
                        row = {**row, "vocabularies": (raw_vocabs,)}
                    else:
                        row = {**row, "vocabularies": tuple(raw_vocabs)}
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


def build_engine(database_url: Optional[str]) -> sa.Engine:
    """Build a SQLAlchemy engine for the configured OMOP database."""

    load_dotenv()
    resolved_url = database_url or os.getenv("OMOP_DATABASE_URL")
    if not resolved_url:
        raise RuntimeError(
            "No database URL provided. Pass --database-url or set OMOP_DATABASE_URL."
        )

    return sa.create_engine(resolved_url, future=True, echo=False)


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

    domains = (case.domain,) if case.domain else None
    vocabularies = case.vocabularies if case.vocabularies else DEFAULT_VOCABULARIES

    return SearchConstraintConcept(
        domains=domains,
        vocabularies=vocabularies,
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
