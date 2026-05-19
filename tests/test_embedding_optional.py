"""Tests for optional embedding extension behavior in omop-graph."""

from __future__ import annotations

import builtins
import contextlib
import logging
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import numpy as np
import pytest
from sqlalchemy import Engine

from omop_graph.extensions import emb as emb_ext
from omop_graph.extensions.emb import MissingExtensionError
from omop_graph.graph.kg import KnowledgeGraph, KnowledgeGraphEmbeddingConfiguration
from omop_graph.graph.nodes import LabelMatchKind
from omop_graph.graph.paths import StandardConcept



def test_get_embedding_interface_returns_none_for_missing_extension_error():
    class BrokenKG:
        @property
        def emb(self):
            raise MissingExtensionError()

    assert emb_ext.get_embedding_reader_interface(cast(KnowledgeGraph, BrokenKG())) is None


def test_knowledge_graph_emb_raises_missing_extension_error_when_omop_emb_unavailable(monkeypatch: pytest.MonkeyPatch):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("omop_emb"):
            raise ModuleNotFoundError("No module named 'omop_emb'", name="omop_emb")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    kg = object.__new__(KnowledgeGraph)
    kg._emb = None
    kg._emb_config = None

    with pytest.raises(MissingExtensionError):
        _ = kg.emb


# ── compute_missing_embeddings flag tests (using real KG + dummy CDM DB) ──

def _make_standard_concept(concept_id: int, name: str) -> StandardConcept:
    return StandardConcept(
        concept_id=concept_id,
        concept_name=name,
        separation=0,
        original_id=concept_id,
        original_name=name,
        matched_label=name,
        match_kind=LabelMatchKind.EXACT,
        synonym=False,
    )


def test_fallback_flag_true_logs_attempt_when_concepts_missing(
    mock_cdm_kg: KnowledgeGraph,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """With compute_missing_embeddings=True and missing concepts, the debug
    log confirming an on-the-fly computation attempt must be emitted.

    The test uses the real KG backed by the in-memory CDM DB. Because no real
    embedding model is available, the embedding interfaces are mocked. The
    assertion is on log output rather than on actual embedding computation.
    """
    mock_cdm_kg._emb_config = KnowledgeGraphEmbeddingConfiguration(
        compute_missing_embeddings=True,
        metric_type=cast(emb_ext.EmbeddingMetricType, "cosine"),
    )

    fake_reader = Mock()
    # concept 196653 ("Malignant tumor of kidney") exists in the dummy CDM DB
    fake_reader.get_concepts_without_embedding.return_value = {
        196653: "Malignant tumor of kidney"
    }

    monkeypatch.setattr(emb_ext, "HAS_OMOP_EMB", True)
    monkeypatch.setattr(emb_ext, "get_embedding_reader_interface", lambda _: fake_reader)
    # No writer injected: simulates a read-only config with fallback flag set.
    monkeypatch.setattr(emb_ext, "get_embedding_writer_interface", lambda _: None)
    monkeypatch.setattr(emb_ext, "get_neareast_concepts", lambda **_: None)

    with caplog.at_level(logging.DEBUG, logger="omop_graph.extensions.emb"):
        emb_ext.semantic_similarity(
            kg=mock_cdm_kg,
            standard_concepts=[_make_standard_concept(196653, "Malignant tumor of kidney")],
            text_embedding=np.zeros((1, 3), dtype=np.float32),
        )

    assert "Computing missing embeddings on-the-fly" in caplog.text


def test_fallback_flag_false_logs_disabled_when_concepts_missing(
    mock_cdm_kg: KnowledgeGraph,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """With compute_missing_embeddings=False and missing concepts, the info
    log stating that the flag is disabled must be emitted and no computation
    must be attempted.

    The test uses the real KG backed by the in-memory CDM DB.
    """
    mock_cdm_kg._emb_config = KnowledgeGraphEmbeddingConfiguration(
        compute_missing_embeddings=False,
        metric_type=cast(emb_ext.EmbeddingMetricType, "cosine"),
    )

    fake_reader = Mock()
    fake_reader.get_concepts_without_embedding.return_value = {
        196653: "Malignant tumor of kidney"
    }

    monkeypatch.setattr(emb_ext, "HAS_OMOP_EMB", True)
    monkeypatch.setattr(emb_ext, "get_embedding_reader_interface", lambda _: fake_reader)
    monkeypatch.setattr(emb_ext, "get_neareast_concepts", lambda **_: None)

    with caplog.at_level(logging.INFO, logger="omop_graph.extensions.emb"):
        emb_ext.semantic_similarity(
            kg=mock_cdm_kg,
            standard_concepts=[_make_standard_concept(196653, "Malignant tumor of kidney")],
            text_embedding=np.zeros((1, 3), dtype=np.float32),
        )

    assert "compute_missing_embeddings is disabled" in caplog.text
    fake_reader.embed_texts.assert_not_called()
