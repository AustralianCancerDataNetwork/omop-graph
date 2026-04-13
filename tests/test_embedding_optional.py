"""Tests for optional embedding extension behavior in omop-graph."""

from __future__ import annotations

import builtins
import contextlib
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import numpy as np
import pytest
from sqlalchemy.orm import Session

from omop_graph.extensions import emb as emb_ext
from omop_graph.extensions.emb import MissingExtensionError
from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.graph.nodes import LabelMatchKind
from omop_graph.graph.paths import StandardConcept


def test_get_neareast_concepts_returns_none_when_index_type_missing():
    kg = cast(KnowledgeGraph, SimpleNamespace(emb=SimpleNamespace()))

    result = emb_ext.get_neareast_concepts(
        session=Mock(spec=Session),
        kg=kg,
        text_embedding_model="test-model",
        text_embedding=np.zeros((1, 2), dtype=np.float32),
        concept_filter=None,
        metric_type=None,
        index_type=None,
    )

    assert result is None


def test_get_neareast_concepts_returns_none_when_metric_type_missing():
    kg = cast(KnowledgeGraph, SimpleNamespace(emb=SimpleNamespace()))

    result = emb_ext.get_neareast_concepts(
        session=Mock(spec=Session),
        kg=kg,
        text_embedding_model="test-model",
        text_embedding=np.zeros((1, 2), dtype=np.float32),
        concept_filter=None,
        metric_type=None,
        index_type=cast(emb_ext.EmbeddingIndexType, "flat"),
    )

    assert result is None


def test_get_embedding_interface_returns_none_for_missing_extension_error():
    class BrokenKG:
        @property
        def emb(self):
            raise MissingExtensionError()

    assert emb_ext.get_embedding_interface(cast(KnowledgeGraph, BrokenKG())) is None


def test_knowledge_graph_emb_raises_missing_extension_error_when_omop_emb_unavailable(monkeypatch: pytest.MonkeyPatch):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("omop_emb"):
            raise ModuleNotFoundError("No module named 'omop_emb'", name="omop_emb")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    kg = object.__new__(KnowledgeGraph)
    kg._emb = None
    kg._emb_backend = None
    kg._emb_base_storage_dir = None
    kg._emb_client = None

    with pytest.raises(MissingExtensionError):
        _ = kg.emb


def test_semantic_similarity_fallback_uses_missing_ids_with_index_type(monkeypatch: pytest.MonkeyPatch):
    """Validate the fallback embedding flow when initial nearest retrieval returns no results.

    This test forces ``semantic_similarity`` into its fallback branch by mocking
    ``get_neareast_concepts`` to return ``None`` on the first call and a valid
    score mapping on the second call.

    It verifies two regression-critical contract details:
    1. ``index_type`` is forwarded to
       ``EmbeddingInterface.get_concepts_without_embedding``.
    2. ``EmbeddingInterface.add_to_db`` receives only the concept IDs returned
       as missing by ``get_concepts_without_embedding`` (not the full candidate
       concept set).

    These assertions protect against interface drift with omop-emb where
    ``index_type`` is required and concept IDs must align with the embeddings
    being upserted.
    """
    class FakeEmbeddingInterface:
        def __init__(self):
            self.last_missing_kwargs = None
            self.last_add_kwargs = None

        def get_concepts_without_embedding(self, **kwargs):
            self.last_missing_kwargs = kwargs
            return {1: "alpha", 2: "beta"}

        def embed_texts(self, texts, embedding_client):
            assert tuple(texts) == ("alpha", "beta")
            assert embedding_client is not None
            return np.zeros((2, 3), dtype=np.float32)

        def add_to_db(self, **kwargs):
            self.last_add_kwargs = kwargs

    emb_interface = FakeEmbeddingInterface()

    class FakeKG:
        def session_factory(self):
            return contextlib.nullcontext(Mock(spec=Session))

    def fake_nearest(*args, **kwargs):
        fake_nearest.calls += 1
        if fake_nearest.calls == 1:
            return None
        return {1: 0.9, 2: 0.8}

    fake_nearest.calls = 0

    monkeypatch.setattr(emb_ext, "HAS_OMOP_EMB", True)
    monkeypatch.setattr(emb_ext, "get_embedding_interface", lambda kg: emb_interface)
    monkeypatch.setattr(emb_ext, "get_neareast_concepts", fake_nearest)

    result = emb_ext.semantic_similarity(
        kg=cast(KnowledgeGraph, FakeKG()),
        unique_standard_concepts=[
            StandardConcept(
                concept_id=1,
                concept_name="alpha",
                separation=0,
                original_id=1,
                original_name="alpha",
                matched_label="alpha",
                match_kind=LabelMatchKind.EXACT,
                synonym=False
            ),
            StandardConcept(
                concept_id=2,
                concept_name="beta",
                separation=0,
                original_id=2,
                original_name="beta",
                matched_label="beta",
                match_kind=LabelMatchKind.EXACT,
                synonym=False
            ),
            StandardConcept(
                concept_id=3,
                concept_name="gamma",
                separation=0,
                original_id=3,
                original_name="gamma",
                matched_label="gamma",
                match_kind=LabelMatchKind.EXACT,
                synonym=False
            ),
        ],
        text_embedding=np.zeros((1, 3), dtype=np.float32),
        text_embedding_model="test-model",
        embedding_client=Mock(),
        metric_type=cast(emb_ext.EmbeddingMetricType, "cosine"),
        index_type=cast(emb_ext.EmbeddingIndexType, "flat"),
    )

    assert result is not None
    assert fake_nearest.calls == 2
    assert emb_interface.last_missing_kwargs is not None
    assert emb_interface.last_missing_kwargs["index_type"] == "flat"
    assert emb_interface.last_add_kwargs is not None
    assert emb_interface.last_add_kwargs["concept_ids"] == (1, 2)
