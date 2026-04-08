"""Tests for optional embedding extension behavior in omop-graph."""

from __future__ import annotations

import builtins
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import numpy as np
import pytest
from sqlalchemy.orm import Session

from omop_graph.extensions import emb as emb_ext
from omop_graph.extensions.emb import MissingExtensionError
from omop_graph.graph.kg import KnowledgeGraph


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
