"""Tests for optional embedding extension behavior in omop-graph."""

from __future__ import annotations

import builtins
import logging
from typing import cast
from unittest.mock import Mock

import numpy as np
import pytest

from oa_configurator.resolver import ResolvedModel, ResolvedProvider

from omop_graph.extensions import emb as emb_ext
from omop_graph.extensions.emb import MissingExtensionError
from omop_graph.graph.constraints import SearchConstraintConcept
from omop_graph.graph.kg import KnowledgeGraph, KnowledgeGraphEmbeddingConfiguration
from omop_graph.graph.nodes import LabelMatchKind
from omop_graph.graph.paths import StandardConcept
from omop_graph.reasoning.resolvers import resolvers as resolvers_ext


def _make_resolved_model() -> ResolvedModel:
    return ResolvedModel(
        name="test-model",
        provider=ResolvedProvider(
            name="test-provider", provider="ollama", base_url="http://localhost:11434", api_key=None
        ),
        model="nomic-embed-text:v1.5",
        embedding_dim=None,
        document_prefix=None,
        query_prefix=None,
        embeddings=True,
        tool_use=False,
        structured_output=False,
        extended_thinking=False,
        configuration={},
    )


def test_get_embedding_interface_returns_none_for_missing_extension_error():
    class BrokenKG:
        @property
        def emb(self):
            raise MissingExtensionError()

    assert (
        emb_ext.get_embedding_reader_interface(cast(KnowledgeGraph, BrokenKG())) is None
    )


def test_knowledge_graph_emb_raises_missing_extension_error_when_omop_emb_unavailable(
    monkeypatch: pytest.MonkeyPatch,
):
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
        matched_concept_label=name,
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
        write=True,
        backend=Mock(),
        resolved_model=_make_resolved_model(),
        metric_type=cast(emb_ext.EmbeddingMetricType, "cosine"),
    )

    fake_reader = Mock()
    # concept 196653 ("Malignant tumor of kidney") exists in the dummy CDM DB
    fake_reader.get_concepts_without_embedding.return_value = {
        196653: "Malignant tumor of kidney"
    }

    monkeypatch.setattr(emb_ext, "HAS_OMOP_EMB", True)
    monkeypatch.setattr(
        emb_ext, "get_embedding_reader_interface", lambda _: fake_reader
    )
    # No writer injected: simulates a writer that failed to materialize for some
    # other reason (config says write=True, but the lookup itself returns None).
    monkeypatch.setattr(emb_ext, "try_get_embedding_writer_interface", lambda _: None)
    monkeypatch.setattr(emb_ext, "get_neareast_concepts", lambda **_: None)

    with caplog.at_level(logging.DEBUG, logger="omop_graph.extensions.emb"):
        emb_ext.semantic_similarity(
            kg=mock_cdm_kg,
            standard_concepts=[
                _make_standard_concept(196653, "Malignant tumor of kidney")
            ],
            query_embedding=np.zeros((1, 3), dtype=np.float32),
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
        backend=Mock(),
        resolved_model=_make_resolved_model(),
        metric_type=cast(emb_ext.EmbeddingMetricType, "cosine"),
    )

    fake_reader = Mock()
    fake_reader.get_concepts_without_embedding.return_value = {
        196653: "Malignant tumor of kidney"
    }

    monkeypatch.setattr(emb_ext, "HAS_OMOP_EMB", True)
    monkeypatch.setattr(
        emb_ext, "get_embedding_reader_interface", lambda _: fake_reader
    )
    monkeypatch.setattr(emb_ext, "get_neareast_concepts", lambda **_: None)

    with caplog.at_level(logging.INFO, logger="omop_graph.extensions.emb"):
        emb_ext.semantic_similarity(
            kg=mock_cdm_kg,
            standard_concepts=[
                _make_standard_concept(196653, "Malignant tumor of kidney")
            ],
            query_embedding=np.zeros((1, 3), dtype=np.float32),
        )

    assert "compute_missing_embeddings is disabled" in caplog.text
    fake_reader.embed_texts.assert_not_called()


# ── KnowledgeGraphEmbeddingConfiguration.__post_init__ validation ──


class TestEmbeddingConfigurationValidation:
    def test_compute_missing_embeddings_requires_write(self):
        with pytest.raises(ValueError, match="compute_missing_embeddings=True requires write=True"):
            KnowledgeGraphEmbeddingConfiguration(
                metric_type=cast(emb_ext.EmbeddingMetricType, "cosine"),
                backend=Mock(),
                resolved_model=_make_resolved_model(),
                compute_missing_embeddings=True,
            )

    def test_model_name_and_provider_type_derive_from_resolved_model(self):
        resolved = _make_resolved_model()
        cfg = KnowledgeGraphEmbeddingConfiguration(
            metric_type=cast(emb_ext.EmbeddingMetricType, "cosine"),
            backend=Mock(),
            resolved_model=resolved,
            write=True,
        )
        assert cfg.model_name == resolved.model
        assert cfg.provider_type == resolved.provider.provider


# ── try_get_embedding_writer_interface: the actual bug-fix regression test ──


class TestTryGetEmbeddingWriterInterface:
    def test_returns_none_for_read_only_configured_kg_instead_of_raising(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        """A deliberately read-only-configured KG (write=False) is a normal, valid
        state, not a bug -- try_get_embedding_writer_interface must degrade
        gracefully to None rather than raise, unlike get_embedding_writer_interface.

        A plain Mock() reader is not an instance of the real EmbeddingWriterInterface,
        which is exactly what a real read-only-configured KG.emb would return too.
        """
        fake_reader_interface = Mock()

        class ReadOnlyKG:
            @property
            def emb(self):
                return fake_reader_interface

        monkeypatch.setattr(emb_ext, "HAS_OMOP_EMB", True)

        with caplog.at_level(logging.DEBUG, logger="omop_graph.extensions.emb"):
            result = emb_ext.try_get_embedding_writer_interface(
                cast(KnowledgeGraph, ReadOnlyKG())
            )
        assert result is None
        assert "unittest.mock.Mock" in caplog.text
        assert "write=False" not in caplog.text

        # The raising sibling function must still raise for the exact same input --
        # confirms the two helpers genuinely differ in behavior, not just in name.
        with pytest.raises(TypeError, match="Expected embedding interface to be a writer"):
            emb_ext.get_embedding_writer_interface(cast(KnowledgeGraph, ReadOnlyKG()))

    def test_returns_none_when_no_embedding_config(self):
        class NoConfigKG:
            @property
            def emb(self):
                raise ValueError("Embedding configuration is not set.")

        assert emb_ext.try_get_embedding_writer_interface(cast(KnowledgeGraph, NoConfigKG())) is None

    def test_returns_none_when_extension_missing(self):
        class MissingExtensionKG:
            @property
            def emb(self):
                raise MissingExtensionError()

        assert emb_ext.try_get_embedding_writer_interface(cast(KnowledgeGraph, MissingExtensionKG())) is None


# ── omop-emb#48 split: k threaded explicitly instead of via filter.limit ──


def test_semantic_similarity_splits_cdm_and_knn_filters(
    mock_cdm_kg: KnowledgeGraph,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CDM-side lookups get a CDMConceptFilter capped by len(concept_ids);
    the KNN call gets a plain EmbeddingConceptFilter (no limit) with k passed
    explicitly instead.
    """
    from omop_emb.utils.embedding_utils import CDMConceptFilter, EmbeddingConceptFilter

    mock_cdm_kg._emb_config = KnowledgeGraphEmbeddingConfiguration(
        compute_missing_embeddings=False,
        backend=Mock(),
        resolved_model=_make_resolved_model(),
        metric_type=cast(emb_ext.EmbeddingMetricType, "cosine"),
    )

    fake_reader = Mock()
    fake_reader.get_concepts_without_embedding.return_value = {}
    fake_reader.is_model_registered.return_value = True
    fake_reader.canonical_model_name = "test-model:v1.5"
    fake_reader.get_nearest_concepts.return_value = ()

    monkeypatch.setattr(emb_ext, "HAS_OMOP_EMB", True)
    monkeypatch.setattr(
        emb_ext, "get_embedding_reader_interface", lambda _: fake_reader
    )

    emb_ext.semantic_similarity(
        kg=mock_cdm_kg,
        standard_concepts=[
            _make_standard_concept(196653, "Malignant tumor of kidney"),
            _make_standard_concept(4038835, "Hodgkin's disease (clinical)"),
        ],
        query_embedding=np.zeros((1, 3), dtype=np.float32),
    )

    cdm_filter = fake_reader.get_concepts_without_embedding.call_args.kwargs[
        "concept_filter"
    ]
    assert isinstance(cdm_filter, CDMConceptFilter)
    assert cdm_filter.limit == 2

    knn_kwargs = fake_reader.get_nearest_concepts.call_args.kwargs
    assert knn_kwargs["k"] == 2
    assert isinstance(knn_kwargs["concept_filter"], EmbeddingConceptFilter)
    assert not hasattr(knn_kwargs["concept_filter"], "limit")


def test_embedding_resolver_threads_constraints_limit_as_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """EmbeddingResolver.get_matches no longer sets EmbeddingConceptFilter.limit;
    constraints.limit is threaded through to get_neareast_concepts as k= instead.
    """
    monkeypatch.setattr(resolvers_ext, "HAS_OMOP_EMB", True)
    fake_get_neareast = Mock(return_value=((),))
    monkeypatch.setattr(resolvers_ext, "get_neareast_concepts", fake_get_neareast)

    kg = Mock()
    kg.concept_views.return_value = ()

    resolver = resolvers_ext.EmbeddingResolver()
    constraints = SearchConstraintConcept(domains=("Condition",), limit=5)

    resolver.get_matches(
        kg=kg,
        query="kidney cancer",
        constraints=constraints,
        query_embedding=np.zeros((1, 3), dtype=np.float32),
    )

    call_kwargs = fake_get_neareast.call_args.kwargs
    assert call_kwargs["k"] == 5
    assert call_kwargs["concept_filter"].domains == ("Condition",)
    assert not hasattr(call_kwargs["concept_filter"], "limit")
