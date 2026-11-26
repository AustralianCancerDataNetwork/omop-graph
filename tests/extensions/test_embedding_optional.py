"""The optional embedding boundary: typed errors, guards, capability flags and the KG methods."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
from oa_configurator.resolver import ResolvedModel, ResolvedProvider
from omop_alchemy.cdm.query import ConceptFilter
from omop_emb import MetricType
from omop_emb.utils.embedding_utils import EmbeddingConceptFilter

from omop_graph.graph.capabilities import embeddings as emb_module
from omop_graph.graph.capabilities.embeddings import (
    EmbeddingBackendNotWritableError,
    EmbeddingError,
    EmbeddingModelNotRegisteredError,
    EmbeddingsNotConfiguredError,
    EmbeddingWriterRequiredError,
    KnowledgeGraphEmbeddingConfiguration,
    MissingEmbeddingError,
    MissingExtensionError,
    embedding_filter,
)
from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.graph.nodes import LabelMatchKind
from omop_graph.graph.paths import StandardConcept
from omop_graph.reasoning import grounding as grounding_ext
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


def _configuration(*, write: bool = False, compute_missing: bool = False):
    return KnowledgeGraphEmbeddingConfiguration(
        metric_type=MetricType.COSINE,
        backend=Mock(),
        resolved_model=_make_resolved_model(),
        write=write,
        compute_missing_embeddings=compute_missing,
    )


def _bare_graph(configuration=None, interface=None) -> KnowledgeGraph:
    """A graph without a database, holding the given configuration and interface."""
    kg = object.__new__(KnowledgeGraph)
    kg._emb_config = configuration
    kg._emb = interface
    return kg


def _standard_concept(concept_id: int, name: str) -> StandardConcept:
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


# the errors


def test_the_errors_share_one_base() -> None:
    assert issubclass(MissingExtensionError, EmbeddingError)
    assert issubclass(MissingExtensionError, ImportError)
    for error in (
        EmbeddingsNotConfiguredError,
        EmbeddingWriterRequiredError,
        EmbeddingBackendNotWritableError,
        EmbeddingModelNotRegisteredError,
        MissingEmbeddingError,
    ):
        assert issubclass(error, EmbeddingError)


def test_a_missing_embedding_names_the_concepts() -> None:
    assert MissingEmbeddingError([3, 4]).concept_ids == (3, 4)


# the guards


def test_without_the_extension_every_capability_raises_and_none_is_claimed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(emb_module, "HAS_OMOP_EMB", False)
    kg = _bare_graph(_configuration(write=True))

    assert not kg.has_embeddings
    assert not kg.can_embed_text
    with pytest.raises(MissingExtensionError):
        _ = kg.emb
    with pytest.raises(MissingExtensionError):
        kg.nearest_concepts(np.zeros((1, 3)))
    with pytest.raises(MissingExtensionError):
        kg.embed_text("kidney")


def test_without_a_configuration_every_capability_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    kg = _bare_graph()

    assert not kg.has_embeddings
    assert not kg.can_embed_text
    for call in (
        lambda: kg.emb,
        lambda: kg.nearest_concepts(1),
        lambda: kg.concept_embeddings([1]),
        lambda: kg.embedding_similarity(1, 2),
        lambda: kg.embedding_models(),
        lambda: kg.concepts_without_embedding([1]),
    ):
        with pytest.raises(EmbeddingsNotConfiguredError):
            call()


def test_a_read_only_configuration_reads_but_cannot_embed() -> None:
    kg = _bare_graph(_configuration(write=False), interface=Mock())

    assert kg.has_embeddings
    assert not kg.can_embed_text
    for call in (
        lambda: kg.embed_text("kidney"),
        lambda: kg.nearest_concepts("kidney"),
        lambda: kg.embed_concepts([1]),
    ):
        with pytest.raises(EmbeddingWriterRequiredError):
            call()


def test_a_write_configuration_can_embed_text() -> None:
    kg = _bare_graph(_configuration(write=True), interface=Mock())

    assert kg.has_embeddings
    assert kg.can_embed_text


def test_the_guards_keep_the_name_and_docstring_of_the_method() -> None:
    assert KnowledgeGraph.nearest_concepts.__name__ == "nearest_concepts"
    assert "nearest to a query" in (KnowledgeGraph.nearest_concepts.__doc__ or "")


def test_a_write_configuration_needs_a_write_capable_backend() -> None:
    kg = _bare_graph(_configuration(write=True))

    with pytest.raises(EmbeddingBackendNotWritableError, match="open_vector_store_writer"):
        _ = kg.emb


# the configuration


class TestEmbeddingConfigurationValidation:
    def test_compute_missing_embeddings_requires_write(self):
        with pytest.raises(ValueError, match="compute_missing_embeddings=True requires write=True"):
            KnowledgeGraphEmbeddingConfiguration(
                metric_type=MetricType.COSINE,
                backend=Mock(),
                resolved_model=_make_resolved_model(),
                compute_missing_embeddings=True,
            )

    def test_model_name_and_provider_type_derive_from_resolved_model(self):
        resolved = _make_resolved_model()
        cfg = KnowledgeGraphEmbeddingConfiguration(
            metric_type=MetricType.COSINE,
            backend=Mock(),
            resolved_model=resolved,
            write=True,
        )
        assert cfg.model_name == resolved.model
        assert cfg.provider_type == resolved.provider.provider


# the KG methods over a fake interface


def test_the_scope_becomes_the_embedding_filter() -> None:
    concept_filter = ConceptFilter(
        concept_ids=(1, 2),
        domains=("Condition",),
        vocabularies=("SNOMED",),
        require_standard=True,
        require_active=True,
        limit=9,
    )

    assert embedding_filter(concept_filter) == EmbeddingConceptFilter(
        concept_ids=(1, 2),
        domains=("Condition",),
        vocabularies=("SNOMED",),
        require_standard=True,
        require_active=True,
    )


def test_classification_concepts_are_not_passed_to_the_embedding_filter() -> None:
    translated = embedding_filter(ConceptFilter(require_standard=True, include_classification=True))

    assert translated == EmbeddingConceptFilter(require_standard=True)


def _registered_interface() -> Mock:
    interface = Mock()
    interface.is_model_registered.return_value = True
    interface.canonical_model_name = "test-model:v1.5"
    return interface


def test_the_nearest_concepts_of_a_vector_are_those_of_its_one_query() -> None:
    interface = _registered_interface()
    interface.get_nearest_concepts.return_value = (("near",),)
    kg = _bare_graph(_configuration(), interface)
    vector = np.zeros(3, dtype=np.float32)

    matches = kg.nearest_concepts(vector, concept_filter=ConceptFilter(domains=("Condition",)), k=5)

    assert matches == ("near",)
    call = interface.get_nearest_concepts.call_args
    assert call.args[0].shape == (1, 3)
    assert call.kwargs["k"] == 5
    assert call.kwargs["concept_filter"].domains == ("Condition",)
    assert not hasattr(call.kwargs["concept_filter"], "limit")


def test_a_concept_id_is_searched_through_its_stored_vector() -> None:
    interface = _registered_interface()
    interface.get_similar_concepts.return_value = (("similar",),)
    kg = _bare_graph(_configuration(), interface)

    assert kg.nearest_concepts(196653, k=3) == ("similar",)
    assert interface.get_similar_concepts.call_args.args[:2] == (196653, 3)


def test_a_text_is_embedded_before_it_is_searched() -> None:
    interface = _registered_interface()
    interface.embed_texts.return_value = np.ones((1, 3), dtype=np.float32)
    interface.get_nearest_concepts.return_value = (("near",),)
    kg = _bare_graph(_configuration(write=True), interface)

    assert kg.nearest_concepts("kidney cancer") == ("near",)
    assert interface.embed_texts.call_args.args == (("kidney cancer",),)


def test_several_query_vectors_are_rejected() -> None:
    kg = _bare_graph(_configuration(), _registered_interface())

    with pytest.raises(ValueError, match="one query vector"):
        kg.nearest_concepts(np.zeros((2, 3)))


def test_an_unregistered_model_raises_instead_of_returning_nothing() -> None:
    interface = Mock()
    interface.is_model_registered.return_value = False
    interface.canonical_model_name = "test-model:v1.5"
    kg = _bare_graph(_configuration(), interface)

    with pytest.raises(EmbeddingModelNotRegisteredError, match="test-model:v1.5"):
        kg.nearest_concepts(np.zeros((1, 3)))


def test_the_stored_embeddings_are_those_the_store_has() -> None:
    interface = _registered_interface()
    interface.get_embeddings_by_concept_ids.return_value = {1: np.array([1.0, 0.0])}
    kg = _bare_graph(_configuration(), interface)

    assert set(kg.concept_embeddings([1, 2, 1])) == {1}
    assert interface.get_embeddings_by_concept_ids.call_args.args == ((1, 2),)
    assert kg.concept_embeddings([]) == {}


def test_embedding_similarity_is_the_cosine_of_the_stored_vectors() -> None:
    interface = _registered_interface()
    interface.get_embeddings_by_concept_ids.return_value = {
        1: np.array([1.0, 0.0]),
        2: np.array([1.0, 1.0]),
    }
    kg = _bare_graph(_configuration(), interface)

    assert kg.embedding_similarity(1, 2) == pytest.approx(1 / np.sqrt(2))


def test_embedding_similarity_names_the_concepts_without_an_embedding() -> None:
    interface = _registered_interface()
    interface.get_embeddings_by_concept_ids.return_value = {1: np.array([1.0, 0.0])}
    kg = _bare_graph(_configuration(), interface)

    with pytest.raises(MissingEmbeddingError) as raised:
        kg.embedding_similarity(1, 2)
    assert raised.value.concept_ids == (2,)


def test_the_models_are_those_registered_in_the_store() -> None:
    interface = _registered_interface()
    interface.backend.get_registered_models.return_value = ("record",)
    kg = _bare_graph(_configuration(), interface)

    assert kg.embedding_models() == ("record",)


def test_missing_embeddings_are_embedded_from_their_names(
    mock_cdm_kg: KnowledgeGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = Mock()
    writer.get_concepts_without_embedding.return_value = {
        196653: SimpleNamespace(concept_name="Malignant tumor of kidney")
    }
    fetched = Mock(return_value={196653: "meta"})
    monkeypatch.setattr("omop_emb.utils.cdm.fetch_cdm_concepts_for_filter", fetched)
    mock_cdm_kg._emb_config = _configuration(write=True)
    mock_cdm_kg._emb = writer

    assert mock_cdm_kg.embed_concepts([196653]) == (196653,)
    call = writer.embed_and_upsert_concepts.call_args.kwargs
    assert call["concept_ids"] == (196653,)
    assert call["concept_texts"] == ("Malignant tumor of kidney",)
    assert call["concept_meta"] == {196653: "meta"}


def test_nothing_is_embedded_when_nothing_is_missing(mock_cdm_kg: KnowledgeGraph) -> None:
    writer = Mock()
    writer.get_concepts_without_embedding.return_value = {}
    mock_cdm_kg._emb_config = _configuration(write=True)
    mock_cdm_kg._emb = writer

    assert mock_cdm_kg.embed_concepts([196653]) == ()
    writer.embed_and_upsert_concepts.assert_not_called()


# grounding


def test_grounding_computes_missing_embeddings_when_the_kg_does(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    kg = Mock()
    kg.has_embeddings = True
    kg.compute_missing_embeddings = True
    kg.concepts_without_embedding.return_value = (196653,)
    kg.nearest_concepts.return_value = ()

    with caplog.at_level(logging.DEBUG, logger="omop_graph.reasoning.grounding"):
        grounding_ext._embedding_matches(
            kg, [_standard_concept(196653, "Malignant tumor of kidney")], np.zeros((1, 3))
        )

    assert "Computing missing embeddings on-the-fly" in caplog.text
    kg.embed_concepts.assert_called_once_with((196653,))


def test_grounding_skips_missing_embeddings_when_the_kg_does_not_compute_them(
    caplog: pytest.LogCaptureFixture,
) -> None:
    kg = Mock()
    kg.has_embeddings = True
    kg.compute_missing_embeddings = False
    kg.concepts_without_embedding.return_value = (196653,)
    kg.nearest_concepts.return_value = ()

    with caplog.at_level(logging.INFO, logger="omop_graph.reasoning.grounding"):
        grounding_ext._embedding_matches(
            kg, [_standard_concept(196653, "Malignant tumor of kidney")], np.zeros((1, 3))
        )

    assert "compute_missing_embeddings is disabled" in caplog.text
    kg.embed_concepts.assert_not_called()


def test_grounding_searches_the_given_concepts_with_k_set_to_their_number() -> None:
    kg = Mock()
    kg.has_embeddings = True
    kg.compute_missing_embeddings = False
    kg.concepts_without_embedding.return_value = ()
    kg.nearest_concepts.return_value = ("match",)

    matches = grounding_ext._embedding_matches(
        kg,
        [_standard_concept(196653, "Kidney"), _standard_concept(4038835, "Hodgkin")],
        np.zeros((1, 3)),
    )

    assert matches == (("match",),)
    call = kg.nearest_concepts.call_args
    assert call.kwargs["concept_filter"] == ConceptFilter(concept_ids=(196653, 4038835))
    assert call.kwargs["k"] == 2


def test_grounding_scores_nothing_without_an_embedding_store_or_matches() -> None:
    no_store = Mock()
    no_store.has_embeddings = False
    unregistered = Mock()
    unregistered.has_embeddings = True
    unregistered.concepts_without_embedding.return_value = ()
    unregistered.nearest_concepts.side_effect = EmbeddingModelNotRegisteredError("test-model")
    empty = Mock()
    empty.has_embeddings = True
    empty.concepts_without_embedding.return_value = ()
    empty.nearest_concepts.return_value = ()
    concepts = [_standard_concept(196653, "Kidney")]

    for kg in (no_store, unregistered, empty):
        assert grounding_ext._embedding_matches(kg, concepts, np.zeros((1, 3))) is None


# the resolver


def test_embedding_resolver_threads_constraints_limit_as_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(resolvers_ext, "HAS_OMOP_EMB", True)
    kg = Mock()
    kg.has_embeddings = True
    kg.nearest_concepts.return_value = ()
    kg.concept_views.return_value = ()
    constraints = ConceptFilter(domains=("Condition",), limit=5)

    resolvers_ext.EmbeddingResolver().get_matches(
        kg=kg, query="kidney cancer", constraints=constraints, query_embedding=np.zeros((1, 3))
    )

    call = kg.nearest_concepts.call_args
    assert call.kwargs["k"] == 5
    assert call.kwargs["concept_filter"] == constraints


def test_embedding_resolver_returns_nothing_without_a_store_or_a_registered_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(resolvers_ext, "HAS_OMOP_EMB", True)
    resolver = resolvers_ext.EmbeddingResolver()
    vector = np.zeros((1, 3))
    no_store = Mock()
    no_store.has_embeddings = False
    unregistered = Mock()
    unregistered.has_embeddings = True
    unregistered.nearest_concepts.side_effect = EmbeddingModelNotRegisteredError("test-model")

    assert resolver.get_matches(no_store, "kidney", query_embedding=vector) == ()
    assert resolver.get_matches(unregistered, "kidney", query_embedding=vector) == ()
    assert resolver.get_matches(unregistered, "kidney", query_embedding=None) == ()


def test_embedding_resolver_needs_the_extension(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(resolvers_ext, "HAS_OMOP_EMB", False)

    with pytest.raises(MissingExtensionError):
        resolvers_ext.EmbeddingResolver()


def test_a_registered_model_is_checked_once() -> None:
    interface = _registered_interface()
    kg = _bare_graph(_configuration(), interface=interface)

    for _ in range(3):
        kg._registered()

    assert interface.is_model_registered.call_count == 1


def test_a_missing_package_that_is_found_but_does_not_import_is_missing() -> None:
    from omop_graph.graph.capabilities.embeddings import _omop_emb_installed

    with patch("importlib.import_module", side_effect=ImportError("broken")):
        assert not _omop_emb_installed()
