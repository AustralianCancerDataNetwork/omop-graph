"""Embedding vectors served from the closure models and from a stored model."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from fixtures.embedding_store import sqlite_vec_reader
from fixtures.helpers import curie
from fixtures.hierarchy_cdm import CANCER, LUNG, LUNG_CANCER, hierarchy_cdm
from oa_configurator.resolver import ResolvedModel, ResolvedProvider
from oaklib.interfaces.embedding_provider_interface import CLOSURE_IC_MODEL, CLOSURE_MODEL
from omop_emb import MetricType

from omop_graph.graph.capabilities.embeddings import (
    EmbeddingWriterRequiredError,
    KnowledgeGraphEmbeddingConfiguration,
)
from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.oaklib_interface.omop_implementation import OMOPAlchemyImplementation
from omop_graph.oaklib_interface.unsupported import OMOPUnsupportedError

STORED = "nomic-embed-text:v1.5"


def _configuration(write: bool) -> KnowledgeGraphEmbeddingConfiguration:
    resolved = ResolvedModel(
        name="test-model",
        provider=ResolvedProvider(
            name="test-provider", provider="ollama", base_url="http://localhost", api_key=None
        ),
        model=STORED,
        embedding_dim=None,
        document_prefix=None,
        query_prefix=None,
        embeddings=True,
        tool_use=False,
        structured_output=False,
        extended_thinking=False,
        configuration={},
    )
    return KnowledgeGraphEmbeddingConfiguration(
        metric_type=MetricType.COSINE, backend=Mock(), resolved_model=resolved, write=write
    )


@pytest.fixture(scope="module")
def sessions():
    with hierarchy_cdm() as sessions:
        yield sessions


@pytest.fixture
def plain(sessions) -> OMOPAlchemyImplementation:
    return OMOPAlchemyImplementation(kg=KnowledgeGraph(sessions))


@pytest.fixture
def store() -> Mock:
    interface = Mock()
    interface.is_model_registered.return_value = True
    interface.canonical_model_name = STORED
    interface.get_embeddings_by_concept_ids.return_value = {
        LUNG_CANCER: np.array([1.0, 0.0]),
        CANCER: np.array([1.0, 1.0]),
    }
    interface.backend.get_stored_concept_ids.side_effect = lambda *, model_name, concept_filter: (
        {LUNG_CANCER, CANCER} & set(concept_filter.concept_ids)
    )
    interface.get_similar_concepts.return_value = (
        (
            SimpleNamespace(concept_id=LUNG_CANCER, similarity=1.0),
            SimpleNamespace(concept_id=CANCER, similarity=0.7),
            SimpleNamespace(concept_id=LUNG, similarity=0.2),
        ),
    )
    interface.get_nearest_concepts.return_value = (
        (SimpleNamespace(concept_id=CANCER, similarity=0.9),),
    )
    interface.embed_texts.return_value = np.array([[0.5, 0.5]])
    return interface


def _with_store(sessions, store: Mock, write: bool) -> OMOPAlchemyImplementation:
    kg = KnowledgeGraph(sessions, emb_config=_configuration(write))
    kg._emb = store
    return OMOPAlchemyImplementation(kg=kg)


def test_the_closure_models_are_always_available(plain) -> None:
    assert plain.embedding_models() == [CLOSURE_MODEL, CLOSURE_IC_MODEL]


def test_a_concept_is_as_similar_to_itself_as_it_can_be_in_the_closure_model(plain) -> None:
    assert plain.embedding_similarity(curie(CANCER), curie(CANCER), model=CLOSURE_MODEL) == (
        pytest.approx(1.0)
    )
    related = plain.embedding_similarity(curie(LUNG_CANCER), curie(CANCER), model=CLOSURE_MODEL)
    unrelated = plain.embedding_similarity(curie(LUNG_CANCER), curie(LUNG), model=CLOSURE_MODEL)
    assert related is not None and unrelated is not None
    assert related > unrelated


def test_the_closure_models_cannot_embed_text(plain) -> None:
    with pytest.raises(NotImplementedError):
        plain.text_embedding("lung cancer", model=CLOSURE_MODEL)


def test_the_closure_models_are_not_searched_without_candidates(plain) -> None:
    with pytest.raises(OMOPUnsupportedError, match="candidates"):
        plain.nearest_entities(curie(CANCER))
    with pytest.raises(OMOPUnsupportedError, match="candidates"):
        plain.nearest_entities_to_vector(np.zeros(3))


def test_the_closure_models_are_searched_among_candidates(plain) -> None:
    found = list(
        plain.nearest_entities(
            curie(LUNG_CANCER),
            limit=2,
            model=CLOSURE_MODEL,
            candidates=[curie(CANCER), curie(LUNG)],
        )
    )

    assert [c for c, _ in found] == [curie(CANCER), curie(LUNG)]


def test_the_stored_model_is_listed_when_the_graph_has_an_embedding_store(sessions, store) -> None:
    adapter = _with_store(sessions, store, write=False)

    assert adapter.embedding_models() == [CLOSURE_MODEL, CLOSURE_IC_MODEL, STORED]


def test_the_stored_vectors_are_those_of_the_concepts_that_have_one(sessions, store) -> None:
    adapter = _with_store(sessions, store, write=False)

    ids, matrix = adapter.entity_embeddings(
        [curie(LUNG_CANCER), curie(LUNG), curie(CANCER)], model=STORED
    )

    assert ids == [curie(LUNG_CANCER), curie(CANCER)]
    assert matrix.shape == (2, 2)
    assert adapter.embedding_similarity(curie(LUNG_CANCER), curie(CANCER), model=STORED) == (
        pytest.approx(1 / np.sqrt(2))
    )


def test_a_real_store_serves_the_concepts_that_have_a_vector_and_skips_the_others(sessions) -> None:
    kg = KnowledgeGraph(sessions, emb_config=_configuration(write=False))
    kg._emb = sqlite_vec_reader({LUNG_CANCER: [1.0, 0.0], CANCER: [1.0, 1.0]})
    adapter = OMOPAlchemyImplementation(kg=kg)

    ids, matrix = adapter.entity_embeddings([curie(LUNG_CANCER), curie(LUNG), curie(CANCER)])

    assert ids == [curie(LUNG_CANCER), curie(CANCER)]
    assert matrix.shape == (2, 2)


def test_the_stored_model_is_searched_through_its_index_without_the_query_concept(
    sessions, store
) -> None:
    adapter = _with_store(sessions, store, write=False)

    found = list(adapter.nearest_entities(curie(LUNG_CANCER), limit=2, model=STORED))

    assert found == [(curie(CANCER), 0.7), (curie(LUNG), 0.2)]
    assert store.get_similar_concepts.call_args.args[:2] == (LUNG_CANCER, 3)


def test_a_vector_is_searched_in_the_stored_model(sessions, store) -> None:
    adapter = _with_store(sessions, store, write=False)

    found = list(adapter.nearest_entities_to_vector(np.array([1.0, 0.0]), model=STORED))

    assert found == [(curie(CANCER), 0.9)]


def test_a_text_is_embedded_and_searched_with_a_write_configuration(sessions, store) -> None:
    adapter = _with_store(sessions, store, write=True)

    assert list(adapter.nearest_entities_to_text("cancer", model=STORED)) == [(curie(CANCER), 0.9)]
    assert adapter.text_embedding("cancer", model=STORED).tolist() == [0.5, 0.5]


def test_a_text_cannot_be_embedded_with_a_read_only_configuration(sessions, store) -> None:
    adapter = _with_store(sessions, store, write=False)

    with pytest.raises(EmbeddingWriterRequiredError):
        adapter.text_embedding("cancer", model=STORED)


def test_the_stored_model_is_the_default_when_the_graph_has_one(sessions, store) -> None:
    adapter = _with_store(sessions, store, write=True)

    ids, _ = adapter.entity_embeddings([curie(LUNG_CANCER), curie(CANCER)])
    found = list(adapter.nearest_entities(curie(LUNG_CANCER), limit=1))

    assert ids == [curie(LUNG_CANCER), curie(CANCER)]
    assert [c for c, _ in found] == [curie(CANCER)]
    assert adapter.text_embedding("lung cancer").shape == (2,)


def test_the_closure_model_is_the_default_without_a_store(plain) -> None:
    ids, matrix = plain.entity_embeddings([curie(LUNG_CANCER), curie(CANCER)])

    assert ids == [curie(LUNG_CANCER), curie(CANCER)]
    assert matrix.shape[0] == 2
