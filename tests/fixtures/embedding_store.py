"""A real sqlite-vec embedding store, which refuses a request for a concept it does not hold."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from oa_configurator import ConnectionConfig, Dialect, GenericDatabaseConfig, Resolver, StackConfig
from omop_emb.backends.embedding_table import ConceptEmbeddingRecord
from omop_emb.backends.index_config import FlatIndexConfig
from omop_emb.backends.sqlitevec import SQLiteVecEmbeddingBackend
from omop_emb.backends.sqlitevec.sqlitevec_backend import _load_sqlite_vec
from omop_emb.config import MetricType
from omop_emb.interface import EmbeddingReaderInterface
from omop_emb.model_registry import registry_writer_engine

STORE_MODEL = "test-model:v1"


def sqlite_vec_reader(vectors: Mapping[int, Sequence[float]]) -> EmbeddingReaderInterface:
    """
    Build an in-memory sqlite-vec store holding the given vectors, read through omop-emb.

    Parameters
    ----------
    vectors : Mapping[int, Sequence[float]]
        The vector of each concept the store holds, all of one length.

    Returns
    -------
    EmbeddingReaderInterface
        The reader of the stored model ``STORE_MODEL``.
    """
    database = Resolver(
        StackConfig.for_session(
            connections={"db": ConnectionConfig(dialect=Dialect.SQLITE, database_name=":memory:")},
            databases={"default": GenericDatabaseConfig(connection="db")},
        )
    ).resolve_database("default")
    backend = SQLiteVecEmbeddingBackend(
        emb_engine=registry_writer_engine(database, extensions=[_load_sqlite_vec])
    )
    matrix = np.array(list(vectors.values()), dtype=np.float32)
    backend.register_model(
        model_name=STORE_MODEL,
        provider_type="ollama",
        index_config=FlatIndexConfig(),
        dimensions=matrix.shape[1],
    )
    backend.upsert_embeddings(
        model_name=STORE_MODEL,
        records=[
            ConceptEmbeddingRecord(
                concept_id=concept_id,
                domain_id="Condition",
                vocabulary_id="SNOMED",
                is_standard=True,
                is_valid=True,
            )
            for concept_id in vectors
        ],
        embeddings=matrix,
    )
    return EmbeddingReaderInterface(
        model=STORE_MODEL, backend=backend, metric_type=MetricType.COSINE
    )
