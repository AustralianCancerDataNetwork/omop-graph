"""The embedding provider interface over the OMOP knowledge graph."""

from collections.abc import Iterable, Iterator

import numpy as np
from oaklib.interfaces.embedding_provider_interface import EmbeddingProviderInterface
from oaklib.types import CURIE

from omop_graph.oaklib_interface.basic_ontology import OMOPBasicOntologyInterface
from omop_graph.oaklib_interface.unsupported import OMOPUnsupportedError


class OMOPEmbeddingProviderInterface(OMOPBasicOntologyInterface, EmbeddingProviderInterface):
    """
    Mixin serving concept embeddings.

    The oaklib ``closure`` and ``closure_ic`` models are always available, built from the
    ancestry. When the knowledge graph has an embedding configuration, its model is
    available too, with the vectors stored by omop-emb. Embedding a text, and so searching
    by text, needs a write-capable configuration, and the graph raises a typed error when it
    is not. The closure models cannot be searched without candidates, since that would
    compare against every concept.
    """

    @property
    def _store_model(self) -> str | None:
        """The name of the stored model, or None when the graph has no embedding store."""
        configuration = self.kg.embedding_configuration
        return configuration.model_name if configuration and self.kg.has_embeddings else None

    @property
    def default_embedding_model(self) -> str | None:
        """The stored model when the graph has an embedding store, otherwise oaklib's closure model."""
        return self._store_model

    def _is_store_model(self, model: str | None) -> bool:
        """Whether model, or the default, is the stored one."""
        return self._store_model is not None and self._resolve_model(model) == self._store_model

    def embedding_models(self) -> list[str]:
        """The closure models, and the stored model when the graph has an embedding store."""
        models = super().embedding_models()
        if self._store_model is not None:
            models.append(self._store_model)
        return models

    def _fetch_embeddings(self, curies: list[CURIE], model: str) -> dict[CURIE, np.ndarray | None]:
        """Read the stored vectors of the concepts that have one."""
        if not self._is_store_model(model):
            return super()._fetch_embeddings(curies, model)
        ids = self._resolve_concepts(curies)
        vectors = self.kg.concept_embeddings(ids.values())
        return {
            curie: vectors[concept_id] for curie, concept_id in ids.items() if concept_id in vectors
        }

    def text_embedding(self, text: str, model: str | None = None) -> np.ndarray:
        """
        Embed a text with the stored model, which needs a write-capable configuration.

        Raises
        ------
        omop_graph.graph.capabilities.embeddings.EmbeddingWriterRequiredError
            If the configuration is read-only.
        omop_graph.oaklib_interface.unsupported.OMOPUnsupportedError
            If model is not the stored model, or the graph has no embedding store.
        """
        if not self._is_store_model(model):
            raise OMOPUnsupportedError("EmbeddingProviderInterface.text_embedding(model=closure)")
        return self.kg.embed_text(text)[0]

    def _nearest(
        self, query: int | np.ndarray, limit: int, exclude: int | None
    ) -> Iterator[tuple[CURIE, float]]:
        """Yield the stored concepts nearest to a concept or vector, best first."""
        matches = self.kg.nearest_concepts(
            query, concept_filter=self._population, k=limit + (exclude is not None)
        )
        kept = (match for match in matches if match.concept_id != exclude)
        for _, match in zip(range(limit), kept, strict=False):
            yield self.codec.concept_curie(match.concept_id), match.similarity

    def nearest_entities(
        self,
        curie: CURIE,
        limit: int = 10,
        model: str | None = None,
        candidates: Iterable[CURIE] | None = None,
    ) -> Iterator[tuple[CURIE, float]]:
        """
        Find the entities whose embeddings are most similar to that of an entity.

        The stored model searches its vector index. Without candidates the closure
        models are not supported.
        """
        if candidates is not None:
            return super().nearest_entities(curie, limit, model, candidates)
        if not self._is_store_model(model):
            raise OMOPUnsupportedError(
                "EmbeddingProviderInterface.nearest_entities(model=closure, candidates=None)"
            )
        concept_id = self._resolve_concepts([curie]).get(curie)
        return iter(()) if concept_id is None else self._nearest(concept_id, limit, concept_id)

    def nearest_entities_to_vector(
        self,
        vector: np.ndarray,
        limit: int = 10,
        model: str | None = None,
        candidates: Iterable[CURIE] | None = None,
    ) -> Iterator[tuple[CURIE, float]]:
        """
        Find the entities whose embeddings are most similar to a vector.

        The stored model searches its vector index. Without candidates the closure
        models are not supported.
        """
        if candidates is not None:
            return super().nearest_entities_to_vector(vector, limit, model, candidates)
        if not self._is_store_model(model):
            raise OMOPUnsupportedError(
                "EmbeddingProviderInterface.nearest_entities_to_vector(model=closure, candidates=None)"
            )
        return self._nearest(np.asarray(vector), limit, None)
