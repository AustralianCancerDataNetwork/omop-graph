::: omop_graph.graph.capabilities.concepts
    options:
      members:
        - UnknownConceptError

::: omop_graph.graph.capabilities.embeddings
    options:
      members:
        - EmbeddingError
        - MissingExtensionError
        - EmbeddingsNotConfiguredError
        - EmbeddingWriterRequiredError
        - EmbeddingBackendNotWritableError
        - EmbeddingModelNotRegisteredError
        - MissingEmbeddingError
        - KnowledgeGraphEmbeddingConfiguration
