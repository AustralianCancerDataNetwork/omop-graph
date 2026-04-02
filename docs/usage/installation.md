# Installation: Core

!!! note

    The dependency on uv is not strictly enforced and can be replaced using `pip install omop-graph`.
    ```

The package can be regularly installed using `pip` and `uv`:

```bash
uv pip install omop-graph
```

## Installation: Embedding and RAG support (optional, recommended) {:#embedding-rag}

!!! tip
    This is a recommended setting and improves the functionality of the library detrimental.

The optional [`omop-emb` module](https://australiancancerdatanetwork.github.io/omop-emb/) can be installed using the option `[emb]`:
```bash
uv pip install omop-graph[emb]
```

This allows:

- RAG-based retrieval
- semantic similarity searches
- graph reasoning
- (Future): Agentic LLM interfaces