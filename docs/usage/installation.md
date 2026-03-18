# Installation: Core
The package can be regularly installed using `pip` and `uv`:

```bash
uv pip install omop-graph
```

!!! note

    The dependency on uv is temporarily. The dependencies `omop-emb` (see below) will be eventually hosted on PyPI, which
    allows the installation using regular pip. Furthermore, `omop-alchemy` currently depends on a feature branch with PR imminent.
    See `pyproject.toml` for details:
    ```toml
    [tool.uv.sources]
    omop-alchemy = { git = "https://github.com/AustralianCancerDataNetwork/OMOP_Alchemy.git", branch = "spires" }
    omop-emb = { git = "https://github.com/AustralianCancerDataNetwork/omop-emb.git", branch = "main" }
    ```

## Installation: Embedding and RAG support (optional, recommended)

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