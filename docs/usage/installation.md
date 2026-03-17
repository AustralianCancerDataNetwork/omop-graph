# Installation instructions
The package can be regularly installed using `pip`:

```bash
pip install omop-graph
```

## Optional: Embedding and RAG support
The optional [`omop-emb` module](https://australiancancerdatanetwork.github.io/omop-emb/) can be installed using the option `[emb]`:
```bash
pip install omop-graph[emb]
```

This allows to do RAG-based retrieval and semantic similarity searches during knowledge extraction and graph traversal.

!!! tip
    This is a recommended setting and improves the functionality of the library detrimental.