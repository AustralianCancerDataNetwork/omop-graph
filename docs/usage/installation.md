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

This installs the PostgreSQL-backed embedding support used by `omop-graph`.

If you want a different embedding backend, install `omop-emb` separately with the backend extra you need, for example:

```bash
pip install "omop-emb[faiss]"
```

A database backend is still required for OMOP concept metadata and model registration, even when vector retrieval uses FAISS.

This allows semantic retrieval and similarity searches during knowledge extraction and graph traversal.

!!! tip
    This is a recommended setting and improves the functionality of the library detrimental.
