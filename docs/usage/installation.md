# Installation

The package can be installed with `pip` or `uv`:

```bash
pip install omop-graph
# or
uv pip install omop-graph
```

## Embedding and RAG support (optional, recommended) {:#embedding-rag}

!!! tip
    Installing with `[emb]` is recommended. Without it the `KnowledgeGraph` operates in text-only mode and all embedding-based similarity scoring is disabled.

```bash
pip install "omop-graph[emb]"
# or
uv pip install "omop-graph[emb]"
```

This pulls in [`omop-emb`](https://australiancancerdatanetwork.github.io/omop-emb/) with its default **sqlite-vec** backend — a file-based vector store that requires no external database server and works out of the box.

It enables:

- vector similarity search over OMOP concepts
- embedding-weighted grounding scores
- on-the-fly embedding computation for un-indexed concepts

### Scaling up: pgvector or FAISS

For larger deployments or approximate-nearest-neighbour acceleration, install the corresponding extra instead of (or alongside) `[emb]`:

| Extra | What it adds |
|---|---|
| `omop-graph[emb]` | sqlite-vec backend (default, zero config) |
| `omop-graph[pgvector]` | PostgreSQL/pgvector backend |
| `omop-graph[faiss-cpu]` | FAISS sidecar for fast approximate search |

These can be combined:

```bash
pip install "omop-graph[pgvector,faiss-cpu]"
```

!!! info "omop-emb documentation"
    Backend configuration, CLI reference, and index management are covered in the
    [omop-emb documentation](https://australiancancerdatanetwork.github.io/omop-emb/usage/installation/).
    The `[emb]` extra mirrors the base `omop-emb` install; `[pgvector]` and `[faiss-cpu]` mirror
    `omop-emb[pgvector]` and `omop-emb[faiss-cpu]` respectively.
