# Knowledge Graph Facade

The `KnowledgeGraph` class is the primary interface for interacting with the OMOP Common Data Model (CDM) as a graph. It implements a **Virtual Knowledge Graph (VKG)** layer, providing a high-level, object-oriented facade over relational database tables.

## Rationale

While the OMOP CDM is stored in a Relational Database Management System (RDBMS), its vocabulary structure (concepts, relationships, and hierarchies) is inherently graph-based. However, querying these structures using standard SQL often requires complex joins and recursive logic that can be difficult to maintain and interpret.

`omop-graph` bridges this gap by:

* **Virtualization:** Operating directly on existing RDBMS tables without requiring a separate graph database (like Neo4j), ensuring compatibility with standard OHDSI deployments.
* **Information Retrieval:** Enabling sophisticated graph traversal (parents, children, ancestors) and semantic search which are critical for concept grounding and medical entity linking.
* **Abstraction:** Providing a deterministic framework for validating medical logic through a Pythonic API, hiding the underlying SQL complexity.

---

## Key Features

* **SQLAlchemy Integration:** Efficiently manages database sessions and executes optimized queries against the CDM.
* **LRU Caching:** Implements high-performance caching for frequent lookups, such as concept IDs, labels, and predicates, to minimize database round-trips.
* **Semantic Predicates:** Resolves standard OMOP relationship IDs into rich `Predicate` objects that understand hierarchy and directionality. See [here for more information](edges.md)
* **Flexible Search:** Supports exact matches, fuzzy `ILIKE` searches, and full-text search (bag-of-words) across concept names and synonyms. See [documentation for more information](../reasoning/resolvers.md)
* **Graph Traversal:** Simple methods to retrieve `edges`, `parents`, `children`, `roots`, and `leaves`.
* **Extensibility:** Includes a dedicated namespace for embedding-based operations (requires `omop-emb` - see [Installation instructions](../usage/installation.md#embedding-rag) for more information).

---

### Basic Usage

The `KnowledgeGraph` can be used standalone after connecting to the OMOP CDM database.

```python
from sqlalchemy import create_engine
from omop_graph.graph.kg import KnowledgeGraph

engine = create_engine("postgresql://user:pass@localhost/omop")
kg = KnowledgeGraph(engine)

# Lookup a concept by its label
match_group = kg.label_lookup("Atrial Fibrillation", fuzzy=False)
concept = match_group.best_match

print(f"ID: {concept.concept_id}, Name: {concept.matched_label}")

# Traverse the hierarchy
parents = kg.parents(concept.concept_id)
print(f"Parent IDs: {parents}")
```

---

### Embedding Configuration

To enable semantic similarity and RAG-based retrieval, pass a `KnowledgeGraphEmbeddingConfiguration` when initialising the graph.
This requires the optional `omop-emb` package — see the [installation guide](../usage/installation.md#embedding-rag).

!!! info "omop-emb documentation"
    `omop-emb` manages all embedding storage, backends, and retrieval. Full documentation — including backend setup, CLI reference, FAISS sidecar, and configuration — is available at [australiancancerdatanetwork.github.io/omop-emb](https://australiancancerdatanetwork.github.io/omop-emb/).

#### Read-only (pre-computed embeddings already in the DB)

Use this when embeddings have already been indexed and you only need retrieval:

```python
from sqlalchemy import create_engine
from omop_graph.graph.kg import KnowledgeGraph, KnowledgeGraphEmbeddingConfiguration
from omop_emb.config import BackendType, MetricType, ProviderType

engine = create_engine("postgresql://user:pass@localhost/omop")

emb_config = KnowledgeGraphEmbeddingConfiguration(
    backend_type=BackendType.PGVECTOR,      # or BackendType.SQLITEVEC
    provider_type=ProviderType.OLLAMA,
    model_name="nomic-embed-text:v1.5",     # must match the name used at ingestion time
    metric_type=MetricType.COSINE,
)
kg = KnowledgeGraph(engine, emb_config=emb_config)
```

The backend is resolved from `backend_type` or, as a fallback, from the `OMOP_EMB_BACKEND` environment variable.
See the [omop-emb configuration reference](https://australiancancerdatanetwork.github.io/omop-emb/usage/configuration/) for all connection variables.

#### Write-capable (generate and store embeddings at runtime)

Provide an `EmbeddingClient` to enable both reading and writing embeddings. The `provider_type` and `model_name`
are derived automatically from the client:

```python
from omop_emb import EmbeddingClient
from omop_emb.config import BackendType, MetricType

client = EmbeddingClient(
    model="nomic-embed-text:v1.5",
    api_base="http://ollama:11434/v1",
)

emb_config = KnowledgeGraphEmbeddingConfiguration(
    backend_type=BackendType.PGVECTOR,
    metric_type=MetricType.COSINE,
    client=client,
)
kg = KnowledgeGraph(engine, emb_config=emb_config)
```

#### Fallback embedding calculation

When some concepts in the OMOP DB have not been pre-indexed, similarity scoring will silently skip them.
Setting `compute_missing_embeddings=True` instructs the graph to compute and persist embeddings
for any missing concepts on-the-fly during a similarity call.

!!! warning
    This flag has no effect unless a write-capable interface is configured (i.e. a `client` is provided).
    Without a `client`, the graph holds a read-only interface and cannot write back to the embedding store.

```python
emb_config = KnowledgeGraphEmbeddingConfiguration(
    backend_type=BackendType.PGVECTOR,
    metric_type=MetricType.COSINE,
    client=client,
    compute_missing_embeddings=True,
)
kg = KnowledgeGraph(engine, emb_config=emb_config)
```

| `compute_missing_embeddings` | `client` present | Behaviour when concepts are missing |
|---|---|---|
| `False` (default) | any | Log at INFO and skip missing concepts in scoring |
| `True` | no | Log warning that computation is not possible; skip missing concepts |
| `True` | yes | Compute embeddings, persist to DB, then score |