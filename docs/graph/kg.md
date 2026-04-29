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

The `KnowledgeGraph` can be used standalone after connecting to the OMOP CDM database on disk.

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from omop_graph.graph.kg import KnowledgeGraph

# Setup your SQLAlchemy session
engine = create_engine("postgresql://user:pass@localhost/omop")
SessionLocal = sessionmaker(bind=engine)

# Initialize the Virtual Knowledge Graph
kg = KnowledgeGraph(SessionLocal)

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

#### Read-only (pre-computed embeddings already in the DB)

Use this when embeddings have already been indexed and you only need retrieval:

```python
from omop_graph.graph.kg import KnowledgeGraph, KnowledgeGraphEmbeddingConfiguration
from omop_emb import BackendType, ProviderType

emb_config = KnowledgeGraphEmbeddingConfiguration(
    backend_type=BackendType.FAISS,
    provider_type=ProviderType.OLLAMA,
    canonical_model_name="text-embedding-3-small:0.6b",
    base_storage_dir="/data/embeddings",
)
kg = KnowledgeGraph(SessionLocal, emb_config=emb_config)
```

#### Write-capable (generate and store embeddings at runtime)

Provide an `EmbeddingClient` to enable both reading and writing embeddings:

```python
from omop_emb import EmbeddingClient
from omop_emb import BackendType, ProviderType

client = EmbeddingClient(...)  # configured for your provider

emb_config = KnowledgeGraphEmbeddingConfiguration(
    backend_type=BackendType.FAISS,
    base_storage_dir="/data/embeddings",
    client=client,
)
kg = KnowledgeGraph(SessionLocal, emb_config=emb_config)
```
The `provider_type` will be automatically determined from the `client`.

#### Fallback embedding calculation

When some concepts in the OMOP DB have not been pre-indexed, similarity scoring will silently skip them.
Setting `compute_missing_embeddings=True` instructs the graph to compute and persist embeddings
for any missing concepts on-the-fly during a similarity call.

!!! warning
    This flag has no effect unless a write-capable interface is configured (i.e. a `client` is provided).
    Without a `client`, the graph holds a read-only interface and cannot write back to the embedding store.

```python
emb_config = KnowledgeGraphEmbeddingConfiguration(
    backend_type="faiss",
    base_storage_dir="/data/embeddings",
    client=client,
    compute_missing_embeddings=True,  # compute embeddings for concepts not yet in the store
)
kg = KnowledgeGraph(SessionLocal, emb_config=emb_config)
```

| `compute_missing_embeddings` | `client` present | Behaviour when concepts are missing |
|---|---|---|
| `False` (default) | any | Log at INFO and skip missing concepts in scoring |
| `True` | no | Log warning that computation is not possible; skip missing concepts |
| `True` | yes | Compute embeddings, persist to DB, then score |