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
from omop_graph.graph.nodes import LabelMatchKind

engine = create_engine("postgresql://user:pass@localhost/omop")
kg = KnowledgeGraph(engine)

# Lookup a concept by its label
matches = kg.concept_lookup("Atrial Fibrillation", match_kind=LabelMatchKind.EXACT)
if matches:
    print(f"ID: {matches[0].matched_concept_id}, Name: {matches[0].matched_concept_label}")

    # Traverse the hierarchy
    parents = kg.parents(matches[0].matched_concept_id)
    print(f"Parent IDs: {parents}")
```

---

### Embedding Configuration

To enable semantic similarity and RAG-based retrieval, pass a `KnowledgeGraphEmbeddingConfiguration` when initialising the graph.
This requires the optional `omop-emb` package — see the [installation guide](../usage/installation.md#embedding-rag).

!!! info "omop-emb documentation"
    `omop-emb` manages all embedding storage, backends, and retrieval. Full documentation — including backend setup, CLI reference, FAISS sidecar, and configuration — is available at [australiancancerdatanetwork.github.io/omop-emb](https://australiancancerdatanetwork.github.io/omop-emb/).

`KnowledgeGraphEmbeddingConfiguration` separates two independent things: 
1. *which* model (`model_name` + `provider_type`, plain strings and always required, since the embedding registry is keyed by both, even for a purely read-only lookup)
2. *how to call* that model (a fully resolved `oa_configurator.ResolvedModel`, carrying a real provider *connection* [`base_url`/`api_key`], resolved from an actual `[providers.*]` entry. Only required when `write=True`). 

A read-only consumer still needs to know the provider's plain string key (e.g. `"ollama"`), but that key never has to correspond to a resolved, connectable `[providers.*]` entry. This means a bare string is enough, since the connection details are never used.

#### Read-only (pre-computed embeddings already in the DB, or an externally-supplied query embedding)

Use this when embeddings have already been indexed and you only need retrieval — including the case where the caller supplies an already-computed `query_embedding` directly (see `annotate_text(query_embedding=...)`), so the KG never needs to call a model at all:

```python
from sqlalchemy import create_engine
from omop_graph.graph.kg import KnowledgeGraph, KnowledgeGraphEmbeddingConfiguration
from omop_emb.config import BackendType, MetricType

engine = create_engine("postgresql://user:pass@localhost/omop")

emb_config = KnowledgeGraphEmbeddingConfiguration(
    backend_type=BackendType.PGVECTOR,      # or BackendType.SQLITEVEC
    provider_type="ollama",
    model_name="nomic-embed-text:v1.5",     # must match the name used at ingestion time
    metric_type=MetricType.COSINE,
    # write defaults to False
)
kg = KnowledgeGraph(engine, emb_config=emb_config)
```

The backend is resolved from `backend_type` or, as a fallback, from the `OMOP_EMB_BACKEND` environment variable.
See the [omop-emb configuration reference](https://australiancancerdatanetwork.github.io/omop-emb/usage/configuration/) for all connection variables.

#### Write-capable (generate and store embeddings at runtime)

Set `write=True` and provide `resolved_model` (an `oa_configurator.ResolvedModel`, resolved from a `[models.*]` entry) to enable generating and persisting embeddings on demand, not just reading. The model name and provider are derived automatically from it — no need to also pass `model_name`/`provider_type`:

```python
from oa_configurator import Resolver
from omop_emb.config import BackendType, MetricType

resolved_model = Resolver.from_active_config().resolve_model("nomic-embed")  # a [models.*] entry name

emb_config = KnowledgeGraphEmbeddingConfiguration(
    backend_type=BackendType.PGVECTOR,
    metric_type=MetricType.COSINE,
    write=True,
    resolved_model=resolved_model,
)
kg = KnowledgeGraph(engine, emb_config=emb_config)
```

`write=True` without `resolved_model` raises `ValueError` at construction — fail fast, not deep inside a similarity call.

See [omop-llm: Asymmetric Embeddings](https://AustralianCancerDataNetwork.github.io/omop-llm/usage/asymmetric-embeddings/) and
[oa-configurator: `[models.<name>]`](https://AustralianCancerDataNetwork.github.io/OA_Configurator/config-reference/#modelsname)
for how a model (provider, connection details, embedding dimension, and asymmetric-prefix configuration) gets registered under a name in the first place.

#### Fallback embedding calculation

When some concepts in the OMOP DB have not been pre-indexed, similarity scoring will silently skip them.
Setting `compute_missing_embeddings=True` instructs the graph to compute and persist embeddings
for any missing concepts on-the-fly during a similarity call.

!!! warning
    `compute_missing_embeddings=True` requires `write=True`. This is  validated at construction and raises `ValueError` immediately if `write` is `False`

```python
emb_config = KnowledgeGraphEmbeddingConfiguration(
    backend_type=BackendType.PGVECTOR,
    metric_type=MetricType.COSINE,
    write=True,
    resolved_model=resolved_model,
    compute_missing_embeddings=True,
)
kg = KnowledgeGraph(engine, emb_config=emb_config)
```

| `compute_missing_embeddings` | `write` | Behaviour |
|---|---|---|
| `True` | `False` | `ValueError` at construction as it is invalid combination |
| `False` | any | Log at INFO and skip missing concepts in scoring |
| `True` | `True` | Compute embeddings, persist to DB, then score |