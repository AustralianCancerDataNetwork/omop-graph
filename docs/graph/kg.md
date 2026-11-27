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
* **Cached Classification:** Reads the [relationship classification](edges.md) once, when the graph is created, and keeps the valid domains and vocabularies after their first use.
* **Semantic Predicates:** Resolves standard OMOP relationship IDs into rich `Predicate` objects that understand hierarchy and directionality, and selects edges with an `EdgeFilter`. See [here for more information](edges.md)
* **Flexible Search:** Supports exact matches, fuzzy `ILIKE` searches, and full-text search (bag-of-words) across concept names and synonyms. See [documentation for more information](../reasoning/resolvers.md)
* **Graph Traversal:** Methods to retrieve `edges`, `relationships`, `parents`, `children`, `roots`, `leaves` and `singletons`, and the `Is a` `closure` and `entailed_relationships` described below.
* **Large inputs and results:** Methods that take concept ids split them into batches of `KnowledgeGraph.LOOKUP_BATCH_SIZE`, so any number of ids works. `entities`, `roots`, `leaves`, `singletons` and `relationships` without subjects or objects read through a server-side cursor and return generators, so they never hold the whole table in memory.
* **Standardisation and Mappings:** `standardise` finds the standard concepts a concept stands for, and `mappings` lists the mappings of concepts with their vocabularies.
* **Structural similarity and statistics:** Common ancestors, information content, pairwise similarity and counts, see [Capabilities](#capabilities).
* **Embeddings:** Nearest concepts, stored vectors and cosine similarity over `omop-emb`, which is optional - see [Embedding Configuration](#embedding-configuration) and the [installation instructions](../usage/installation.md#embedding-rag).

---

### Basic Usage

The `KnowledgeGraph` takes a factory of sessions on the OMOP CDM. `cdm_session_factory()` builds one for the configured database; to build it yourself, pass the pair from `create_engines()` to `omop_alchemy.cross_database.cdm_sessionmaker`. Its sessions send each table to the database hosting it, so the graph works the same when the vocabulary lives on its own server.

```python
from omop_graph.db.session import cdm_session_factory
from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.graph.nodes import LabelMatchKind

kg = KnowledgeGraph(cdm_session_factory())  # the configured cdm_db

# Lookup a concept by its label
matches = kg.concept_lookup("Atrial Fibrillation", match_kind=LabelMatchKind.EXACT)
if matches:
    print(f"ID: {matches[0].matched_concept_id}, Name: {matches[0].matched_concept_label}")

    # Traverse the hierarchy
    parents = kg.parents(matches[0].matched_concept_id)
    print(f"Parent IDs: {parents}")
```

---

### Capabilities

`KnowledgeGraph` holds construction and the relationship classification, and takes its methods from capability mixins in `omop_graph.graph.capabilities`. Each mixin builds on the ones it lists, and the oaklib adapter only translates CURIEs and oaklib types around these methods.

| Capability | Builds on | Methods |
| --- | --- | --- |
| Concepts | core | `concept_view`, `concept_views`, `concepts_by_code`, `missing_concepts`, `require_concepts` |
| Predicates | core | `predicate`, `predicates`, `predicate_kind`, `default_relationship_ids` |
| Edges | predicates | `relationships`, `all_relationships`, `edges`, `edges_between`, `iter_edges` |
| Enumeration | edges | `entities`, `roots`, `leaves`, `singletons`, `obsolete_concepts` |
| Hierarchy | edges, concepts | `parents`, `children`, `closure`, `standardise` |
| Entailment | hierarchy, enumeration | `entailed_relationships`, `all_entailed_relationships`, `gap_filled_relationships`, `entailing_relationship_ids` |
| Mappings | hierarchy | `mappings`, `mapping_predicates`, `standard_mappings`, `replacements` |
| Vocabulary | core | `vocabularies`, `domain_concept_ids`, `synonyms`, `translations`, `synonym_languages` |
| Search | enumeration | `search`, `concept_lookup`, `concept_ids_by_label`, `check_concept_filter` |
| Structural similarity | hierarchy | `common_ancestors`, `most_recent_common_ancestors`, `common_descendants`, `jaccard_similarity`, `information_content`, `pairwise_similarity` |
| Statistics | core | `concept_counts`, `relationship_counts`, `synonym_counts`, their `_by` forms for relationships and synonyms, and `group_statistics` |
| Embeddings | concepts | `nearest_concepts`, `concept_embeddings`, `embedding_similarity`, `embedding_models`, `concepts_without_embedding`, `embed_text`, `embed_concepts` |

* A scope is a `ConceptFilter` that decides which concepts a method reports or counts, see [Scopes](#scopes).
* `closure`, `entailed_relationships` and `all_entailed_relationships` take `standardise=True`, so that a non-standard concept stands in for the one standard concept it maps to. A stand-in reports the entailed relationships of the standard concept for itself, and not the mappings. With both subjects and objects, the stand-ins apply to the subjects. A concept that maps to several standard concepts raises a `ValueError`, since none is the more standard. `standardise` lists them, and `standard_concept` returns the one or raises.
* `entailed_relationships` entails the relationships that generalise and returns the others as stored, first.
* `search` ranks the matches of the kinds you give, on concept names and then, if asked, synonyms, with the concepts you name as identifiers first.

#### Scopes

Every `concept_filter` a method takes is a scope, a membership rule that decides which concepts are reported or counted.

* A scope that sets `limit` is refused, since a limit truncates a list and cannot decide membership. For a generator the error comes with its first item. The methods that truncate take a `limit` parameter of their own: `entities`, `roots`, `leaves`, `singletons`, `obsolete_concepts`, `concept_lookup` and `search`. With a `limit`, the enumerators return the concepts with the lowest ids, so the same ones come back on every run.
* A scope that sets `concept_ids` restricts what is reported to those ids.
* A scope only filters what is reported. A walk still passes through the concepts it excludes. A scope also applies to the concepts you name, and a relationship asserted to a concept is reported whatever the scope. In `entailed_relationships`, a requested object the scope excludes gets only the relationships asserted to it, and a warning lists it.
* The methods that describe the stored graph, `closure`, `relationships` and `entailed_relationships`, default to no scope. The methods that report or count concepts default to `DEFAULT_CONCEPT_FILTER`, active standard and classification concepts, and the counts and lookups to `ALL_CONCEPTS`.
* `omop_graph.graph.scope` has the constants and the helpers that derive one scope from another, which keep the restrictions the scope already has.

#### Unknown concepts

The methods that work out a property of the concepts you give them, `closure`, `standardise`, `standard_mappings`, `information_content`, the similarity methods and `concept_view`, raise `UnknownConceptError` when a concept does not exist. The error lists every missing id, and `missing_concepts` finds them beforehand. The queries that select rows by id, `relationships`, `edges` and `mappings`, return no rows for an id that does not exist. The oaklib adapter checks its CURIEs once, leaves out those that name no concept and logs them.

#### Structural similarity

Structural similarity compares two concepts by their place in the strict `Is a` hierarchy, and counts each concept as its own ancestor and descendant. Embedding similarity, by contrast, compares their stored vectors, see [Embeddings](#embedding-configuration).

```python
similarity = kg.pairwise_similarity(afib_id, flutter_id)
similarity.ancestor_id, similarity.jaccard, similarity.phenodigm
```

* The most recent common ancestors come from the ancestor sets alone. Only they and the two concepts have their descendants counted, so the cost does not grow with the number of common ancestors.
* The information content of a concept is `-log2(descendants / concepts in scope)`. The counts are cached for the life of the graph. Counting the descendants of a broad concept takes seconds the first time, such as 14 s for Clinical finding, and a pair whose only common ancestor is broad takes about as long.
* `ICSource.CONCEPT_ANCESTOR` counts from the precomputed `concept_ancestor` table instead, which is fast in bulk. It also holds classification and cohort ancestry, so its scores can differ from those of the strict hierarchy.
* The strict hierarchy has no ancestry for RxNorm ingredients, whose ATC classes are only in `concept_ancestor`, so two drugs have no common ancestor and score zero.

#### Statistics

`concept_counts` counts concepts by vocabulary, domain and deprecation, `relationship_counts` the active relationships by id, and `synonym_counts` the synonyms. `relationship_counts_by` and `synonym_counts_by` count them by vocabulary or domain, and `group_statistics` joins both per group and keeps the result for the life of the graph. The database counts them, and a scope that selects every concept counts the relationship table without joining the concepts. On a full Athena release the relationship count and the synonym count take about 6 s each, and counting by vocabulary or domain takes about a minute for every group together.

---

### Two Kinds of Ancestry

The graph has two notions of ancestry, and they differ:

| | Source | Holds |
|---|---|---|
| `parents`, `children`, and the ancestor checks of [grounding](../reasoning/grounding.md) | OMOP's `concept_ancestor` table | All ancestry OMOP precomputes: `Is a`, but also classification such as ATC classes, SPL labels and cohorts. Only standard and classification concepts. |
| `closure`, `entailed_relationships` and the [OAK adapter](../oaklib/interface.md) | The active `Is a` edges | Strict `Is a` ancestors and descendants, including concepts that are not standard. The hierarchy is assumed acyclic, as it is in the OMOP vocabularies, and `closure` ends on a cycle without repeating a concept. |

Grounding keeps `concept_ancestor` because it needs the distance to a parent, `min_levels_of_separation`, which `closure` does not return. For example Metformin has no `Is a` ancestors but six ancestors in `concept_ancestor`, ATC classes such as "Biguanides". `closure` walks one query for a whole batch of concepts, and takes an `EdgeFilter` to follow other relationships:

```python
from omop_graph.extensions.omop_alchemy import HierarchyDirection

kg.closure([concept_id])[concept_id]  # strict Is a ancestors
kg.closure([concept_id], HierarchyDirection.DOWN)[concept_id]  # strict Is a descendants
```

A non-standard concept, such as an ICD10CM code, has no ancestry of its own. `kg.standardise([concept_id])` maps it to the standard concepts at the end of its identity edges such as `Maps to`, which can be several. `kg.standard_concept` returns the one standard concept, and raises when there are several. A classification concept with a `Maps to` stands for its target in the same way, as an ATC class stands for its ingredient. Only a standard concept is a valid target.

---

### Embedding Configuration

To enable semantic similarity and RAG-based retrieval, pass a `KnowledgeGraphEmbeddingConfiguration` when initialising the graph.
This requires the optional `omop-emb` package, see the [installation guide](../usage/installation.md#embedding-rag).

!!! info "omop-emb documentation"
    `omop-emb` manages all embedding storage, backends, and retrieval. Full documentation, including backend setup, CLI reference, FAISS sidecar, and configuration, is available at [australiancancerdatanetwork.github.io/omop-emb](https://australiancancerdatanetwork.github.io/omop-emb/).

`KnowledgeGraphEmbeddingConfiguration` is a complete configuration: `backend` (an already-constructed `omop_emb.EmbeddingBackend`) and `resolved_model` (an `oa_configurator.ResolvedModel`) are both required fields, not Optional. omop-graph never resolves either itself: the caller (the actual CLI/entry-point boundary, e.g. omop-spires) resolves the vector store and the model, builds the backend, and passes both in here. `model_name`/`provider_type` are plain properties reading straight off `resolved_model` (`.model`/`.provider.provider`); there's nothing to pass for them separately.

```python
from oa_configurator import Resolver
from omop_emb.backends import open_vector_store_writer
from omop_graph.db.session import cdm_session_factory
from omop_graph.graph.kg import KnowledgeGraph, KnowledgeGraphEmbeddingConfiguration
from omop_emb.config import MetricType

resolver = Resolver.from_active_config()
resolved_vector_store = resolver.resolve_vector_store(
    "vector_store"
)  # a [vector_stores.*] entry name
resolved_model = resolver.resolve_model("embedding-model")  # a [models.*] entry name
backend = open_vector_store_writer(resolved_vector_store)

emb_config = KnowledgeGraphEmbeddingConfiguration(
    metric_type=MetricType.COSINE,
    backend=backend,
    resolved_model=resolved_model,
    # write defaults to False
)
kg = KnowledgeGraph(cdm_session_factory(), emb_config=emb_config)
```

See [omop-llm: Asymmetric Embeddings](https://AustralianCancerDataNetwork.github.io/omop-llm/usage/asymmetric-embeddings/) and
[oa-configurator: `[models.<name>]`](https://AustralianCancerDataNetwork.github.io/OA_Configurator/config-reference/#modelsname)
for how a model (provider, connection details, embedding dimension, and asymmetric-prefix configuration) gets registered under a name in the first place, and [oa-configurator: `[vector_stores.<name>]`](https://AustralianCancerDataNetwork.github.io/OA_Configurator/config-reference/#vector_storesname) for the storage backend side.

#### Read-only vs write-capable

`write=False` (the default) gives a read-only interface over already-computed embeddings, including the case where the caller supplies an already-computed `query_embedding` directly (see `annotate_text(query_embedding=...)`), so the KG never needs to call a model at all. Set `write=True` to enable generating and persisting embeddings on demand, using the same required `resolved_model`.

```python
emb_config = KnowledgeGraphEmbeddingConfiguration(
    metric_type=MetricType.COSINE,
    backend=backend,
    resolved_model=resolved_model,
    write=True,
)
kg = KnowledgeGraph(cdm_session_factory(), emb_config=emb_config)
```

`faiss_cache_dir` (optional `str`) is read-only-path-only: a directory to cache FAISS index files, passed straight through to `EmbeddingReaderInterface`.

#### What the graph offers

Embeddings are optional, and every method that needs them raises a typed error instead of returning `None`. All errors derive from `EmbeddingError`, in `omop_graph.graph.capabilities.embeddings`.

| Error | Raised when |
| --- | --- |
| `MissingExtensionError` | `omop-emb` is not installed. It is also an `ImportError`. |
| `EmbeddingsNotConfiguredError` | The graph has no `emb_config`. |
| `EmbeddingWriterRequiredError` | The method embeds text or computes embeddings, and the configuration is read-only. |
| `EmbeddingBackendNotWritableError` | The configuration has `write=True`, and the backend cannot write embeddings. |
| `EmbeddingModelNotRegisteredError` | The configured model is not registered in the vector store. |
| `MissingEmbeddingError` | `embedding_similarity` is asked for a concept with no stored vector. |

Code that degrades gracefully asks first: `kg.has_embeddings` is true when `omop-emb` is installed and a configuration is set, and `kg.can_embed_text` when the configuration can also write.

```python
matches = kg.nearest_concepts(vector, concept_filter=ConceptFilter(domains=("Condition",)), k=10)
matches = kg.nearest_concepts(concept_id)  # through its stored vector
matches = kg.nearest_concepts("kidney cancer")  # needs write=True
kg.embedding_similarity(concept_a, concept_b)  # cosine of the stored vectors
```

Vector search returns standard concepts when the scope requires them. It cannot return classification concepts, so `include_classification` has no effect on it.

#### Fallback embedding calculation

When some concepts in the OMOP DB have not been pre-indexed, similarity scoring will silently skip them.
Setting `compute_missing_embeddings=True` instructs grounding to compute and persist the embeddings
of any missing concepts before it scores them. `embedding_similarity` never writes, and raises `MissingEmbeddingError` for a concept without a vector. `embed_concepts` is the explicit write.

!!! warning
    `compute_missing_embeddings=True` requires `write=True`. This is validated at construction and raises `ValueError` immediately if `write` is `False`.

```python
emb_config = KnowledgeGraphEmbeddingConfiguration(
    metric_type=MetricType.COSINE,
    backend=backend,
    resolved_model=resolved_model,
    write=True,
    compute_missing_embeddings=True,
)
kg = KnowledgeGraph(cdm_session_factory(), emb_config=emb_config)
```

| `compute_missing_embeddings` | `write` | Behaviour |
|---|---|---|
| `True` | `False` | `ValueError` at construction as it is invalid combination |
| `False` | any | Log at INFO and skip missing concepts in scoring |
| `True` | `True` | Compute embeddings, persist to DB, then score |
