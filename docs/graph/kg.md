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
* **Flexible Search:** Supports exact matches, fuzzy `ILIKE` searches, and full-text search (bag-of-words) across concept names and synonyms. See [documentaiton for more information](../reasoning/resolvers.md)
* **Graph Traversal:** Simple methods to retrieve `edges`, `parents`, `children`, `roots`, and `leaves`.
* **Extensibility:** Includes a dedicated namespace for embedding-based operations (requires `omop-emb` - see [Installation instructions](../usage/installation.md#optional-embedding-and-rag-support) for more information).

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