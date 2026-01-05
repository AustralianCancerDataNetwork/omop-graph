# Architecture

This library provides a lightweight, query-time knowledge-graph layer over an OMOP vocabulary database, with explicit separation between:

* graph access (nodes, edges, predicates),
* graph algorithms (traversal, pathfinding),
* path scoring and explanation, and
* presentation / inspection utilities.

# kg-core

**kg-core** is a lightweight, opinionated knowledge-graph traversal and path-analysis library built on top of the OMOP vocabulary model.

It provides:
- a stable **KnowledgeGraph façade** over OMOP concepts and relationships
- flexible **graph traversal** (forward, backward, bidirectional)
- **path discovery and ranking** with transparent scoring
- **traceable explanations** of why one path is preferred over another
- multiple **rendering backends** (text, HTML, Mermaid)

The library is designed for:
- interactive analysis (Jupyter)
- reproducible research
- downstream tooling (NLP pipelines, ontology alignment, curation tools)

---

## Installation

```bash
pip install kg-core
```

## Core Concepts

### KnowledgeGraph

KnowledgeGraph is the main entry point. It wraps an existing SQLAlchemy session connected to an OMOP vocabulary schema. kg-core assumes OMOP semantics and tables.

```python
from kg_core.omop.session import OmopKnowledgeGraph as KnowledgeGraph
```

### Nodes and Edges

Nodes are OMOP Concepts; Edges are OMOP Concept_Relationships

Relationships are classified into semantic kinds:

* ONTOLOGICAL
* MAPPING
* ATTRIBUTE
* VERSIONING
* METADATA

This classification drives traversal and scoring.

### Traversal

You can:

* expand neighbourhoods
* extract subgraphs
* trace traversal decisions
* control which relationship kinds are followed

```python
from kg_core.graph.predicates import PredicateKind

sg, trace = kg.extract_subgraph_traced(
    seeds=[concept_id],
    predicate_kinds={PredicateKind.ONTOLOGICAL},
    max_depth=2,
)
```

### Paths and Scoring
kg-core can discover multiple candidate paths between concepts and rank them.

```python
paths = kg.find_shortest_paths(
    source=a,
    target=b,
    max_depth=6,
)
ranked = kg.rank_paths(paths)
```

Scoring is configurable and explainable.

### Rendering

Outputs can be rendered as:

* plain text (CLI / logs)
* HTML (Jupyter)
* Mermaid diagrams

Rendering auto-detects the environment.

```python 
from kg_core.render import render_path

render_path(kg, path)          # auto
render_path(kg, path, format="mmd")
```

## Project Structure
```graphql

kg_core/
├── graph/        # graph logic, traversal, paths, scoring
├── omop/         # OMOP-specific models, queries, session
├── render/       # HTML / text / Mermaid renderers
├── api.py        # stable public API surface
└── db/           # session helpers

```