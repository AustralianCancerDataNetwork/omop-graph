# Architecture

This library provides a lightweight, query-time knowledge-graph layer over an OMOP vocabulary database, with explicit separation between:

* graph access (nodes, edges, predicates),
* graph algorithms (traversal, pathfinding),
* path scoring and explanation, and
* presentation / inspection utilities.

# omop-graph

**omop-graph** is a lightweight, opinionated knowledge-graph traversal and path-analysis library built on top of the OMOP vocabulary model.

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
pip install omop-graph
```

## Core Concepts

### KnowledgeGraph

KnowledgeGraph is the main entry point. It wraps an existing SQLAlchemy session connected to an OMOP vocabulary schema. kg-core assumes OMOP semantics and tables.

```python
from from omop_graph.graph.kg import KnowledgeGraph
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

### Traversal, Paths and Scoring

You can:

* expand neighbourhoods
* extract subgraphs
* trace traversal decisions
* control which relationship kinds are followed
* discover multiple candidate paths between concepts and rank them
* render simple HTML cards for easy interactive exploration

```python
from omop_graph.graph.scoring import find_shortest_paths
from omop_graph.graph.edges import PredicateKind

ingredient = kg.concept_id_by_code("RxNorm", "6809") # Metformin
drug = kg.concept_id_by_code("RxNorm", "860975") # Metformin 500 MG Oral Tablet

kg.concept_view(drug) # ConceptView(id=40163924, RxNorm:860975, name='24 HR metformin hydrochloride 500 MG Extended Release Oral Tablet')
kg.concept_view(ingredient) # ConceptView(id=1503297, RxNorm:6809, name='metformin')

paths, trace = find_shortest_paths(
    kg,
    source=drug,
    target=ingredient,
    predicate_kinds={
        PredicateKind.ONTOLOGICAL,
        PredicateKind.MAPPING,
    },
    max_depth=6,
    traced=True,
)

ranked = rank_paths(kg, paths)

```

### 

```python
paths = kg.find_shortest_paths(
    source=a,
    target=b,
    max_depth=6,
)
ranked = kg.rank_paths(paths)
```

### Rendering

Outputs can be rendered as:

* plain text (CLI / logs)
* HTML (Jupyter)
* Mermaid diagrams

Rendering auto-detects the environment.

```python 
from IPython.display import HTML, display
from kg_core.render import render_trace

display(HTML(render_trace(kg, trace)))
```

## Project Structure
```graphql

omop_graph/
├── graph/          # graph logic, traversal, paths, scoring
├── render/         # HTML / text / Mermaid renderers
├── reasoning/      # Ontology traversal methods for specific reasoner tasks
├────── resolvers/  # Resolve labels for exact / fuzzy / synonym matches - TODO: embedding matches
├────── phenotypes/ # Set operations to build efficient hierarchical groupings for reasoning
├── api.py          # stable public API surface
└── db/             # session helpers

```