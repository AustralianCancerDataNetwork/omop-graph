# omop-graph

**omop-graph** is a lightweight, opinionated knowledge-graph traversal and path-analysis library built on top of the OMOP vocabulary model.

It provides:
- a stable **KnowledgeGraph façade** over OMOP concepts and relationships
- flexible **graph traversal** (forward, backward, bidirectional)
- **path discovery** with transparent scoring
- **traceable explanations** of traversal decisions
- multiple **rendering backends** (text, HTML, Mermaid)
- an **[OAK](https://incatools.github.io/ontology-access-kit/) adapter**, so tools written for oaklib work on an OMOP CDM

The library is designed for:
- interactive analysis (Jupyter)
- reproducible research
- downstream tooling (NLP pipelines, ontology alignment, curation tools)

---

## Installation

```bash
pip install omop-graph
```

The extras for a DuckDB CDM (`[duckdb]`) and for embeddings (`[emb]`, `[pgvector]`, `[faiss-cpu]`) are described in [Installation](docs/usage/installation.md).

---

## Core Concepts

### KnowledgeGraph

`KnowledgeGraph` is the main entry point. It takes a factory of sessions on the OMOP CDM and provides a high-level Pythonic API over the relational tables.

```python
from omop_graph.db.session import cdm_session_factory
from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.graph.model.matches import LabelMatchKind

kg = KnowledgeGraph(cdm_session_factory())  # the configured cdm_db

# Lookup a concept by label
matches = kg.concept_lookup("Atrial Fibrillation", match_kind=LabelMatchKind.EXACT)
concept = matches[0]
print(f"ID: {concept.matched_concept_id}, Name: {concept.matched_concept_label}")

# Traverse the hierarchy
parents = kg.parents(concept.matched_concept_id)
```

### Nodes and Edges

Nodes are OMOP Concepts; Edges are OMOP Concept_Relationships.

Relationships are pre-classified into semantic kinds (`PredicateKind`):

- `HIERARCHY` — parent/child ontological relationships
- `IDENTITY` — mapping to standard concepts
- `COMPOSITION` — part-of relationships
- `ASSOCIATION` — lateral clinical associations
- `ATTRIBUTE` — concept attribute relationships

This classification drives traversal filtering and scoring.

### Traversal and Paths

```python
from omop_graph.graph.model.edges import EdgeFilter
from omop_graph.graph.paths import find_shortest_paths
from omop_graph.extensions.relationship_classification import PredicateKind

[ingredient] = kg.concepts_by_code("RxNorm", "6809")  # Metformin
[drug] = kg.concepts_by_code("RxNorm", "860975")  # Metformin 500 MG Oral Tablet

paths, trace = find_shortest_paths(
    kg,
    source=drug,
    target=ingredient,
    edge_filter=EdgeFilter(
        predicate_kinds=frozenset({PredicateKind.HIERARCHY, PredicateKind.IDENTITY}),
        within_domain=True,
    ),
    max_depth=6,
    traced=True,
)
```

### OAK adapter

```python
from oaklib import get_adapter

adapter = get_adapter("omop:")  # the configured cdm_db

adapter.label("OMOP:313217")  # Atrial fibrillation
list(adapter.ancestors("OMOP:313217", reflexive=False))
list(adapter.basic_search("atrial fibrillation"))
```

See the [OAK interface](docs/oaklib/interface.md) for what it implements.

### Rendering

Outputs can be rendered as plain text, HTML (Jupyter), or Mermaid diagrams. Rendering auto-detects the environment.

```python
from IPython.display import HTML, display
from omop_graph.render import render_trace

display(HTML(render_trace(kg, trace)))
```

---

## Project Structure

```
omop_graph/
├── graph/          # the knowledge graph, its value types, paths and traversal
│   ├── model/      # immutable value types and scopes
│   ├── capabilities/  # the behaviour of the graph, as mixins
│   └── queries.py  # SQLAlchemy statement builders
├── render/         # HTML / text / Mermaid renderers
├── reasoning/      # grounding, scoring and path explanations
│   ├── resolvers/  # resolve labels via exact / fuzzy / full-text / synonym search
│   └── phenotypes/ # set operations for hierarchical groupings
├── oaklib_interface/  # OAK-compliant adapter
├── extensions/     # the relationship classification tables and data
├── data/           # the packaged relationship classification
└── db/             # session helpers
```

---

## Configuration

omop-graph reads database connection settings from
[oa-configurator](https://github.com/AustralianCancerDataNetwork/oa-configurator).
It requires the CDM database configured by omop-alchemy.

Run once after installation:

```bash
omop-config init
omop-config configure omop_alchemy
omop-config configure omop_graph
```

See [Configuration](docs/getting-started/configuration.md) for full details.
