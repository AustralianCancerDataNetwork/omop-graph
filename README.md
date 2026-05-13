# omop-graph

**omop-graph** is a lightweight, opinionated knowledge-graph traversal and path-analysis library built on top of the OMOP vocabulary model.

It provides:
- a stable **KnowledgeGraph façade** over OMOP concepts and relationships
- flexible **graph traversal** (forward, backward, bidirectional)
- **path discovery** with transparent scoring
- **traceable explanations** of traversal decisions
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

With embedding support (sqlite-vec backend, zero config):

```bash
pip install "omop-graph[emb]"
```

For larger deployments use `[pgvector]` or `[faiss-cpu]` instead (or in addition).
Full setup is covered in the [omop-emb documentation](https://australiancancerdatanetwork.github.io/omop-emb/).

---

## Core Concepts

### KnowledgeGraph

`KnowledgeGraph` is the main entry point. It wraps a SQLAlchemy `Engine` connected to an OMOP vocabulary schema and provides a high-level Pythonic API over the relational tables.

```python
from sqlalchemy import create_engine
from omop_graph.graph.kg import KnowledgeGraph

engine = create_engine("postgresql://user:pass@localhost/omop")
kg = KnowledgeGraph(engine)

# Lookup a concept by label
match_group = kg.label_lookup("Atrial Fibrillation", fuzzy=False)
concept = match_group.best_match
print(f"ID: {concept.concept_id}, Name: {concept.matched_label}")

# Traverse the hierarchy
parents = kg.parents(concept.concept_id)
```

### Nodes and Edges

Nodes are OMOP Concepts; Edges are OMOP Concept_Relationships.

Relationships are pre-classified into semantic kinds (`ClassIDEnum`):

- `HIERARCHY` — parent/child ontological relationships
- `IDENTITY` — mapping to standard concepts
- `COMPOSITION` — part-of relationships
- `ASSOCIATION` — lateral clinical associations
- `ATTRIBUTE` — concept attribute relationships

This classification drives traversal filtering and scoring.

### Traversal and Paths

```python
from omop_graph.graph.paths import find_shortest_paths
from omop_graph.extensions.omop_alchemy import ClassIDEnum

ingredient = kg.concept_id_by_code("RxNorm", "6809")    # Metformin
drug = kg.concept_id_by_code("RxNorm", "860975")         # Metformin 500 MG Oral Tablet

paths, trace = find_shortest_paths(
    kg,
    source=drug,
    target=ingredient,
    predicate_kinds=frozenset({ClassIDEnum.HIERARCHY, ClassIDEnum.IDENTITY}),
    max_depth=6,
    traced=True,
)
```

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
├── graph/          # graph logic, traversal, paths, scoring
├── render/         # HTML / text / Mermaid renderers
├── reasoning/      # ontology traversal methods for specific reasoner tasks
│   ├── resolvers/  # resolve labels via exact / fuzzy / full-text / synonym search
│   └── phenotypes/ # set operations for hierarchical groupings
├── oaklib_interface/  # OAK-compliant adapter
├── api.py          # stable public API surface
└── db/             # session helpers
```
