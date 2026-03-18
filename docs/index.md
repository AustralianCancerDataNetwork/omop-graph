# omop-graph

**omop-graph** is a lightweight virtual knowledge Graph (VKG) built on-top of the OMOP CDM.
It transforms the static OMOP vocabulary tables into a dynamic graph environment suitable for NLP grounding, clinical reasoning and other tasks that benefit from a knowledge graph.

## Why omop-graph?

Unlike generic graph libraries, `omop-graph` is built specifically for clinical data:

- **Semantic Awareness**: Understands the difference between relationships.
- **Efficient Grounding**: Instead of traversing every possible path, the library uses a **Standard Anchor** approach: translating non-standard terms to standard concepts and leveraging the OMOP `concept_ancestor` table for high-speed hierarchy validation.
- **Transparent Scoring**: Decisions aren't black boxes. Every path is scored based on textual similarity, graph distance (parsimony), and clinical generality (broadness).
- **Pre-classification**: Relationships are already pre-classified into overarching groups, allowing quicker restrictions of connections and more efficient graph traversal.
---

## Documentation Overview

### Core Components
- [KnowledgeGraph](graph/kg.md): The VKG interface and what it attempts to solve.
- [Relationships](graph/edges.md): Pre-classification of edges/relationships of the OMOP CDM.
- [Oaklib Interface](oaklib/interface.md): `oaklib`-compliant interface

### Reasoning
Explore the grounding pipeline used by clinical NLP tools.

- [Semantic grounding](reasoning/grounding.md): How regular search terms can be traced to a standard Ontology

### Interactive Exploration
`omop-graph` includes built-in HTML renderers for Jupyter Notebooks, allowing you to visualize concepts and relationship summaries instantly.