# omop-graph

**omop-graph** is a lightweight Virtual Knowledge Graph (VKG) built on top of the OMOP CDM.
It transforms the static OMOP vocabulary tables into a dynamic graph environment suitable for NLP grounding, clinical reasoning, and other tasks that benefit from a knowledge graph.

## Why omop-graph?

Unlike generic graph libraries, `omop-graph` is built specifically for clinical data:

- **Semantic Awareness**: Understands the difference between relationship kinds (hierarchy, identity, composition, association, attribute).
- **Efficient Grounding**: Instead of traversing every possible path, the library uses a **Standard Anchor** approach — translating non-standard terms to standard concepts and leveraging the OMOP `concept_ancestor` table for high-speed hierarchy validation.
- **Transparent Scoring**: Decisions aren't black boxes. Every candidate concept is scored based on textual similarity, graph distance (parsimony), and clinical generality (broadness).
- **Pre-classification**: Relationships are pre-classified into semantic groups, enabling quicker traversal restrictions and more targeted reasoning.

---

## Documentation Overview

### Core Components
- [KnowledgeGraph](graph/kg.md): The VKG interface — connecting to OMOP and traversing the graph.
- [Relationships](graph/edges.md): Pre-classification of OMOP edges into semantic kinds.
- [Oaklib Interface](oaklib/interface.md): OAK-compliant adapter for cross-ontology tooling.

### Reasoning
Explore the grounding pipeline used by clinical NLP tools.

- [Semantic Grounding](reasoning/grounding.md): Mapping free-text terms to standard OMOP concepts.
- [Resolver Pipelines](reasoning/resolvers.md): How candidate concepts are retrieved from the database.

### Embedding Support

!!! info "Powered by omop-emb"
    Embedding-based similarity (vector search, RAG retrieval, on-the-fly embedding computation) is provided by the companion [`omop-emb`](https://australiancancerdatanetwork.github.io/omop-emb/) package.
    Install it with `pip install "omop-graph[emb]"` and see [Knowledge Graph — Embedding Configuration](graph/kg.md#embedding-configuration) for integration details.

### Interactive Exploration
`omop-graph` includes built-in HTML and Mermaid renderers for Jupyter Notebooks, allowing you to visualise concepts, traversal traces, and relationship summaries directly in a notebook.

### Testing
- [Testing](usage/testing.md): Test configuration, coverage, and how to set up environment variables for local runs.
