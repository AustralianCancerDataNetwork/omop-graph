# omop-graph

**omop-graph** is a lightweight, opinionated knowledge-graph traversal and path-analysis library built on top of the OMOP Common Data Model.

It transforms the static OMOP vocabulary tables into a dynamic graph environment suitable for NLP grounding, phenotype construction, and clinical reasoning.

## Why omop-graph?

Unlike generic graph libraries, `omop-graph` is built specifically for clinical data:
- **Semantic Awareness**: Understands the difference between an "Is a" relationship and a "Maps to" relationship.
- **Efficient Grounding**: Instead of traversing every possible path, the library uses a **Standard Anchor** approach—translating non-standard terms to standard concepts and leveraging the OMOP `concept_ancestor` table for high-speed hierarchy validation.
- **Transparent Scoring**: Decisions aren't black boxes. Every path is scored based on textual similarity, graph distance (parsimony), and clinical generality (broadness).

---

## Documentation Overview

### Core Components
Learn how the **KnowledgeGraph Facade** provides a clean API over SQLAlchemy and how OMOP relationships are classified into semantic kinds.
- [KnowledgeGraph](graph/kg.md)
- [Edges & Predicates](graph/edges.md)

### Algorithms & Reasoning
Explore the pathfinding logic and the grounding pipeline used by clinical NLP tools.
- [Pathfinding](algorithms/paths.md)
- [Scoring](algorithms/scoring.md)
- [Grounding & Resolvers](reasoning/grounding.md)

### Interactive Exploration
`omop-graph` includes built-in HTML renderers for Jupyter Notebooks, allowing you to visualize concepts and relationship summaries instantly.