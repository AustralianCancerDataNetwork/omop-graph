# Architecture

The package is layered. At runtime, the first four layers import only from the layers listed before them. Annotations are the one exception, since `graph.model.edges` names `KnowledgeGraph` under `TYPE_CHECKING`.

| Layer | Package | Holds |
|---|---|---|
| Relationship classification | `omop_graph.extensions` | The tables that store the classification of relationships, their loader, and the enums that name its kinds, such as `PredicateKind`. They extend the CDM schema of omop-alchemy. |
| Value types | `omop_graph.graph.model` | Immutable data and enums with no database access: `EdgeFilter`, `EdgeView`, `Predicate`, `ConceptView`, scopes, concept counts and similarity results. |
| Queries | `omop_graph.graph.queries` | SQLAlchemy statement builders, one `q_*` function per question. They return statements and never execute them. |
| Capabilities | `omop_graph.graph.capabilities` | The behaviour of the graph as mixins. Each runs queries through the session factory and returns model types. |
| Knowledge graph | `omop_graph.graph.kg` | `KnowledgeGraph`, the class that combines the capabilities. |
| Algorithms | `omop_graph.graph.paths`, `traverse`, `scoring` | Path search, traversal and candidate scoring on top of a `KnowledgeGraph`. |
| Reasoning | `omop_graph.reasoning` | Resolvers and grounding, which turn text into standard concepts. |
| Adapter | `omop_graph.oaklib_interface` | The oaklib implementation. It translates CURIEs and oaklib types around `KnowledgeGraph` methods, and holds no query logic of its own. |
| Rendering | `omop_graph.render` | Text, HTML and Mermaid views of paths and traces. |

## Why the types sit apart from the capabilities

The queries import the value types, such as `EdgeFilter` and `Grouping`, to build their statements. The capabilities import the queries. A type defined in a capability file would make that file and the queries import each other, so the types live below both, in `graph.model`.

## Capabilities

The mixins are layered, and each inherits the ones it uses. The table in [Knowledge Graph](kg.md#capabilities) lists them with their methods. `KnowledgeGraph` lists the derived mixins before their bases, which keeps the method resolution order consistent.

A capability reads the database only through `GraphCore.session_factory`. A query that concerns one capability is still kept in `queries.py`, so that all SQL is in one module.

## Scope

A scope is a `ConceptFilter` from omop-alchemy. The methods that report or count concepts take one, and `omop_graph.graph.model.scope` holds the common scopes and the helpers that derive one scope from another.

## Optional parts

Embeddings need the optional `omop-emb` package. Only `omop_graph.graph.capabilities.embeddings` imports it when it is used. `omop_graph.graph.scoring` and `omop_graph.reasoning.grounding` import its types under `TYPE_CHECKING`, so the whole package imports without it.

## Package root

`omop_graph` re-exports the names most callers need, such as `KnowledgeGraph`, `EdgeFilter`, `GraphPath` and `GroundingConstraints`. Everything else is imported from its module.
