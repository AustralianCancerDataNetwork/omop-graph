# Paths and Traversal

Path search and traversal run on a `KnowledgeGraph` and follow the edges its relationship classification allows. Both take an `EdgeFilter`, see [Relationships](edges.md).

## Shortest paths

`find_shortest_paths` searches from both ends at once and returns the shortest paths, each once, up to `max_paths`, as `GraphPath` objects.

```python
from omop_graph.graph.paths import find_shortest_paths
from omop_graph.render import render_path

paths, trace = find_shortest_paths(kg, source_id, target_id, traced=True)
for path in paths:
    print(render_path(kg, path))
```

* A path follows each edge from its subject to its object, so the search from the target walks edges backwards.
* The default filter follows active edges within one domain. Pass `EdgeFilter(within_domain=False)` to cross domains, for example along `Has finding site`.
* `max_depth` is the most steps a path may have, as in `traverse`. The two ends of the search share it, and paths longer than it are not found.
* Each step reads the edges of a whole frontier in one query, from the end with fewer concepts to expand, so the queries made depend on the depth of the search and not on the number of concepts. The edges of one frontier are held in memory at a time.
* A search stops as soon as an end has nothing left to expand, because then no path exists.
* When there are more than `max_paths` shortest paths, the ones through the lowest concept id where the two ends meet come first. The result does not depend on the order in which the database returns the edges.
* A concept to itself gives one empty path.
* `traced=True` also returns a `GraphTrace` of the search, which shows why a path was or was not found. It lists each concept expanded with the edges read from it and its depth, counted from the end of the search that expanded it, and says why the search stopped as a `TerminationReason`.
* `edge_filter`, `max_depth`, `max_paths` and `traced` are keyword arguments.

A `GraphPath` holds its `steps`, each a subject, a relationship id and an object. `nodes()` lists the concept ids along it.

## Traversal

`traverse` walks outwards from seed concepts breadth first.

```python
from omop_graph.graph.traverse import traverse

subgraph, trace = traverse(
    kg, [concept_id], edge_filter=None, max_depth=2, max_nodes=500, trace=True
)
```

* The result is a `Subgraph` of nodes and edges.
* `max_nodes` stops the walk once that many concepts are found. Edges are read in key order, so the same concepts fill the budget on every run.
* The edges of a concept become objects one at a time, and the walk stops as soon as the budget is reached, so a hub concept is not expanded in full.

## Explaining a path

`PathExplanation.from_path(kg, path, trace)` in `omop_graph.reasoning.explanation` combines a path with the trace of its search. It gives the concept the path resolves to, a `PathProfile` and for each step the kind of its relationship and the depth at which the search read it, counted from the end of the search that found it. The result renders with [`render_explained_path`](rendering.md).
