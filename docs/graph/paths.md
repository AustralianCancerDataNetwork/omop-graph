# Paths and Traversal

Path search and traversal run on a `KnowledgeGraph` and follow the edges its relationship classification allows. Both take an `EdgeFilter`, see [Relationships](edges.md).

## Shortest paths

`find_shortest_paths` searches from both ends at once and returns the shortest paths, each once, up to `max_paths`, as `GraphPath` objects.

```python
from omop_graph.graph.paths import find_shortest_paths

paths, trace = find_shortest_paths(kg, source_id, target_id, traced=True)
for path in paths:
    print(path.explain(kg))
```

* A path follows each edge from its subject to its object, so the search from the target walks edges backwards.
* The default filter follows active edges within one domain. Pass `EdgeFilter(within_domain=False)` to cross domains, for example along `Has finding site`.
* `max_depth` is the depth searched from each end, so a shortest path of up to twice as many steps is found. Longer paths are not.
* When there are more than `max_paths` shortest paths, the ones through the lowest concept id where the two searches meet come first. The order within them follows the order of the edges, which is not specified.
* A concept to itself gives one empty path.
* `traced=True` also returns a `GraphTrace` of the search, which shows why a path was or was not found.
* `find_shortest_paths_batch` finds the same paths but reads the edges of a whole frontier in one query, which suits a database with many edges per concept. It holds the edges of one frontier in memory at a time.

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

`PathExplanation.from_path(kg, path, trace, match_kind)` in `omop_graph.reasoning.explanation` combines a path with the trace of its search. It gives the concept the path resolves to, a `PathProfile`, and for each step the kind of its relationship and the depth at which the search expanded it. The result renders with [`render_explained_path`](rendering.md).
