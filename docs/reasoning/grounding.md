# Semantic Grounding

Grounding is the process of mapping a raw string to a **Standard OMOP Concept ID** while respecting hierarchical constraints.

## The Standard Anchor Approach

Older versions of this library attempted to traverse every topological path between a candidate and a parent. This was computationally expensive and ignored the optimization provided by the OMOP `concept_ancestor` table.

The new **Standard Anchor** algorithm follows these steps:

1.  **Resolve**: Use the `ResolverPipeline` to find any concepts (Standard or Non-Standard) matching the text.
2.  **Anchor**: For each candidate, find the nearest **Standard Concept**.
    - If the candidate is already Standard, the hop count is 0.
    - If Non-Standard, follow "Maps to" or "Versioning" edges to find the Standard equivalent.
3.  **Verify**: Check the `concept_ancestor` table to see if the Standard Anchor is a descendant of the required `parent_ids`.
4.  **Rank**: Apply the scoring algorithm to the resulting valid Standard Concepts.

## Grounding Constraints
You can restrict the search using `GroundingConstraints`:
- **parent_ids**: Only return concepts that fall under these ancestors (e.g., only search within "Procedures").
- **search_constraint**: Limit search to specific vocabularies or domains (e.g., "RxNorm" only).

```python
from omop_graph.reasoning.grounding import ground_term, GroundingConstraints

constraints = GroundingConstraints(
    parent_ids=(441484,), # 'Clinical Finding'
    max_depth=6
)

results = ground_term(pipeline, kg, "chest pain", constraints)
```