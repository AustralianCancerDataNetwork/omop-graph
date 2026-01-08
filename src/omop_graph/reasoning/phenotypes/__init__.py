from .phenotype_simplifier import find_common_parents

__all__ = [
    "find_common_parents",
]

"""
For phenotype development, we want to identify sets of concepts that are optimally situated in the semantic hierarchy.

e.g. regex hits on labels could give candidate concepts (e.g. ICDO morph code), but this may or may not correspond to an easily identifiable portion of the target hieararchy.

We want to have concept sets that are as compact as possible, while still covering the candidate concepts, and introducing as few extraneous concepts as possible.

Algorithmically:

* Start with a seed set `S` of concept IDs
(e.g. ICD-O-3 morphologies, regex hits, NLP outputs)
* We must find a parent set P such that:
    Every s ∈ S is a descendant of at least one p ∈ P
    P is small (not just S itself)
    P is pure (most of its descendants are in S)
* We need to support:
    multi-inheritance
    parents at different depths
    some contamination may be acceptable (e.g. 10% of descendants not in S), depending on use case

Definitions:
For any candidate parent node p:
    Desc(p) = all descendants of p (bounded by max depth)
    Target(p) = Desc(p) ∩ S
    Coverage(p) = |Target(p)|
    Pollution(p) = |Desc(p) - S|
    Purity(p) = Coverage(p) / |Desc(p)|
Coverage constraint:
    |⋃ Target(pᵢ)| ≥ C_min
Purity constraint:
    Purity(pᵢ) ≥ P_min  ∀ pᵢ -> OR (sum Target) / (sum Desc) ≥ P_min
Compression constraint:
    |P| << |S|
Depth bias:
    prefer parents closer to S

Approach:
    Pareto frontier optimisation over these objectives/constraints to identify optimal parent sets.


Pseudo-code:

Step 1: Build ancestor profiles for each seed
    For each s ∈ S:
        A = Ancestors(s, max_depth)    
        for each a ∈ A:
            candidates.append(a)
            tally[a].coverage += 1
            tally[a].pollution += |Desc(a) - S|

Step 2: Precompute candidate parent statistics
    For each c in candidates:
        coverage = tally[c].coverage
        pollution = tally[c].pollution
        purity = coverage / (coverage + pollution)
        depth = depth(c)

Step 3: Filter aggressively

| ancestor | coverage | total_desc | purity | depth |
| -------- | -------- | ---------- | ------ | ----- |


MIN_COVERAGE = 2           # must explain ≥ 2 seeds
MIN_PURITY = 0.3           # configurable
MAX_DESC = 1_000           # stop SNOMED-root nonsense

Step 4: Greedy set cover with purity-aware scoring
    remaining = set(S)
    parents = []

    while remaining and coverage_ratio(remaining) < target:
        best = argmax_a(score(a ∩ remaining))
        parents.append(best)
        remaining -= Target(best)

Step 5: Optional refinement pass
    After greedy selection:
        Remove parents that contribute < ε coverage
        Merge parents if one subsumes another with similar purity
"""