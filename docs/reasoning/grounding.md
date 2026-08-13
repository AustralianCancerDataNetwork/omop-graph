# Semantic Grounding

Grounding is the process of mapping a raw string to a **Standard OMOP Concept ID**, i.e. a standardised Ontology.
This extraction of standardised concepts from free-text and results in efficient information extraction.
Traditional string matching fails on clinical free-text due to synonyms and ambiguity. By integrating **Database Constraints** and **Hierarchical Reasoning**, `omop-graph` ensures that an extraction of "Heart Attack" is correctly mapped to `OMOP:312327` (Acute myocardial infarction) and validated as a `Condition`.

!!! info

    The backbone of this capability is the [Knowledge Graph](../graph/kg.md)

## Approach overview

To accelerate the grounding to standard concepts, `omop-graph` makes use of:

- [pre-classified relationships](../graph/edges.md),
- [the virtual knowledge graph](../graph/kg.md)


!!! tip

    The following steps summarise the entire grounding approach and are found in `omop_graph.reasoning.grounding`

1. **Configuration**: Determine graph restrictions using [`GroundingConstraints`](#grounding-constraints)
    - `parent_ids`: OMOP Concept IDs that act as required ancestors — any valid result must be a descendant of at least one of these.
    - `search_constraint`: A [`ConceptFilter`](#conceptfilter) that filters the initial resolver query by domain, vocabulary, active status, and/or standard status.
    - `max_depth` / `predicate_kinds`: Control how far and along which relationship kinds the anchor walk is allowed to travel.

2.  **Resolve**: Use the [`ResolverPipeline`](resolvers.md) to find any concepts (Standard or Non-Standard) matching the text.
3.  **Anchor**: For each candidate, find the nearest **Standard Concept**. This is required for Step 3 as all standard concepts are in `concept_ancestor`.
    - If the candidate is already Standard, the hop count is 0.
    - If Non-Standard, follow `IDENTITY` relationship to the next standard concept.
4.  **Verify**: Check the `concept_ancestor` table to see if the Standard Anchor is a descendant of the required `parent_ids`. 
    - Requires [`GroundingConstraints`](#grounding-constraints) for accurate grounding/verification
5.  **Scoring**: Apply the scoring algorithm to the resulting valid Standard Concepts
    - Details of scoring algorithm shown [here](#scoring)

## Grounding Constraints

`GroundingConstraints` is composed of two layers:

| Field | Type | Default | Purpose |
|---|---|---|---|
| `parent_ids` | `tuple[int, ...]` | `None` | Only accept candidates that are descendants of these OMOP concept IDs (hierarchy validation via `concept_ancestor`). |
| `search_constraint` | `ConceptFilter` | `None` | Filters applied to the initial resolver query (concept IDs, domain, vocabulary, standard/active flags, and limit). |
| `max_depth` | `int` | `6` | Maximum hop distance allowed between a candidate and its standard anchor. |
| `predicate_kinds` | `frozenset[PredicateKind]` | `{IDENTITY}` | Relationship kinds followed when walking from a non-standard candidate to its standard anchor. |

### ConceptFilter

`ConceptFilter`, provided by OMOP Alchemy, controls which concepts are even considered as candidates during the resolve step. All fields are optional and composable:

| Field | Type | Default | Purpose |
|---|---|---|---|
| `concept_ids` | `tuple[int, ...]` | `None` | Restrict to a specific set of concept IDs. |
| `domains` | `tuple[str, ...]` | `None` | Restrict by OMOP Domain ID (e.g. `"Condition"`, `"Drug"`). |
| `vocabularies` | `tuple[str, ...]` | `None` | Restrict by Vocabulary ID (e.g. `"SNOMED"`, `"RxNorm"`). |
| `require_standard` | `bool` | `False` | When `True`, only concepts with `standard_concept` in `('S', 'C')` are returned. |
| `require_active` | `bool` | `False` | When `True`, only concepts with an unset `invalid_reason` are returned. |
| `limit` | `int` | `None` | Cap the number of candidates returned from the resolver query. |

### Example

```python
from omop_graph.reasoning.grounding import ground_term, GroundingConstraints
from omop_alchemy.cdm.query import ConceptFilter
from omop_graph.extensions.omop_alchemy import PredicateKind

constraints = GroundingConstraints(
    parent_ids=(441484,),   # 'Clinical Finding' — only accept descendants of this ancestor
    search_constraint=ConceptFilter(
        domains=("Condition",),
        vocabularies=("SNOMED",),
        require_standard=True,
    ),
    max_depth=6,
    predicate_kinds=frozenset({PredicateKind.IDENTITY}),
)

results = ground_term(
    resolver_pipeline=pipeline,
    kg=kg,
    query="chest pain",
    query_embedding=None,   # pass a precomputed vector, or None to skip embedding-based scoring
    constraints=constraints,
)
```

`ground_term` also accepts an optional `context: str | None` folded into the on-demand query-embedding text (used only when `query_embedding` is omitted and the KG has a write-capable embedding config); see [`ground_term`'s own docstring](../reference/ref_grounding.md) for details.

## Scoring
It usually happens that multiple viable candidates are extracted for each search term, especially if multiple resolvers are used. To rank these exctracted concepts, we devised a scoring algorithm, which is detailed in the following:

### The Scoring Formula

The total score for a concept is calculated as:

$$
TotalScore = Relevance - ParsimonyPenalty + BroadnessBonus
$$

#### 1. Relevance
Relevance represents the initial semantic fit and is computed as **either** embedding similarity **or** textual similarity — not both simultaneously:

- **Without embeddings**: textual similarity is used exclusively.
- **With embeddings** (default when `omop-graph[emb]` is installed and configured): embedding cosine similarity **replaces** the textual score entirely.

The two scoring modes:

- **Embedding Similarity**: Cosine similarity between the input text embedding and the concept embedding. Requires `omop-graph[emb]` and a configured `KnowledgeGraphEmbeddingConfiguration` — see the [Knowledge Graph docs](../graph/kg.md#embedding-configuration) and the [omop-emb documentation](https://australiancancerdatanetwork.github.io/omop-emb/) for setup.
- **Textual Similarity**: A custom token-overlap score that heavily penalizes missing words from the user's query but allows for "extra" descriptive words in the OMOP concept name. Used as a fallback when no embedding is available.

#### 2. Parsimony: Distance Penalty
OMOP is a deep hierarchy. A concept that is 1 hop away from your search term is more likely to be correct than one found 5 hops away.

- **Formula**: $\alpha \times separation$
- We apply a penalty for every "hop" in the graph required to reach a standard concept.

#### 3. Broadness: Generality Bonus
In clinical coding, we often prefer a specific match, but when choosing between two equally relevant concepts, the "Broadness" bonus rewards concepts that have a well-defined place in the hierarchy.

- **Formula**: $\beta \times \ln(1 + AncestorCount)$
- Concepts with more ancestors are higher up in the hierarchy. This bonus helps "tie-break" by favoring well-established standard concepts.

### Implementation
Scoring is performed in a batch operation to minimize database overhead:

```python
from omop_graph.graph.scoring import score_standard_concepts

scored = score_standard_concepts(
    text="Hodgkin lymphoma",
    standard_concepts=candidates,
    kg=kg,
    nearest_concept_matches=nearest_matches,  # optional; from omop-emb embedding index
)
ranked = sorted(scored, key=lambda s: s.total_score, reverse=True)
```
