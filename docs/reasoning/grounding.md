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

0. **Configuration**: Determine graph restrictions using [`GroundingConstraints`](#grounding-constraints)
    - `parent_id`: The `concept_id` of the parent Ontology. This attribute is **required** and allows testing whether a standard concept is part of the correct branch
    - `domains`: The OMOP CDM domains that are allowed to be searched for. Each Ontolgoy has an associated domain as described in the [OMOP CDM](https://ohdsi.github.io/CommonDataModel/cdm54.html#concept). Specifying multiple permits all specified domains.
    - `vocabs`: The OMOP CDM vocabularies that are allowed to be searched for. Each Ontology is also part of a vocabulary as described in the [OMOP CDM](https://ohdsi.github.io/CommonDataModel/cdm54.html#concept). Specifying multiple values permits all specified vocabularies.

1.  **Resolve**: Use the [`ResolverPipeline`](resolvers.md) to find any concepts (Standard or Non-Standard) matching the text.
2.  **Anchor**: For each candidate, find the nearest **Standard Concept**. This is required for Step 3 as all standard concepts are in `concept_ancestor`.
    - If the candidate is already Standard, the hop count is 0.
    - If Non-Standard, follow `IDENTITY` relationship to the next standard concept.
3.  **Verify**: Check the `concept_ancestor` table to see if the Standard Anchor is a descendant of the required `parent_ids`. 
    - Requires [`GroundingConstraints`](#grounding-constraints) for accurate grounding/verification
4.  **Scoring**: Apply the scoring algorithm to the resulting valid Standard Concepts
    - Details of scoring algorithm shown [here](#scoring)

## Grounding Constraints
You can restrict the search using `GroundingConstraints`:
- **parent_ids**: Only return concepts that fall under these ancestors (e.g., only search within "Procedures").
- **search_constraint**: Limit search to specific vocabularies or domains (e.g., "RxNorm" only).
- `parent_ids`: Restricts the search to descendants of specific OMOP concepts (e.g., only search for concepts under `Condition`).
- `vocabs`: Restricts the search to specific vocabularies (e.g., `SNOMED`, `RxNorm`).
- `domains`: Restricts by OMOP Domain ID (e.g., `Condition`, `Drug`).

```python
from omop_graph.reasoning.grounding import ground_term, GroundingConstraints

constraints = GroundingConstraints(
    parent_ids=(441484,), # 'Clinical Finding'
    max_depth=6
)

results = ground_term(pipeline, kg, "chest pain", constraints)
```

## Scoring
It usually happens that multiple viable candidates are extracted for each search term, especially if multiple resolvers are used. To rank these exctracted concepts, we devised a scoring algorithm, which is detailed in the following:

### The Scoring Formula

The total score for a concept is calculated as:

$$
TotalScore = Relevance - ParsimonyPenalty + BroadnessBonus
$$

#### 1. Relevance
Relevance represents the initial semantic fit. It is the product of:

- **Embedding Similarity**: Cosine similarity between the input text and the concept name.
- **Textual Similarity**: A custom token-overlap score that heavily penalizes missing words from the user's query but allows for "extra" descriptive words in the OMOP concept name.

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

ranked = score_standard_concepts(
    text="Hodgkin lymphoma",
    standard_concepts=candidates,
    kg=kg,
    similarity_scores=embeddings_array
)
```