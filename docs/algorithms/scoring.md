# Path Scoring & Ranking

When multiple candidate concepts are found for a clinical term, `omop-graph` applies a multi-dimensional scoring algorithm to determine the "Best Match."



## The Scoring Formula

The total score for a concept is calculated as:
$$TotalScore = Relevance - ParsimonyPenalty + BroadnessBonus$$

### 1. Relevance
Relevance represents the initial semantic fit. It is the product of:
- **Embedding Similarity**: Cosine similarity between the input text and the concept name.
- **Textual Similarity**: A custom token-overlap score that heavily penalizes missing words from the user's query but allows for "extra" descriptive words in the OMOP concept name.

### 2. Parsimony (The Distance Penalty)
OMOP is a deep hierarchy. A concept that is 1 hop away from your search term is more likely to be correct than one found 5 hops away.
- **Formula**: $\alpha \times separation$
- We apply a penalty for every "hop" in the graph required to reach a standard concept.

### 3. Broadness (The Generality Bonus)
In clinical coding, we often prefer a specific match, but when choosing between two equally relevant concepts, the "Broadness" bonus rewards concepts that have a well-defined place in the hierarchy.
- **Formula**: $\beta \times \ln(1 + AncestorCount)$
- Concepts with more ancestors are higher up in the hierarchy. This bonus helps "tie-break" by favoring well-established standard concepts.

## Implementation
Scoring is performed in a batch operation to minimize database overhead:

```python
from omop_graph.graph.scoring import score_standard_concepts

ranked = score_standard_concepts(
    text="Acute Myocardial Infarction",
    standard_concepts=candidates,
    kg=kg,
    similarity_scores=embeddings_array
)
```