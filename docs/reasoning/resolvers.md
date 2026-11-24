# Resolver Pipelines
The `ResolverPipeline` is intended to specify multiple resolvers searching in various ways for the string of the concept to be searched.
Let us consider the example of finding `"Hodgkin lymphoma"` as the concept to be searched. This could be coming from a LLM extracting it from free-text or just a simple case to see how the grounding works. 

The backbone of the `ResolverPipeline` are specific **resolvers**. `omop-graph` introduces the following:

- **`ExactLabelResolver`**: Exact case-insensitive match. Is there anywhere in the the [`concept`](https://ohdsi.github.io/CommonDataModel/cdm54.html#concept) table the exact string `"Hodgkin lymphoma"`
- **`ExactSynonymResolver`**: Similar to `ExactLabelResolver` yet searching the [`concept_synonym`](https://ohdsi.github.io/CommonDataModel/cdm54.html#concept_synonym) table
- **`PartialLabelResolver`**: Substring match. Is the search string `"Hodgkin lymphoma"` a partial component of any [`concept`](https://ohdsi.github.io/CommonDataModel/cdm54.html#concept)?
- **`PartialSynonymResolver`**: Similar to `PartialLabelResolver` yet searching the [`concept_synonym`](https://ohdsi.github.io/CommonDataModel/cdm54.html#concept_synonym) table
- **`FullTextResolver`**: Matches irrespective of word order. Not as relevant for the example above but relevant for others (e.g., "Kidney Cancer" -> "Cancer of Kidney").
- **`FullTextSynonymResolver`**: Similar to `FullTextResolver` yet searching the [`concept_synonym`](https://ohdsi.github.io/CommonDataModel/cdm54.html#concept_synonym) table
- **`EmbeddingResolver`**: Vector-similarity match, appended to the pipeline only when `omop-graph[emb]` (`omop-emb`) is installed.

!!! tip
    
    Traversing each of the resolvers one by one can be an exhaustive search. The `ResolverPipeline` therefore offers a `stop_after_resolver` option. If set, retrieval from the DB stops after that resolver has concluded. `ALL_RESOLVERS`, the default sequence, is ordered exactly as listed above (i.e. **`ExactLabelResolver`** >> **`ExactSynonymResolver`** >> **`PartialLabelResolver`** >> ... >> **`EmbeddingResolver`**).

```python
from omop_graph.reasoning.resolvers import (
    ExactLabelResolver,
    ExactSynonymResolver,
    ResolverPipeline,
)

# Every resolver, in the order above
pipeline = ResolverPipeline.with_all_resolvers()

# Or some resolvers, stopping after one has run. The order of the resolvers matters
pipeline = ResolverPipeline(
    (ExactLabelResolver(), ExactSynonymResolver()),
    stop_after_resolver=ExactLabelResolver,
)

candidates = pipeline.resolve(kg, "Hodgkin lymphoma", constraints=concept_filter)
```

The pipeline is used by [`ground_term`](grounding.md), together with the `GroundingConstraints` described there. The `constraints` argument of `resolve` is a `ConceptFilter`, which restricts the candidates by domain, vocabulary and standard or active status.
