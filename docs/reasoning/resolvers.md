# Resolver Pipelines
The `ResolverPipeline` is intended to specify multiple resolvers searching in various ways for the string of the concept to be searched.
Let us consider the example of finding `"Hodgkin lymphoma"` as the concept to be searched. This could be coming from a LLM extracting it from free-text or just a simple case to see how the grounding works. 

The backbone of the `ResolverPipeline` are specific **resolvers**. `omop-graph` introduces the following:

- **`ExactLabelResolver`**: Exact case-insensitive match. Is there anywhere in the the [`concept`](https://ohdsi.github.io/CommonDataModel/cdm54.html#concept) table the exact string `"Hodgkin lymphoma"`
- **`ExactSynonymResolver`**: Similar to `ExactLabelResolver` yet searching the [`concept_synonym`](https://ohdsi.github.io/CommonDataModel/cdm54.html#concept_synonym) table
- **`FullTextResolver`**: Matches irrespective of word order. Not as relevant for the example above but relevant for others (e.g., "Kidney Cancer" -> "Cancer of Kidney").
- **`FullTextSynonymResolver`**: Similar to `FullTextResolver` yet searching the [`concept_synonym`](https://ohdsi.github.io/CommonDataModel/cdm54.html#concept_synonym) table
- **`PartialLabelResolver`**: Substring match. Is the search string `"Hodgkin lymphoma"` a partial component of any [`concept`](https://ohdsi.github.io/CommonDataModel/cdm54.html#concept)?
- **`PartialSynonymResolver`**: Similar to `PartialLabelResolver` yet searching the [`concept_synonym`](https://ohdsi.github.io/CommonDataModel/cdm54.html#concept_synonym) table

!!! tip
    
    Traversing each of the resolvers one by one can be an exhaustive search. The `ResolverPipeline` therefore offers the option of a `stop_after_confidence` attribute. If set, the retrieval from the DB stop after that respective resolver has concluded. The resolvers are ordered based on their confidence as above (i.e. **`ExactLabelResolver`** >> **`ExactSynonymResolver`** >> etc.)

```python
from omop_graph.graph.nodes import LabelMatchKind
from omop_graph.reasoning.grounding import GroundingConstraints

# Simplified Logic View
constraints = GroundingConstraints(
    parent_ids=parsed_parent_ids,
    search_constraint=SearchConstraintConcept(
        domains=parsed_domains,
        vocabs=parsed_vocabs,
        require_standard=True  # We prioritize Standard Concepts
    ),
    max_depth=6  # How deep to traverse the hierarchy
)

# The resolver pipeline stops once it finds a high-confidence match
resolver_pipeline = ResolverPipeline.with_all_resolvers()
# Optionally you can also set the stopping on a specific resolver. The order of the resolvers matters
# resolver_pipeline = ResolverPipeline(<your_tuple_of_resolvers>, ExactLabelResolver)
```