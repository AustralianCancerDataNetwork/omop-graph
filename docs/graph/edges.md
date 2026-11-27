# Classification of Edges

To provide further reasoning capabilities beyond the conventional OMOP CDM, we decided to classify the majority of edges into pre-defined categories. These categories are split into:

- `class`: Parent categorisation of edges. Currently one of 
    - `Association`, 
    - `Attribute`,
    - `Composition`, 
    - `Hierarchy`, 
    - `Identity`,
    - `Metadata`.
- `subclass`: Subclass to further differentiate between edges of the same class. Fine-grained reasoning capabilities including description for the LLM regarding semantics and inference. These are not as limited as the parent `class`.

`Metadata` relationships carry release and data-model bookkeeping (module, status, data type, CDM version, schema variable, item dates). They describe the vocabulary itself, not clinical content.

Each `Hierarchy` relationship has a `direction`, stored in `relationship_mapping.direction` and read from the stored edge `concept_id_1 -> concept_id_2`. `up` means `concept_id_2` is the parent or group of `concept_id_1`, `down` means the reverse. A relationship and its OMOP reverse therefore always have opposite directions. No other class has a direction.

## Predicate Classification

To allow reproduction and evaluation of this approach, we provide clear guidelines how we classified edge predicates of the OMOP CDM into the aforementioned groups. The following table provides exact descriptions about each `class` and `subclass`.

??? "Expand to see the grouping classification of predicates"
    
    {{ to_grouped_table('src/omop_graph/data/predicate_classification.csv', [0, 1], [0, 1, 2, 3, 4], [0, 1],) }}

## Predicate Mappings
Following the predicate classification guidelines of the previous section, we classified the following predicates into their respective classification groups.

!!! warning
    
    This classification is currently still under development and most likely may change with increased feedback from clinicians. The respective interface to store these classifications in the OMOP CDM has been prepared and we are in talks to potentially include this classification eventually in the official OMOP CDM.

??? "Expand to see the classification of all edge connections"
    
    {{ to_grouped_table('src/omop_graph/data/predicate_mapping.csv', [0, 1], [0, 1, 2, 3, 7], [0, 1], {"r_id": "relationship_id", "r_name": "relationship_name"}) }}
    