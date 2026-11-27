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

## Generalising Relationships

Some relationships are carried along the `Is a` hierarchy and some are not. Take these stored facts:

- Small cell lung cancer `Is a` Lung cancer
- Lung cancer `Has finding site` Lung
- Lung `Is a` Thoracic organ

`Has finding site` generalises, so two more facts follow:

- The object moves up: Lung cancer `Has finding site` Thoracic organ.
- The subject moves down: Small cell lung cancer `Has finding site` Lung, and Thoracic organ.

Its OMOP reverse, Lung `Finding site of` Lung cancer, does not generalise. Moving the subject down would give Right lung `Finding site of` Lung cancer, and moving the object up would give Lung `Finding site of` Cancer. Neither follows, because not every lung is the site of a cancer.

A relationship generalises when it reads as "every X has some Y". Every lung cancer has a finding site, but not every lung is the finding site of something.

**Which relationships generalise**

- The `generalises` flag in `predicate_mapping.csv`, stored in `relationship_mapping.generalises`, is the only thing that decides. A relationship generalises when it is true.
- Attribute, composition and association relationships can be true. Examples are `Has finding site`, `Has method` and `May treat`.
- A relationship and its OMOP reverse are never both true.
- Of the Hierarchy relationships, only the upward ones can be true, such as `Is a`. `Subsumes` (down) and the categorical ones such as `Has Topic` are false.
- Identity and Metadata relationships never generalise, and neither do the `Mapping`, `Administrative`, `Contraindication` and `Value mapping` subkinds of Association. A mapping of a concept does not apply to its subtypes, and a contraindication of a drug class is not a contraindication of every drug.
- The loader refuses a file that breaks one of these rules.
- A relationship is only set to true when the data and its meaning agree. Where it is not certain, the flag is left empty, which means undecided.

**How the flag was set**

`scripts/generalises_audit.py` measures each relationship through the strict `Is a` hierarchy, the same walk that entailment follows. For a concept that is an `Is a` child of a parent, where the parent has the relationship to some object, the check asks whether the child also has it, to that object or to a more specific one. The flag is proposed as true when at least 70% of the children that have the relationship carry the parent's object, over at least 30 measured pairs, and the reverse relationship, where it can be measured, does so for at most 15%.

These three thresholds are judgement thresholds chosen with the script. They are not derived from anything. The sample of stored edges follows the storage order of the database, and a measurement that times out is reported as unmeasured and never as a result.

`src/omop_graph/data/generalises_provenance.json` records the vocabulary release and date of the run, all parameters, how many proposals change at nearby thresholds, the flags that were changed to follow the audit, and the nine relationships whose flag was kept or left empty against the proposal. Where the check is mixed or cannot be run, such as for vocabularies whose hierarchy does not follow their attributes, the flag stays empty. The check measures whether the data is consistent under inheritance, not whether the claim holds, so an empty flag does not mean the relationship does not generalise.

## Predicate Classification

To allow reproduction and evaluation of this approach, we provide clear guidelines how we classified edge predicates of the OMOP CDM into the aforementioned groups. The following table provides exact descriptions about each `class` and `subclass`.

??? "Expand to see the grouping classification of predicates"
    
    {{ to_grouped_table('src/omop_graph/data/predicate_classification.csv', [0, 1], [0, 1, 2, 3, 4], [0, 1],) }}

## Predicate Mappings
Following the predicate classification guidelines of the previous section, we classified the following predicates into their respective classification groups.

!!! warning
    
    This classification is currently still under development and most likely may change with increased feedback from clinicians. The respective interface to store these classifications in the OMOP CDM has been prepared and we are in talks to potentially include this classification eventually in the official OMOP CDM.

??? "Expand to see the classification of all edge connections"
    
    {{ to_grouped_table('src/omop_graph/data/predicate_mapping.csv', [0, 1], [0, 1, 2, 3, 7, 8], [0, 1], {"r_id": "relationship_id", "r_name": "relationship_name"}) }}
    