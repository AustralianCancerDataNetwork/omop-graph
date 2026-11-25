# OAK Lib Interface

The `omop-graph` library provides an adapter for the [Ontology Access Kit (OAK)](https://incatools.github.io/ontology-access-kit/introduction.html). This allows researchers to interact with an OMOP CDM database using the same standardized Pythonic interface used for OBO, OWL, and other semantic web formats.

---

## Overview

The OAK implementation translates OMOP-specific concepts (Domains, Vocabularies, and Concept IDs) into the standardized **CURIE** (Compact URI) format and implements the OAK interfaces for labels and metadata, relationships and entailment, graph traversal, mappings, searching, and text annotation. The adapter only reads: vocabularies are loaded with `omop-maint`, and embeddings with `omop-emb`.

### Compliance Status
The adapter implements the oaklib interfaces below and raises `OMOPUnsupportedError`, with the reason, for the methods that have no OMOP meaning. These tables are generated from the code, and a test checks that every method that raises `NotImplementedError` in an implemented interface is either implemented or listed here.

{{ oaklib_compliance() }}

---

## Core Components

!!! info
    
    The components are found in `omop_graph.oaklib_interface`

### `OMOPAlchemyImplementation` 
The primary adapter class that inherits from multiple OAK interfaces:

* **`BasicOntologyInterface`**: Labels (also in other languages), aliases, ontologies, deprecation and replacements, relationships, and the roots, leaves and singletons of the hierarchy.
* **`MappingProviderInterface`**: SSSOM mappings from the identity and mapping relationships, see [Mappings](#mappings).
* **`OboGraphInterface`**: Nodes, `ancestors`, `descendants`, graphs and paths, see [Walking the hierarchy](#walking-the-hierarchy).
* **`RelationGraphInterface`**: Relationships that follow from the stored ones through the `Is a` hierarchy, see [Entailed relationships](#entailed-relationships).
* **`SearchInterface`**: Term-based searching across labels, synonyms and identifiers, see [Searching and Traversal](#searching-and-traversal).
* **`TextAnnotatorInterface`**: Grounds raw text to OMOP Concept IDs with the [grounding pipeline](../reasoning/grounding.md), see [Text Annotation](#text-annotation-grounding).

### Identifiers
| What | CURIE |
| --- | --- |
| A concept | `OMOP:<concept_id>`, for example `OMOP:313217`. A `vocabulary_id:concept_code`, such as `SNOMED:49436004`, is accepted as input. |
| `Is a` | `rdfs:subClassOf` |
| Any other relationship | `OMOP_REL:<relationship_id>`, percent-encoded, for example `OMOP_REL:Has%20finding%20site` |
| An ontology | `OMOP:<vocabulary_concept_id>`, one for each vocabulary |
| A category | `OMOP:<domain_concept_id>`, the domain of the concept |

* A relationship is shown from the child to the parent: for hierarchy relationships only the upward direction (`Is a`) is listed, and the children of a concept are the subjects of the relationships into it. The downward direction (`Subsumes`) can be asked for by name.
* A vocabulary id that is not a valid CURIE prefix, such as `Nebraska Lexicon`, is written `Nebraska_Lexicon` in CURIEs, and either spelling is accepted as input, including in `id_prefixes`.
* A concept code is percent-encoded in a CURIE, so the code `[iU]/L` of the `UCUM` vocabulary is `UCUM:%5BiU%5D%2FL`.
* OMOP does not keep a code unique within its vocabulary. A `vocabulary_id:concept_code` names the valid concept when only one has the code. When several do, it names none and a warning lists their `OMOP:<concept_id>` CURIEs, which are always unambiguous.
* A CURIE that names no concept, and a predicate CURIE that names no OMOP relationship such as `BFO:0000050`, are left out of the result instead of raising.
* `prefix_map` holds the oaklib defaults, `OMOP` and `OMOP_REL`, and each vocabulary that is a canonical prefix of the [Bioregistry](https://bioregistry.io), matched by name. A vocabulary the Bioregistry does not have, such as `Nebraska_Lexicon`, has no URI.
* `entities`, `roots`, `leafs` and `singletons` list active standard and classification concepts. Pass `concept_filter=ConceptFilter(...)` to the constructor to change that.
* `entities(filter_obsoletes=False)` lists every concept in the filter's vocabularies and domains, whatever its standardness or validity, so it includes everything `obsoletes` yields. `obsoletes` stays within those vocabularies and domains.

### Entailed relationships
Entailed relationships are the ones that follow from the stored ones through the `Is a` hierarchy. A stored relationship is carried in two directions:

* The object moves up. Lung cancer `Has finding site` Lung also gives Lung cancer `Has finding site` Thoracic organ, since Lung `Is a` Thoracic organ.
* The subject moves down. Small cell lung cancer `Is a` Lung cancer, so it also has the finding site Lung, and Thoracic organ.

| Stored | Entailed |
| --- | --- |
| Lung cancer `Has finding site` Lung | Lung cancer `Has finding site` Thoracic organ |
| Lung `Is a` Thoracic organ | Small cell lung cancer `Has finding site` Lung |
| Small cell lung cancer `Is a` Lung cancer | Small cell lung cancer `Has finding site` Thoracic organ |
| Lung cancer `Is a` Cancer | Lung cancer `Is a` Disease |
| Cancer `Is a` Disease | |

Only some relationships are carried along, see [Generalising Relationships](../graph/edges.md#generalising-relationships):

* Carried along: `Is a`, and relationships such as `Has finding site`, `Has method` and `May treat`.
* Reported as stored only: the reverse of those (`Finding site of`), mappings and lineage (`Maps to`, `Concept replaced by`), `Subsumes`, and any relationship the data could not decide.

How the entailment is implemented:

* oaklib has largely subsumed `RelationGraphInterface` into `BasicOntologyInterface`, so the entailment is implemented on `OMOPBasicOntologyInterface` as `relationships(include_entailed=True)`. The `RelationGraphInterface` methods wrap it.
* oaklib applies this to every relationship. The adapter applies it only to those that generalise. The other relation graph rules have nothing to act on here, since OMOP has no relationship hierarchy, declares no transitive relationships and stores both directions of each.
* Only `Is a` is followed, not the other ancestry in `concept_ancestor` such as ATC or SPL classes.
* Only concepts within the adapter's `concept_filter` are reported as entailed objects, and for incoming relationships as entailed subjects, though the walk passes through the others.
* For a concept near the top of a hierarchy the incoming direction returns very many relationships, since each subtype below it is a subject.

### Walking the hierarchy
`ancestors`, `descendants` and the graphs built on them follow the predicates you give, each walk taking one query for all start concepts. Without predicates they follow `rdfs:subClassOf`, the strict `Is a` hierarchy. oaklib walks every predicate in that case, but over all OMOP relationships a walk reaches most of the vocabulary within a few hops.

```python
adapter.ancestors("OMOP:258369", reflexive=False)
adapter.ancestors("OMOP:258369", predicates=["rdfs:subClassOf", "OMOP_REL:Has%20finding%20site"])
```

* `ancestors` and `descendants` report only concepts within the adapter's `concept_filter`, though the walk passes through the others. `ancestor_graph` and `descendant_graph` keep every concept on the walk so the graph stays connected.
* A non-standard concept has no ancestry, so its `ancestors` are empty. With `OMOPAlchemyImplementation(standardise=True)` it stands in for the standard concepts it maps to: its walks include them and their ancestors, and its entailed relationships are those of its standard concepts that generalise, reported for the concept itself. Mappings and lineage of the standard concepts are not lent to it. This is off by default because the result holds concepts that no stored edge connects to the concept.
* `nodes`, `edges` and `as_obograph` enumerate the whole graph and are unbounded on a full vocabulary.
* `descendants` of a concept near the top of a hierarchy return very many concepts.

### Mappings
`sssom_mappings` serves the OMOP mappings as SSSOM, with the looked up concept as the subject.

| OMOP relationship | SSSOM predicate |
| --- | --- |
| Identity, exact equivalence (`Maps to`, `Mapped from`) | `skos:exactMatch` |
| Identity, near equivalence (`Brand name of`) | `skos:closeMatch` |
| Association, mapping (`ICDO - SNOMED`) | `skos:relatedMatch` |
| Concept vocabulary and code | `oio:hasDbXref` |

* Version lineage such as `Concept replaced by` is not a mapping, and neither is a concept mapped to itself.
* OMOP stores each pair in both directions, so a standard concept has a mapping to every source concept that maps to it. Without concepts, `sssom_mappings` lists both directions of each pair.
* The justification is `semapv:UnspecifiedMatching`, since OMOP does not record how a mapping was made. The date is when the relationship became valid, and the comment names the OMOP relationship.
* A source or code can use a vocabulary id that is not a valid CURIE prefix, such as `Nebraska Lexicon`. It is written `Nebraska_Lexicon`, and either spelling is accepted as input.
* The cross-reference is a mapping only, so it is not in `entity_alias_map`. It can be looked up from the concept or from `vocabulary_id:concept_code`.
* Mappings are not limited to the adapter's `concept_filter`.
* `get_transitive_mappings_by_curie` follows exact equivalence from a non-standard concept to the standard concepts it maps to, and a standard concept has none. OMOP maps many concepts to one and stores each mapping in both directions, so following them both ways would join every concept that shares a target, and the result would be most of a vocabulary. `get_mapping_clusters` is unsupported since it would load about 20 million mappings.

### Resource Management
The adapter resolves its database through oa-configurator; there is no URL-based construction.

* **`OMOPOntologyResource`**: The oaklib resource for the `omop` scheme. Its `slug` names a `[databases.*]` entry; no slug resolves `OmopGraphConfig.cdm_db`.
* **`resolved=`**: Pass an already-resolved `ResolvedCDMDatabase` to build the engines from it directly. A separate `vocab_connection` produces a second, vocabulary-only engine.
* **`kg=`**: Pass an existing `KnowledgeGraph` to use its engines as-is.

---

## Usage Examples

### Initializing the Adapter
Select the adapter through oaklib with an oa-configurator database name:

```python
from oaklib import get_adapter

# Named [databases.*] entry
adapter = get_adapter("omop:cdm_db")

# Or OmopGraphConfig.cdm_db
adapter = get_adapter("omop:")

# Use standard OAK methods
label = adapter.label("OMOP:44819488")
print(f"Label: {label}")
```

Or construct it from an already-resolved database:

```python
from omop_graph.db.session import resolve_cdm_database
from omop_graph.oaklib_interface.omop_implementation import OMOPAlchemyImplementation

adapter = OMOPAlchemyImplementation(resolved=resolve_cdm_database("cdm_db"))
```

The constructor also takes an existing `KnowledgeGraph` as `kg`, a `concept_filter` for the concepts the adapter lists, and `standardise` (see [Walking the hierarchy](#walking-the-hierarchy)).

### Searching and Traversal
Since the adapter implements the `SearchInterface`, you can perform standardized searches:

```python
# Search for concepts matching a term
for curie in adapter.basic_search("Atrial Fibrillation"):
    print(f"Found: {curie}")

# Direct parents (returns CURIEs)
for parent in adapter.hierarchical_parents("OMOP:313217"):
    print(f"Parent CURIE: {parent}")
```

`basic_search` returns the best matches first and honours `limit`:

* Labels are always searched, aliases with the `ALIAS` property, and identifiers (an `OMOP` CURIE or `vocabulary:code`) with `IDENTIFIER`.
* The order is identifier, exact label, labels starting with the term (`STARTS_WITH` syntax or `is_partial`), then labels containing it (`is_partial`), then full-text matches (`is_fuzzy`, only where the database has the index). Within a kind, standard and active concepts come first.
* Matching is case-insensitive, and `%`, `_` and `\` in the term match themselves. Deprecated concepts are left out unless `include_obsoletes_in_results` is set.
* The adapter's `concept_filter` narrows the vocabularies and domains searched, but not standardness.
* `categories`, `cursor`, `is_complete` and the `REPLACEMENT_IDENTIFIER` and `COMMENT` properties have no effect. An info message names the ones that were set.
* Regular expression, SQL and Lucene syntax raise `ValueError`.

### Text Annotation (Grounding)
The whole text is grounded as one term, which requires to set `matches_whole_text=True`. Spans within a longer text are not annotated, and a configuration without `matches_whole_text` (including no configuration) raises `OMOPUnsupportedError`.

```python
from oaklib.datamodels.text_annotator import TextAnnotationConfiguration

config = TextAnnotationConfiguration(matches_whole_text=True, limit=5)

for annotation in adapter.annotate_text("atrial fibrillation", configuration=config):
    print(f"Match: {annotation.object_label} ({annotation.object_id})")
```

* `annotate_text` yields nothing when the text does not ground, as a sentence such as `"patient has cancer"` usually does not.
* `limit` caps the number of annotations, and `token_exclusion_list` drops tokens from the text before grounding.
* `sources`, `categories`, `model` and `include_aliases` have no effect. An info message names the ones that were set.

---
