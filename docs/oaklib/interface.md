# OAK Lib Interface

The `omop-graph` library provides an adapter for the [Ontology Access Kit (OAK)](https://incatools.github.io/ontology-access-kit/introduction.html). This allows researchers to interact with an OMOP CDM database using the same standardized Pythonic interface used for OBO, OWL, and other semantic web formats.

---

## Overview

The OAK implementation translates OMOP-specific concepts (Domains, Vocabularies, and Concept IDs) into the standardized **CURIE** (Compact URI) format and implements core OAK interfaces for searching, graph traversal, and text annotation.

### Compliance Status
!!! info "Development Note"
    
    This interface is designed to adhere to OAK’s general design principles to ensure cross-compatibility with the ecosystem. While core features like searching and hierarchy traversal are stable, some advanced OWL-specific methods (like global relationship enumeration) are currently being refined to better align with the RDBMS nature of OMOP.

---

## Core Components

!!! info
    
    The components are found in `omop_graph.oaklib_interface`

### `OMOPAlchemyImplementation` 
The primary adapter class that inherits from multiple OAK interfaces:

* **`BasicOntologyInterface`**: Handles labels, aliases, and metadata mapping.
* **`OboGraphInterface`**: Nodes, `ancestors`, `descendants`, graphs and paths, see [Walking the hierarchy](#walking-the-hierarchy).
* **`RelationGraphInterface`**: Relationships that follow from the stored ones through the `Is a` hierarchy, see [Entailed relationships](#entailed-relationships).
* **`SearchInterface`**: Enables term-based searching across labels and synonyms.
* **`TextAnnotatorInterface`**: Provides a pipeline to ground raw text spans to OMOP Concept IDs using the internal `KnowledgeGraph` resolvers.

### Entailed relationships
A stored relationship `A -R-> B1` entails `A -R-> B2` for every `B2` reached from `B1` by following `Is a` upwards, repeatedly. `R` is any relationship, including `Is a` itself.

| Stored | Entailed |
| --- | --- |
| Lung cancer `Has finding site` Lung | Lung cancer `Has finding site` Thoracic organ |
| Lung `Is a` Thoracic organ | |
| Lung cancer `Is a` Cancer | Lung cancer `Is a` Disease |
| Cancer `Is a` Disease | |

* oaklib has largely subsumed `RelationGraphInterface` into `BasicOntologyInterface`, so the entailment is implemented on `OMOPBasicOntologyInterface` as `relationships(include_entailed=True)`. The `RelationGraphInterface` methods wrap it.
* Only `Is a` is followed upwards, not the other ancestry in `concept_ancestor` such as ATC or SPL classes.
* The relationships of an ancestor are not inherited by its descendants.
* Only concepts within the adapter's `concept_filter` are reported as entailed targets, though the climb passes through the others.
* The incoming direction is the mirror image, and for a concept near the top of a hierarchy it returns very many relationships.

### Walking the hierarchy
`ancestors`, `descendants` and the graphs built on them follow the predicates you give, each walk taking one query for all start concepts. Without predicates they follow `rdfs:subClassOf`, the strict `Is a` hierarchy. oaklib walks every predicate in that case, but over all OMOP relationships a walk reaches most of the vocabulary within a few hops.

```python
adapter.ancestors("OMOP:258369", reflexive=False)
adapter.ancestors("OMOP:258369", predicates=["rdfs:subClassOf", "OMOP_REL:Has%20finding%20site"])
```

* `ancestors` and `descendants` report only concepts within the adapter's `concept_filter`, though the walk passes through the others. `ancestor_graph` and `descendant_graph` keep every concept on the walk so the graph stays connected.
* `nodes`, `edges` and `as_obograph` enumerate the whole graph and are unbounded on a full vocabulary.
* `descendants` of a concept near the top of a hierarchy return very many concepts.

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

### Text Annotation (Grounding)
The interface allows you to process clinical text and ground it to OMOP:

```python
from oaklib.datamodels.text_annotator import TextAnnotationConfiguration

text = "Patient presents with acute myocardial infarction."
config = TextAnnotationConfiguration()

for annotation in adapter.annotate_text(text, configuration=config):
    print(f"Match: {annotation.object_label} ({annotation.object_id})")
```

---
