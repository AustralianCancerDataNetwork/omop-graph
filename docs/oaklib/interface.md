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
* **`SearchInterface`**: Enables term-based searching across labels and synonyms.
* **`TextAnnotatorInterface`**: Provides a pipeline to ground raw text spans to OMOP Concept IDs using the internal `KnowledgeGraph` resolvers.

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

# Traverse parents (returns CURIEs)
for parent in adapter.parents("OMOP:313217"):
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
