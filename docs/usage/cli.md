# Database Management CLI

The OMOP CDM instantiation tool provides a streamlined way to bootstrap a local OHDSI Common Data Model (CDM) database using Athena vocabulary files and synthetic test data.

---

## `omop-maint load-vocab-source` {: #load-vocab-source }

!!! info "Moved to `omop-maint`"
    Vocabulary loading was previously exposed as `omop-graph omop-cdm`. It is now provided by the [`OMOP_Alchemy`](https://australiancancerdatanetwork.github.io/OMOP_Alchemy/) package under the `omop-maint` CLI.

Load Athena vocabulary CSV files from a configured source path into the OMOP CDM database using the ORM staged CSV loader.


## `relationship-classification` {: #relationship-classification }

This command ingests pre-defined relationship classifications and mappings into the database. It categorizes standard OMOP relationships into semantic groups (e.g., Hierarchical, Lateral, Mapping) to enable more intelligent graph reasoning.

### Rationale
The standard OMOP `relationship` table provides basic metadata, but lacks unified semantic "kinds" out of the box. This tool maps those relationships to a specific `ClassIDEnum` (like `HIERARCHY`, `IDENTITY`, or `ASSOCIATION`) and provides detailed inference descriptions used by the `KnowledgeGraph` facade.

### Prerequisites
Before running the command, ensure your environment is configured with a `.env` file or exported variables:
1. Prepopulated OMOP CDM (e.g. using [`omop-maint load-vocab-source`](#load-vocab-source))
2. **`predicate_classification.csv`**: Defines the semantic classes and subclasses (descriptions, semantics, and inference rules).
3. **`predicate_mapping.csv`**: Maps specific OMOP `relationship_id`s to the classes defined in the classification file.
4. Set following environment variables:
    - **`OMOP_CDM_DB_URL`**: SQLAlchemy connection string (e.g., `postgresql+psycopg://user:pass@localhost:5432/omop`). See [`omop-maint load-vocab-source` options](#load-vocab-source) for connection configuration details.
    - **`OMOP_VOCABULARY_DIR`**: Local directory path where the generated classification tables will be written as CSV files.

### Usage

```bash
omop-graph relationship-classification --pred-class-dir <PATH_TO_CSV_DIR>
```

### Command Options

| Option | Short | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`--pred-class-dir`** | | `String` | `./docs` | Path to the directory containing the classification CSVs. |
| **`--verbose`** | `-v` | `Count` | `0` | Increase logging verbosity (use `-v` or `-vv`). |
---