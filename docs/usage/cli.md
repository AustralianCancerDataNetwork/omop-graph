# Database Management CLI

The OMOP CDM instantiation tool provides a streamlined way to bootstrap a local OHDSI Common Data Model (CDM) database using Athena vocabulary files and synthetic test data.

!!! note "Verbosity flag placement"
    The `--verbose` / `-v` flag is a **global option** and must appear **before** the
    subcommand name, not after it:

    ```
    omop-graph -v relationship-classification ...   # ✓ correct
    omop-graph relationship-classification -v ...   # ✗ flag is ignored
    ```

    Use `-v` for INFO level and `-vv` for DEBUG level.

---

## `omop-cdm`

!!! info "Moved to `omop-maint`"
    Vocabulary loading was previously exposed as `omop-graph omop-cdm`. It is now provided by the [`OMOP_Alchemy`](https://github.com/AustralianCancerDataNetwork/OMOP_Alchemy) package under the [`omop-maint` CLI](https://australiancancerdatanetwork.github.io/OMOP_Alchemy/getting-started/maintenance/).

Load Athena vocabulary CSV files from a configured source path into the OMOP CDM database using the ORM staged CSV loader.

## `populate_with_test_data`

This command adds synthetic patient data into the OMOP CDM after populating the vocabularies using [`omop-maint load-vocab-source`](https://australiancancerdatanetwork.github.io/OMOP_Alchemy/getting-started/maintenance/)

## `relationship-classification` {: #relationship-classification }

This command ingests pre-defined relationship classifications and mappings into the database. It categorizes standard OMOP relationships into semantic groups (e.g., Hierarchical, Lateral, Mapping) to enable more intelligent graph reasoning.

### Rationale
The standard OMOP `relationship` table provides basic metadata, but lacks unified semantic "kinds" out of the box. This tool maps those relationships to a specific `PredicateKind` (like `HIERARCHY`, `IDENTITY`, or `ASSOCIATION`) and provides detailed inference descriptions used by the `KnowledgeGraph` facade.

### Prerequisites
1. Prepopulated OMOP CDM (e.g. using [`omop-maint load-vocab-source`](https://australiancancerdatanetwork.github.io/OMOP_Alchemy/getting-started/maintenance/))
2. `omop-graph` configured via oa-configurator (`omop-config configure omop_graph`, see [Getting Started: Configuration](../getting-started/configuration.md)); the CDM connection is resolved from `OmopGraphConfig.cdm_db`, no environment variables involved.
3. A directory containing:
    - **`predicate_classification.csv`**: Defines the semantic classes and subclasses (descriptions, semantics, and inference rules).
    - **`predicate_mapping.csv`**: Maps specific OMOP `relationship_id`s to the classes defined in the classification file.
    
    Pass this directory via `--pred-class-dir` below.

### Usage

```bash
omop-graph relationship-classification --pred-class-dir <PATH_TO_CSV_DIR>
```

### Command Options

| Option | Short | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`--pred-class-dir`** | | `String` | **Required** | Path to the directory containing the classification CSVs. |
| **`--verbose`** | `-v` | `Count` | `0` | Increase logging verbosity (use `-v` or `-vv`). |
