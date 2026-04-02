# Database Management CLI

The OMOP CDM instantiation tool provides a streamlined way to bootstrap a local OHDSI Common Data Model (CDM) database using Athena vocabulary files and synthetic test data.

---

## `omop-cdm` {: #omop-cdm }

Bootstrap the OMOP CDM and load reference data from Athena into a local database.

!!! danger "Warning"
    This command will wipe the existing database in the target container before loading new data.

### Prerequisites

Before running the command, ensure your environment is configured with a `.env` file or exported variables:

- **`OMOP_DATABASE_URL`**: SQLAlchemy connection string (e.g., `postgresql://user:pass@localhost:5432/omop`).
- **`SOURCE_PATH`**: Local directory path containing the Athena CSV files (e.g., `CONCEPT.csv`, `VOCABULARY.csv`).

### Usage
If installed as a package:
```bash
omop-graph omop-cdm [--add-test-data] --chunk_size=<chunk_size>
```

**Example Usage:**
```bash
# Instantiate with test data and a custom chunk size of 10,000
omop-graph omop-cdm --add-test-data --chunk-size=10_000
```
```bash
# Display the help
omop-graph omop-cdm --help
```

### Command Arguments
| Argument | Type | Default | Description |
| :--- | :--- | :---: | :--- |
| **`--add-test-data`** | `Boolean` | False | Whether to add synthetic test data after loading Athena data.|
| **`--chunk-size`**, **`-c`** | `Integer` | `5000` | Number of rows to process in each chunk. Adjust based on your system's memory capacity to avoid OOM errors. |

---

## `relationship-classification` {: #relationship-classification }

This command ingests pre-defined relationship classifications and mappings into the database. It categorizes standard OMOP relationships into semantic groups (e.g., Hierarchical, Lateral, Mapping) to enable more intelligent graph reasoning.

### Rationale
The standard OMOP `relationship` table provides basic metadata, but lacks unified semantic "kinds" out of the box. This tool maps those relationships to a specific `ClassIDEnum` (like `EQUIVALENT`, `HIERARCHICAL`, or `IDENTITY`) and provides detailed inference descriptions used by the `KnowledgeGraph` facade.

### Prerequisites

The command expects two CSV files to be present in the target directory:

### Prerequisites
Before running the command, ensure your environment is configured with a `.env` file or exported variables:
1. Prepopulated OMOP CDM (e.g. using command [`omop-cdm`](#omop-cdm))
2. **`predicate_classification.csv`**: Defines the semantic classes and subclasses (descriptions, semantics, and inference rules).
3. **`predicate_mapping.csv`**: Maps specific OMOP `relationship_id`s to the classes defined in the classification file.
4. Set following environment variables:
    - **`OMOP_DATABASE_URL`**: SQLAlchemy connection string (e.g., `postgresql://user:pass@localhost:5432/omop`).
    - **`SOURCE_PATH`**: Local directory path containing the Athena CSV files (e.g., `CONCEPT.csv`, `VOCABULARY.csv`). This is required as the new connections/tables are stored there after creation.

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