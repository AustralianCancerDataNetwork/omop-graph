# Database Management CLI

The OMOP CDM instantiation tool provides a streamlined way to bootstrap a local OHDSI Common Data Model (CDM) database using Athena vocabulary files and synthetic test data.

---

## `omop-cdm` {: #omop-cdm }

Bootstrap the OMOP CDM and load reference data from Athena into a local database.

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

## Optional: `add-embeddings` 

!!! warning
    
    This method is only exposed for convenience reasons to populate the concept embeddings if the optional dependency `[emb]` was used during the [installation process](installation.md).

The method is directly imported from [`omop-emb`](https://australiancancerdatanetwork.github.io/omop-emb/). See the [documentation](https://australiancancerdatanetwork.github.io/omop-emb/usage/cli/) for more information.

### Prerequisites
- **Database**: Postgres implementation of OMOP CDM. See [`omop-cdm`](#omop-cdm) for more details.

### Usage
```bash
omop-emb add-embeddings --api-base <URL> --api-key <KEY> [OPTIONS]
```
where `[OPTIONS]` are optional arguments that can be specified as described below.

### Command Options

| Option | Short | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`--api-base`** | | `String` | **Required** | Base URL for the embedding API service. |
| **`--api-key`** | | `String` | **Required** | API key for the embedding API provider. |
| **`--batch-size`** | `-b` | `Integer` | `100` | Number of concepts to process in each chunk. |
| **`--model`** | `-m` | `String` | `text-embedding-3-small` | Name of the embedding model to use for generating vectors. |
| **`--num-embeddings`** | `-n` | `Integer` | `None` | Limit the number of concepts processed (useful for testing). |
---

