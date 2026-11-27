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
3. Nothing further — the classification data ships with the package:
    - **`predicate_classification.csv`**: Defines the semantic classes and subclasses (descriptions, semantics, and inference rules).
    - **`predicate_mapping.csv`**: Maps specific OMOP `relationship_id`s to the classes defined in the classification file.

    Both are bundled at `omop_graph/data/` and used by default. Pass
    `--pred-class-dir` only to override them with your own copies.

### Usage

```bash
# Uses the bundled classification data
omop-graph relationship-classification

# Override with your own CSVs
omop-graph relationship-classification --pred-class-dir <PATH_TO_CSV_DIR>
```

### Command Options

| Option | Short | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`--pred-class-dir`** | | `String` | `None` (bundled CSVs) | Path to a directory of classification CSVs, overriding the bundled defaults. |
| **`--strict`** | | `Flag` | `False` | Fail on the two findings that otherwise only warn (see Validation). |
| **`--verbose`**{: title="Global option, not specific to this subcommand — see the note above." } | `-v` | `Count` | `0` | Increase logging verbosity (use `-v` or `-vv`). Global option; must precede the subcommand name (see note above). |

### Validation

Before any existing classification table is dropped, the load fails with the offending rows if:

- a row of `predicate_mapping.csv` has a `class` and `subclass` that `predicate_classification.csv` does not define, or a class that is not one of the five kinds above
- a `class` and `subclass` is defined more than once in `predicate_classification.csv`
- a mapping row's `r_name` differs from that relationship's CDM `relationship_name`, or a `relationship_id` is mapped more than once
- a `Hierarchy` row has no `up` or `down` `direction`, another row has one, or a `Hierarchy` relationship and its CDM reverse are not both `Hierarchy` with opposite directions

Two findings are logged as warnings with their ids and counts, and fail the load under `--strict`, which the tests of omop-graph use:

- a CDM relationship has no classification, so the `KnowledgeGraph` skips the edges of that relationship
- a classified `relationship_id` is not in the CDM, which happens with a subset or an older vocabulary. These rows are not loaded.

The packaged classification was last checked in strict mode against Athena release v5.0 29-AUG-26. A later release can add relationships it does not classify yet, which the load reports as a warning.

A failed check prints its message and exits with code 1.
