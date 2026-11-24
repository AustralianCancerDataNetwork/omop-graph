# Command Line

The `omop-graph` command prepares a CDM for the graph. It has two commands: `relationship-classification`, which every database needs once, and `populate-with-test-data`, which fills a test database with synthetic data.

!!! note "Verbosity flag placement"
    The `--verbose` / `-v` flag is a **global option** and must appear **before** the
    subcommand name, not after it:

    ```
    omop-graph -v relationship-classification ...   # ✓ correct
    omop-graph relationship-classification -v ...   # ✗ flag is ignored
    ```

    Use `-v` for INFO level and `-vv` for DEBUG level.

---

!!! info "Vocabulary loading moved to `omop-maint`"
    Athena vocabularies are loaded with the [`omop-maint` CLI](https://australiancancerdatanetwork.github.io/OMOP_Alchemy/getting-started/maintenance/) of [`OMOP_Alchemy`](https://github.com/AustralianCancerDataNetwork/OMOP_Alchemy), which replaced `omop-graph omop-cdm`.

## `populate-with-test-data`

Adds synthetic patient data to a CDM whose vocabularies are loaded with [`omop-maint load-vocab-source`](https://australiancancerdatanetwork.github.io/OMOP_Alchemy/getting-started/maintenance/). It refuses to run unless both the primary and the vocabulary connection of the configured database are marked `test_only`.

## `relationship-classification` {: #relationship-classification }

This command loads the classification of the OMOP relationships into the database. It sorts each relationship into a kind (`Hierarchy`, `Identity`, `Composition`, `Association` or `Attribute`) and a subkind, as described in [Classification of Edges](../graph/edges.md).

### Rationale
The standard OMOP `relationship` table provides basic metadata, but lacks unified semantic "kinds" out of the box. The `KnowledgeGraph` reads this classification when it starts, and skips the edges of any relationship without one.

!!! warning "Run it again after upgrading"
    The command drops and recreates both classification tables. A database classified by an earlier release has no `direction` column on `relationship_class`, and the `KnowledgeGraph` refuses to start on it with an error naming this command.

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

- a row of `predicate_mapping.csv` has a `class` and `subclass` that `predicate_classification.csv` does not define, or a class that is not a `PredicateKind`
- a `class` and `subclass` is defined more than once in `predicate_classification.csv`
- a mapping row's `r_name` differs from that relationship's CDM `relationship_name`, or a `relationship_id` is mapped more than once
- a `Hierarchy` row has no `up` or `down` `direction`, another row has one, or a `Hierarchy` relationship and its CDM reverse are not both `Hierarchy` with opposite directions
- a `generalises` cell is not empty, `true` or `false`, a relationship that cannot generalise is `true` (see [Generalising Relationships](../graph/edges.md#generalising-relationships)), or a relationship and its CDM reverse are both `true`

Two findings are logged as warnings with their ids and counts, and fail the load under `--strict`, which the tests of omop-graph use:

- a CDM relationship has no classification, so the `KnowledgeGraph` skips the edges of that relationship
- a classified `relationship_id` is not in the CDM, which happens with a subset or an older vocabulary. These rows are not loaded.

The packaged classification was last checked in strict mode against Athena release v5.0 29-AUG-26. A later release can add relationships it does not classify yet, which the load reports as a warning.

A failed check prints its message and exits with code 1.

### After upgrading

A database loaded by an earlier release lacks the `direction` and `generalises` columns of `relationship_mapping`. The `KnowledgeGraph` then raises a `StaleClassificationError` that names the missing columns and this command. Run the command again to load the current schema. A read-only DuckDB file has to be rebuilt.
