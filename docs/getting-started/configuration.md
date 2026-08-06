# Configuration

omop-graph reads all database connection and schema settings from
[oa-configurator](https://github.com/AustralianCancerDataNetwork/oa-configurator).
No environment variables are needed for the Python package itself.

## Quick start

omop-graph requires the CDM database configured by omop-alchemy. If you have not
already done so, configure omop-alchemy first:

```bash
omop-config init          # creates ~/.config/omop/config.toml if absent
omop-config configure omop_alchemy
omop-config configure omop_graph
```

## What gets configured

`OmopGraphConfig` (`[tools.omop_graph]`) has:

| Field | References | Description |
|---|---|---|
| `cdm_db` | a `[databases.*]` entry, `kind = "cdm"` | Shared by naming convention with omop-alchemy's own `cdm_db` field |
| `embedding_model_name` | a `[models.*]` entry, optional | Not read by omop-graph itself; a caller resolving embedding-based grounding on omop-graph's behalf (e.g. omop-spires) can use it the same way it uses `cdm_db` |
| `vector_store_name` | a `[vector_stores.*]` entry, optional | Same reasoning as `embedding_model_name` |
| `max_depth` | plain int, default `6` | Maximum graph traversal depth for pathfinding and grounding |
| `max_paths` | plain int, default `20` | Maximum number of shortest paths returned per query |

omop-graph itself never resolves `embedding_model_name`/`vector_store_name`, or reads its own `max_depth`/`max_paths` internally: embedding support is entirely caller-supplied via `KnowledgeGraphEmbeddingConfiguration`, and traversal functions take `max_depth`/`max_paths` as plain parameters, resolved by the caller at its own CLI/entry-point boundary. See [KnowledgeGraph — Embedding Configuration](../graph/kg.md#embedding-configuration).

## Verify

```bash
omop-config verify
```

## Multiple instances

omop-graph reads from the `cdm_db` database owned by omop-alchemy. To point
it at a second CDM database (e.g. for production), create it under its own name
and point the field's own flag at it:

```bash
omop-config databases add cdm_db_prod --kind cdm --connection cdm_prod
omop-config configure omop_alchemy --cdm-db cdm_db_prod
omop-config configure omop_graph --cdm-db cdm_db_prod
```

There is no "default" toggle to flip afterward; each deployment's `configure` call names the entry it wants directly.

See the [oa-configurator integration guide](https://AustralianCancerDataNetwork.github.io/oa-configurator/integration/#multiple-environments) for the full multi-environment guide.

## Further reading

- [oa-configurator integration guide](https://AustralianCancerDataNetwork.github.io/oa-configurator/integration/): full config reference, multi-package setups
