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

omop-graph does not own any database resources. It reads from the `cdm_db` resource
configured by omop-alchemy and stores any package-specific settings (traversal depth,
path limits) under `[tools.omop_graph]` in `config.toml`.

## Verify

```bash
omop-config verify
```

## Docker Compose

The included `docker-compose.yaml` spins up a PostgreSQL CDM database and a
`python-graph` container. Default credentials work out of the box:

```bash
docker compose up
```

The `python-graph` container runs `omop-config configure` for both `omop_alchemy` and
`omop_graph` at startup. Your `~/.config/omop/config.toml` on the host is written on
safe to re-run on subsequent starts: connection flags always apply, and any values already stored in `config.toml` are preserved for fields not explicitly provided.

### Overriding default values

The compose file uses built-in defaults for all database credentials. To use different
values, create a `.env` file in this directory with any of the following variables:

| Variable | Default | Description |
|---|---|---|
| `OMOP_CDM_DB_USER` | `omop` | CDM database username |
| `OMOP_CDM_DB_PASSWORD` | `omop` | CDM database password |
| `OMOP_CDM_DB_NAME` | `omop_cdm` | CDM database name |

Copy the example and edit as needed:

```bash
cp .env.example .env
# edit .env
docker compose up
```

The `.env` file is only read by Docker Compose for variable substitution — it is not
loaded by omop-graph at runtime.

## Multiple instances

omop-graph reads from the `cdm_db` resource owned by omop-alchemy. To point
it at a second CDM database (e.g. for production), configure omop-alchemy with
a second resource:

```bash
omop-config configure omop_alchemy --resource-name cdm_db_prod
```

Configure automatically prompts you to choose the default at the end of the same
run — no second invocation needed.

See the [oa-configurator integration guide](https://AustralianCancerDataNetwork.github.io/oa-configurator/integration/#multiple-environments) for the full multi-environment guide.

## Further reading

- [oa-configurator integration guide](https://AustralianCancerDataNetwork.github.io/oa-configurator/integration/) — full config reference, profiles, multi-package setups
