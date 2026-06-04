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
first start and skipped on subsequent starts (`--skip-if-configured` makes this
idempotent).

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

## Further reading

- [oa-configurator integration guide](https://AustralianCancerDataNetwork.github.io/oa-configurator/integration/) — full config reference, profiles, multi-package setups
