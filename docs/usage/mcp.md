# MCP Integration

omop-graph includes an MCP server layer that exposes bounded knowledge-graph tools for agent workflows.

The integration follows a split architecture:

- graph logic remains in core modules under `src/omop_graph/graph`
- MCP transport and tool registration live under `src/omop_graph/mcp`

The example file at `mcp/tool_config.example.yaml` is a future policy template only; the current MCP-only commit does not load it.

## Available MCP Tools

The current server exposes the following tools:

- `concept_view`: retrieve concept metadata by OMOP concept ID
- `concept_lookup`: resolve text labels to candidate concepts
- `shortest_paths`: compute bounded shortest paths between concept IDs
- `explore_connections`: perform bounded neighborhood exploration for distant-link discovery

Each tool has explicit limits to keep runtime predictable.

## Exploration Model

Exploration uses path-native structures from the graph layer:

- `ExplorationStep` extends `PathStep` with a `depth` field
- `ExplorationResult` returns visited nodes, traversal steps, and truncation status

This keeps representation consistent between deterministic pathfinding and iterative exploration.

## Run The Server

Install MCP runtime support:

```bash
uv pip install .[mcp]
```

Set database connection:

```bash
export OMOP_DATABASE_URL="postgresql://..."
```

Start the MCP launcher (currently `kg` is the available server):

```bash
omop-graph-mcp --server kg
```

You can also run the KG server directly:

```bash
omop-graph-mcp-kg
```

The launcher exists so additional servers can be added without changing client startup conventions.
