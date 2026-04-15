"""Typer CLI for starting omop-graph MCP servers."""

from __future__ import annotations

import os
from typing import Annotated, Any, Optional, Type

import typer

from .servers import KGServer, MCPServer

app = typer.Typer(help="Run omop-graph MCP servers")

SERVER_CLASSES: dict[str, Type[MCPServer]] = {
    "kg": KGServer,
}


def build_server(server_name: str = "kg") -> Any:
    """Instantiate an MCP server by name and build it.

    Additional servers (for example emb or maint) can be registered in
    SERVER_CLASSES without changing caller code.
    """
    try:
        server_cls = SERVER_CLASSES[server_name]
    except KeyError as exc:
        valid = ", ".join(sorted(SERVER_CLASSES))
        raise typer.BadParameter(
            f"Unknown server '{server_name}'. Available: {valid}"
        ) from exc
    return server_cls().build()


@app.command()
def run(
    server: Annotated[
        Optional[str],
        typer.Option(
            "--server",
            "-s",
            help="MCP server to run (default: kg or OMOP_GRAPH_MCP_SERVER env var)",
        ),
    ] = None,
) -> None:
    """Run an omop-graph MCP server.

    If --server is not provided, reads from OMOP_GRAPH_MCP_SERVER env var or defaults to 'kg'.
    """
    if server is None:
        server = os.getenv("OMOP_GRAPH_MCP_SERVER", "kg")

    mcp_server = build_server(server_name=server)
    mcp_server.run()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
