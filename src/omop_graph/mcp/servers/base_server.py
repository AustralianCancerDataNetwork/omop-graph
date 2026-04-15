"""Abstract base class for omop-graph MCP servers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


class MCPServer(ABC):
    """Base class for omop-graph MCP servers.

    Subclasses must implement `name` and `register_tools` to define
    a concrete MCP server. The `build()` method handles instantiation
    via FastMCP.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Server name (used as FastMCP app name)."""

    @abstractmethod
    def register_tools(self, app: FastMCP) -> None:
        """Register all tools on the FastMCP app instance."""

    def build(self) -> FastMCP:
        """Build and return the FastMCP server instance."""
        try:
            from mcp.server.fastmcp import FastMCP
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "MCP runtime not installed. Install with: pip install mcp"
            ) from exc

        app = FastMCP(self.name)
        self.register_tools(app)
        return app