"""MCP server exposing project tools via Model Context Protocol.

Registers all BaseTool subclasses as MCP tools so they can be consumed
by any MCP-compatible client (Claude Desktop, other agents, etc.).

Usage:
    # Start as stdio server (for langchain-mcp-adapters / subprocess)
    python src/tools/mcp_server.py

    # Start as HTTP server (for remote access)
    python src/tools/mcp_server.py --transport streamable-http --port 8000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from mcp.server.fastmcp import FastMCP

from src.tools.registry import get_all_tool_instances

mcp = FastMCP("sec-filing-analyst")


def _register_all_tools() -> None:
    """Register all BaseTool instances as MCP tools on the server."""
    for tool in get_all_tool_instances():
        # Create a closure that captures the tool instance
        _make_and_register(tool)


def _make_and_register(tool) -> None:
    """Create an MCP handler for a single tool and register it."""
    schema = tool.input_schema

    # Build a handler function with proper signature for FastMCP
    async def handler(**kwargs):
        return tool.execute(**kwargs)

    # Set metadata that FastMCP reads
    handler.__name__ = tool.name
    handler.__doc__ = tool.description

    # Copy type annotations from the Pydantic schema so FastMCP
    # can generate the correct JSON Schema for the tool
    handler.__annotations__ = {
        field_name: field_info.annotation
        for field_name, field_info in schema.model_fields.items()
    }
    handler.__annotations__["return"] = str

    mcp.tool()(handler)


# Register tools at import time
_register_all_tools()


def main():
    parser = argparse.ArgumentParser(description="MCP server for financial analysis tools")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="Transport protocol (default: stdio)",
    )
    parser.add_argument("--port", type=int, default=8000, help="Port for HTTP transport")
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="streamable-http", host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
