"""Tool registry with class-based auto-registration.

Tools auto-register via BaseTool.__init_subclass__() when their modules
are imported. The registry provides LangChain tool objects for agents
and raw tool instances for the MCP server.

Usage:
    from src.tools.registry import get_tools, list_tools, get_all_tool_instances

    # For agents (returns LangChain StructuredTool objects)
    tools = get_tools(["rag_retrieve", "get_market_data"])

    # For MCP server (returns BaseTool instances)
    instances = get_all_tool_instances()
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool as LCBaseTool
    from src.tools.base_tool import BaseTool

# Maps tool name → tool CLASS (not instance)
_class_registry: dict[str, type[BaseTool]] = {}

# Cached singleton instances (created on first access)
_instance_cache: dict[str, BaseTool] = {}

# Tool modules to import for auto-registration
_TOOL_MODULES = [
    "src.tools.rag_tool",
    "src.tools.market_data_tool",
    "src.tools.fred_tool",
    "src.tools.sentiment_tool",
    "src.tools.date_tool",
    "src.tools.indicators_tool",
]


def _register_class(cls: type[BaseTool]) -> None:
    """Called automatically by BaseTool.__init_subclass__."""
    _class_registry[cls.name] = cls


def _ensure_imported() -> None:
    """Import all tool modules to trigger __init_subclass__ registration."""
    for module_name in _TOOL_MODULES:
        try:
            importlib.import_module(module_name)
        except ImportError:
            pass


def _get_instance(name: str) -> BaseTool:
    """Get or create a singleton instance of a tool by name."""
    if name not in _instance_cache:
        if name not in _class_registry:
            _ensure_imported()
        if name not in _class_registry:
            available = ", ".join(sorted(_class_registry.keys()))
            raise KeyError(f"Tool '{name}' not found. Available: {available}")
        _instance_cache[name] = _class_registry[name]()
    return _instance_cache[name]


def get_tool_by_name(name: str) -> LCBaseTool:
    """Get a single LangChain tool by name."""
    return _get_instance(name).to_langchain_tool()


def get_tools(names: list[str]) -> list[LCBaseTool]:
    """Get multiple LangChain tools by name (used by BaseAgent.get_tools())."""
    _ensure_imported()
    return [get_tool_by_name(n) for n in names]


def get_all_tool_instances() -> list[BaseTool]:
    """Get all registered tool instances (used by MCP server)."""
    _ensure_imported()
    return [_get_instance(n) for n in _class_registry]


def list_tools() -> list[str]:
    """List all registered tool names."""
    _ensure_imported()
    return sorted(_class_registry.keys())
