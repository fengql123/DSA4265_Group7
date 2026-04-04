"""Tool system — BaseTool, registry, and MCP server.

Define new tools by subclassing BaseTool:

    from src.tools.base_tool import BaseTool
    from pydantic import BaseModel, Field

    class MyInput(BaseModel):
        query: str = Field(description="What to search for")

    class MyTool(BaseTool):
        name = "my_tool"
        description = "Does something useful"
        input_schema = MyInput

        def execute(self, query: str) -> str:
            return f"Result: {query}"

Tools auto-register via __init_subclass__ — no manual registration needed.
"""

from src.tools.base_tool import BaseTool, ToolResult

__all__ = ["BaseTool", "ToolResult"]
