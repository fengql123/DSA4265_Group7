"""Base class for all project tools.

All tools inherit from BaseTool. The class provides:
- Auto-registration via __init_subclass__
- Conversion to LangChain StructuredTool for bind_tools()
- Artifact support: tools can return text + images/files via ToolResult
- A clean execute() interface that subclasses implement

Usage:
    from src.tools.base_tool import BaseTool, ToolResult
    from src.artifacts import Artifact, ArtifactType

    class MyTool(BaseTool):
        name = "my_tool"
        description = "Does something useful"
        input_schema = MyInput

        def execute(self, query: str) -> str:
            return "Result"

        # Or with artifacts:
        def execute(self, query: str) -> ToolResult:
            return ToolResult(
                content="Chart saved.",
                artifacts=[Artifact(ArtifactType.IMAGE, "out.png", "image/png", "Chart")],
            )
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Type

from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from src.artifacts import Artifact


@dataclass
class ToolResult:
    """Return type for BaseTool.execute() when producing artifacts.

    Carries text content (what the LLM reads as the tool response)
    plus optional artifacts (files, images, data produced by the tool).

    Tools can return either a plain str or a ToolResult.
    """

    content: str
    artifacts: list[Artifact] = field(default_factory=list)


class BaseTool(abc.ABC):
    """Abstract base class for all tools.

    Subclasses must define:
        name: str              — unique tool name (used by registry and LLM)
        description: str       — docstring the LLM sees when deciding to call the tool
        input_schema: type     — Pydantic BaseModel subclass defining input parameters

    Subclasses must implement:
        execute(**kwargs) -> str | ToolResult — the actual tool logic
    """

    name: str
    description: str
    input_schema: Type[BaseModel]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Auto-register every concrete subclass in the tool registry."""
        super().__init_subclass__(**kwargs)
        if (
            hasattr(cls, "name")
            and hasattr(cls, "input_schema")
            and not getattr(cls, "__abstractmethods__", None)
        ):
            from src.tools.registry import _register_class

            _register_class(cls)

    @abc.abstractmethod
    def execute(self, **kwargs: Any) -> str | ToolResult:
        """Run the tool logic. Receives validated inputs as keyword arguments.

        Can return either:
        - str: plain text result (backwards compatible)
        - ToolResult: text content + artifacts (images, files, data)
        """
        ...

    def to_langchain_tool(self) -> StructuredTool:
        """Convert this tool to a LangChain StructuredTool for use with bind_tools().

        NOTE: We do NOT use response_format="content_and_artifact" because
        StructuredTool.invoke() with that flag only returns the content string
        and discards artifacts when called directly. Instead we return ToolResult
        and let handle_tool_result() in BaseAgent do the unpacking.
        """
        instance = self

        def _run(**kwargs: Any) -> str | ToolResult:
            try:
                result = instance.execute(**kwargs)
                if isinstance(result, ToolResult):
                    return result
                return str(result)
            except Exception as e:
                return f"Tool error ({instance.name}): {e}"

        return StructuredTool.from_function(
            func=_run,
            name=self.name,
            description=self.description,
            args_schema=self.input_schema,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
