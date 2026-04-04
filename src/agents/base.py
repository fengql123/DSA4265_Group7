"""Base agent class and unified agent loader.

All agents inherit from BaseAgent. The class provides:
- A ReAct tool-calling loop (run()) that calls abstract methods at each step
- Parallel tool execution when the LLM issues multiple tool calls in one round
- ALL override points are abstract — subclasses MUST implement them

Override points (all abstract):
    get_system_prompt(state)         — build system prompt
    build_messages(state)            — construct initial [SystemMessage, HumanMessage]
    get_tools()                      — return list of LangChain tools
    handle_tool_result(result)       — unpack tool return into (content, artifacts)
    build_artifact_message(artifacts)— create multimodal HumanMessage from image artifacts
    parse_output(messages)           — produce structured output from conversation
    build_result(output, artifacts)  — construct return dict for LangGraph
    is_vision_capable()              — check if LLM supports multimodal input

Concrete method:
    run(state)                       — ReAct loop skeleton calling abstract methods
"""

from __future__ import annotations

import abc
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from src.artifacts import Artifact
from src.tools.base_tool import ToolResult

if TYPE_CHECKING:
    from src.schemas import PipelineState


class BaseAgent(abc.ABC):
    """Abstract base class for all agents in the pipeline.

    Subclasses MUST implement all abstract methods.
    Only run() is concrete — it provides the ReAct loop skeleton.
    """

    agent_name: str
    tool_names: list[str]
    output_field: str
    output_model: type
    max_tool_rounds: int

    def __init__(
        self,
        agent_name: str,
        tool_names: list[str],
        output_field: str,
        output_model: type,
        mcp_servers: dict | None = None,
        max_tool_rounds: int = 10,
        debug: bool = False,
    ):
        self.agent_name = agent_name
        self.tool_names = tool_names
        self.output_field = output_field
        self.output_model = output_model
        self.mcp_servers = mcp_servers or {}
        self.max_tool_rounds = max_tool_rounds
        self.debug = debug

    def _log(self, msg: str) -> None:
        """Print a debug message if debug mode is enabled."""
        if self.debug:
            print(f"  [{self.agent_name}] {msg}")

    # ------------------------------------------------------------------
    # Abstract methods — subclasses MUST implement ALL of these
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def get_system_prompt(self, state: dict) -> str:
        """Build the system prompt string."""
        ...

    @abc.abstractmethod
    def build_messages(self, state: dict) -> list:
        """Construct the initial message list [SystemMessage, HumanMessage, ...]."""
        ...

    @abc.abstractmethod
    def get_tools(self) -> list:
        """Return list of LangChain tool objects for this agent."""
        ...

    @abc.abstractmethod
    def handle_tool_result(self, result: Any) -> tuple[str, list[Artifact]]:
        """Unpack a tool's return value into (content_str, artifacts)."""
        ...

    @abc.abstractmethod
    def build_artifact_message(self, artifacts: list[Artifact]) -> HumanMessage | None:
        """Create a multimodal HumanMessage from image artifacts, or None."""
        ...

    @abc.abstractmethod
    def parse_output(self, messages: list) -> object:
        """Produce structured Pydantic output from the conversation history."""
        ...

    @abc.abstractmethod
    def build_result(self, output: object, artifacts: list[Artifact]) -> dict:
        """Construct the return dict for LangGraph state update."""
        ...

    @abc.abstractmethod
    def is_vision_capable(self) -> bool:
        """Check if the current LLM supports multimodal (vision) input."""
        ...

    # ------------------------------------------------------------------
    # Concrete: ReAct loop skeleton
    # ------------------------------------------------------------------

    def run(self, state: dict) -> dict:
        """Execute the agent as a LangGraph node.

        Flow:
        1. build_messages(state)         — initial messages
        2. get_tools()                   — tools for this agent
        3. ReAct loop (up to max_tool_rounds):
           a. LLM responds (may include tool_calls)
           b. If multiple tool_calls → execute in parallel via ThreadPool
           c. handle_tool_result()       — unpack each result
           d. build_artifact_message()   — optionally embed images
        4. parse_output(messages)        — structured output
        5. build_result(output, artifacts) — state update dict
        """
        try:
            self._log(f"Input state: {state}")
            messages = self.build_messages(state)
            tools = self.get_tools()
            self._log(f"Tools: {[t.name for t in tools] if tools else '(none)'}")
            collected_artifacts: list[Artifact] = []

            if tools:
                from src.config import get_llm

                llm = get_llm()
                llm_with_tools = llm.bind_tools(tools)
                tool_map = {t.name: t for t in tools}

                for _round in range(self.max_tool_rounds):
                    self._log(f"Round {_round + 1}/{self.max_tool_rounds}")
                    response = llm_with_tools.invoke(messages)
                    messages.append(response)

                    if not response.tool_calls:
                        self._log(f"LLM done (no more tool calls)")
                        self._log(f"LLM response: {response.content[:200] if response.content else '(empty)'}...")
                        break

                    tool_names_called = [tc["name"] for tc in response.tool_calls]
                    self._log(f"Calling tools: {tool_names_called}")

                    # Execute tool calls — parallel if multiple
                    if len(response.tool_calls) == 1:
                        results = [self._execute_tool_call(response.tool_calls[0], tool_map)]
                    else:
                        results = self._execute_tool_calls_parallel(response.tool_calls, tool_map)

                    # Append ToolMessages and artifact messages
                    round_artifacts: list[Artifact] = []
                    for tool_call, content_str, artifacts in results:
                        collected_artifacts.extend(artifacts)
                        round_artifacts.extend(artifacts)
                        messages.append(
                            ToolMessage(content=content_str, tool_call_id=tool_call["id"])
                        )
                        self._log(f"Tool '{tool_call['name']}' returned {len(content_str)} chars, {len(artifacts)} artifacts")

                    art_msg = self.build_artifact_message(round_artifacts)
                    if art_msg:
                        messages.append(art_msg)
                        self._log(f"Injected multimodal message with {len(round_artifacts)} artifacts")

            self._log("Parsing structured output...")
            output = self.parse_output(messages)
            self._log(f"Output: {type(output).__name__}")
            result = self.build_result(output, collected_artifacts)
            self._log(f"Result keys: {list(result.keys())}, artifacts: {len(collected_artifacts)}")
            return result

        except Exception as e:
            self._log(f"ERROR: {e!s}")
            return {"errors": [f"{self.agent_name}: {e!s}"]}

    def as_node(self):
        """Return a callable for LangGraph's add_node()."""
        return self.run

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.agent_name!r}, "
            f"tools={self.tool_names!r}, "
            f"output={self.output_field!r})"
        )

    # ------------------------------------------------------------------
    # Internal helpers (not override points)
    # ------------------------------------------------------------------

    def _execute_tool_call(self, tool_call: dict, tool_map: dict) -> tuple[dict, str, list[Artifact]]:
        """Execute a single tool call and return (tool_call, content, artifacts)."""
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        if tool_name in tool_map:
            try:
                # Call the underlying function directly (not .invoke() which stringifies)
                # so we preserve ToolResult objects with their artifacts.
                raw_result = tool_map[tool_name].func(**tool_args)
            except Exception as e:
                raw_result = f"Tool error: {e}"
        else:
            raw_result = f"Unknown tool: {tool_name}"

        content_str, artifacts = self.handle_tool_result(raw_result)
        return (tool_call, content_str, artifacts)

    def _execute_tool_calls_parallel(
        self, tool_calls: list[dict], tool_map: dict
    ) -> list[tuple[dict, str, list[Artifact]]]:
        """Execute multiple tool calls in parallel. Returns results in original order."""
        with ThreadPoolExecutor(max_workers=len(tool_calls)) as executor:
            futures = {
                executor.submit(self._execute_tool_call, tc, tool_map): tc
                for tc in tool_calls
            }
            results = []
            for future in as_completed(futures):
                results.append(future.result())

        # Sort by original tool_call order (providers require matching order)
        call_id_order = [tc["id"] for tc in tool_calls]
        results.sort(key=lambda r: call_id_order.index(r[0]["id"]))
        return results

    def _load_mcp_tools(self) -> list:
        """Load tools from MCP servers via langchain-mcp-adapters."""
        import asyncio

        from langchain_mcp_adapters.client import MultiServerMCPClient

        async def _get():
            async with MultiServerMCPClient(self.mcp_servers) as client:
                return client.get_tools()

        return asyncio.run(_get())


# ---------------------------------------------------------------------------
# Agent Registry & Unified Loader
# ---------------------------------------------------------------------------

_AGENT_CLASSES: dict[str, type[BaseAgent]] | None = None


def _get_agent_classes() -> dict[str, type[BaseAgent]]:
    """Lazy-load agent subclasses."""
    global _AGENT_CLASSES
    if _AGENT_CLASSES is None:
        from src.agents.main_agent import MainAgent

        _AGENT_CLASSES = {"main": MainAgent}
    return _AGENT_CLASSES


def load_agent(agent_name: str) -> BaseAgent:
    """Load any agent by name."""
    classes = _get_agent_classes()
    if agent_name not in classes:
        available = ", ".join(sorted(classes.keys()))
        raise KeyError(f"Unknown agent: '{agent_name}'. Available: {available}")
    return classes[agent_name]()


def register_agent(name: str, agent_class: type[BaseAgent]) -> None:
    """Register a custom agent class."""
    classes = _get_agent_classes()
    classes[name] = agent_class
