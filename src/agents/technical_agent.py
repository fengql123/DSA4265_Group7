"""Technical Analysis Agent — Simple implementation.

Uses the get_market_data tool via a real ReAct loop to fetch live price
data, moving averages, and beta from yfinance, then produces a TechnicalReport.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import BaseAgent
from src.config import get_llm, load_prompt
from src.schemas import TechnicalReport
from src.artifacts import Artifact
from src.tools.base_tool import ToolResult
from src.tools.registry import get_tools


class TechnicalAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_name="technical",
            tool_names=["get_market_data"],
            output_field="technical_report",
            output_model=TechnicalReport,
        )

    def get_system_prompt(self, state: dict) -> str:
        template = load_prompt(self.agent_name)
        return template.format(
            ticker=state.get("ticker", "UNKNOWN"),
            date=state.get("analysis_date", "UNKNOWN"),
        )

    def build_messages(self, state: dict) -> list:
        return [
            SystemMessage(content=self.get_system_prompt(state)),
            HumanMessage(
                content=(
                    f"Analyze the technicals of {state.get('ticker', 'UNKNOWN')} "
                    f"as of {state.get('analysis_date', 'UNKNOWN')}. "
                    f"Use your tools to gather data, then provide your analysis."
                )
            ),
        ]

    def get_tools(self) -> list:
        tools = get_tools(self.tool_names)
        if self.mcp_servers:
            tools.extend(self._load_mcp_tools())
        return tools

    def handle_tool_result(self, result: Any) -> tuple[str, list[Artifact]]:
        if isinstance(result, ToolResult):
            return (result.content, result.artifacts)
        if isinstance(result, tuple) and len(result) == 2:
            return (str(result[0]), list(result[1]) if result[1] else [])
        return (str(result), [])

    def build_artifact_message(self, artifacts: list[Artifact]) -> HumanMessage | None:
        return None

    def parse_output(self, messages: list) -> TechnicalReport:
        llm = get_llm().with_structured_output(self.output_model)
        messages = messages + [
            HumanMessage(content="Now produce your final structured technical report.")
        ]
        return llm.invoke(messages)

    def build_result(self, output: object, artifacts: list[Artifact]) -> dict:
        result = {self.output_field: output}
        if artifacts:
            result["artifacts"] = artifacts
        return result

    def is_vision_capable(self) -> bool:
        return False
