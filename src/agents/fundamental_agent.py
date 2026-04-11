"""Fundamental Analysis Agent — Simple implementation.

Uses the get_market_data tool via a real ReAct loop to fetch live data
from yfinance and produce a FundamentalReport. Demonstrates the full
agent flow: LLM calls tools autonomously, receives results, produces
structured output.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import BaseAgent
from src.config import get_llm, load_prompt
from src.schemas import FundamentalReport
from src.artifacts import Artifact
from src.tools.base_tool import ToolResult
from src.tools.registry import get_tools


class FundamentalAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_name="fundamental",
            tool_names=["get_market_data", "get_fred_data", "rag_retrieve"],
            output_field="fundamental_report",
            output_model=FundamentalReport,
        )

    def get_system_prompt(self, state: dict) -> str:
        template = load_prompt(self.agent_name)
        return template.format(
            ticker=state.get("ticker", "UNKNOWN"),
            date=state.get("analysis_date", "UNKNOWN"),
        )

    def build_messages(self, state: dict) -> list:
        ticker = state.get("ticker", "UNKNOWN")
        analysis_date = state.get("analysis_date", "UNKNOWN")
        return [
            SystemMessage(content=self.get_system_prompt(state)),
            HumanMessage(content=f"Analyze the fundamentals of {ticker} as of {analysis_date}."),
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

    def parse_output(self, messages: list) -> FundamentalReport:
        llm = get_llm().with_structured_output(self.output_model)
        messages = messages + [
            HumanMessage(
                content=(
                    "Now produce the final structured fundamental report. "
                    "Ensure the report is evidence-based and consistent with the tool outputs. "
                    "Be specific about revenue trend, margin analysis, valuation assessment, macro context, and key metrics. "
                    "Do not invent data that was not retrieved."
                )
            )
        ]
        return llm.invoke(messages)

    def build_result(self, output: object, artifacts: list[Artifact]) -> dict:
        result = {self.output_field: output}
        if artifacts:
            result["artifacts"] = artifacts
        return result

    def is_vision_capable(self) -> bool:
        return False
