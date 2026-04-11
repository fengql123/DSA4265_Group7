"""Risk Assessment Agent

Uses the RAG retrieval tool via a real ReAct loop to fetch risk-related content from SEC filings, 
earning transcripts, and news, then produces a structured RiskReport.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import BaseAgent
from src.artifacts import Artifact
from src.config import get_llm, load_prompt
from src.schemas import RiskReport
from src.tools.base_tool import ToolResult
from src.tools.registry import get_tools

class RiskAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_name="risk",
            tool_names=["rag_retrieve"], 
            output_field="risk_report",
            output_model=RiskReport,
        )

    def get_system_prompt(self, state: dict) -> str:
        template = load_prompt(self.agent_name)
        return template.format(
            ticker=state.get("ticker", "UNKNOWN"),
            date=state.get("analysis_date", "UNKNOWN")
        )

    def build_messages(self, state: dict) -> list:
        ticker = state.get("ticker", "UNKNOWN")
        analysis_date = state.get("analysis_date", "UNKNOWN")
        return [
            SystemMessage(content=self.get_system_prompt(state)),
            HumanMessage(content=f"Assess the key risks for {ticker} as of {analysis_date}."),
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

    def parse_output(self, messages: list) -> RiskReport:
        llm = get_llm().with_structured_output(self.output_model)

        messages = messages + [
            HumanMessage(
                content=(
                "Now produce your final structured risk report. "
                "Requirements:\n"
                "- ticker: the stock ticker analyzed\n"
                "- risk_factors: 3 to 6 concise, company-specific risks grounded in the retrieved documents\n"
                "- risk_level: choose exactly one of low, moderate, high\n"
                "- mitigants: concrete offsets or strengths that reduce the risk\n"
                "- summary: a concise explanation balancing the main risks against the mitigants\n"
                "Do not invent facts that were not supported by the retrieved materials."
                )
            )
        ]

        output = llm.invoke(messages)
        return output

    def build_result(self, output: object, artifacts: list[Artifact]) -> dict:
        result = {self.output_field: output}
        if artifacts:
            result["artifacts"] = artifacts
        return result

    def is_vision_capable(self) -> bool:
        return False

