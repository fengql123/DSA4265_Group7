"""Risk Assessment Agent — STUB.

TODO: Teammates implement this agent. Replace mock implementations with real
logic using RAG retrieval to analyze SEC filing risk factors and news.
"""

from __future__ import annotations

import random
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import BaseAgent
from src.schemas import RiskReport
from src.artifacts import Artifact


class RiskAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_name="risk",
            tool_names=[],  # TODO: add real tools
            output_field="risk_report",
            output_model=RiskReport,
        )

    # ------------------------------------------------------------------
    # STUB implementations — teammates replace these with real logic
    # ------------------------------------------------------------------

    def get_system_prompt(self, state: dict) -> str:
        return "You are a risk analyst."

    def build_messages(self, state: dict) -> list:
        return [
            SystemMessage(content=self.get_system_prompt(state)),
            HumanMessage(content=f"Assess risks for {state.get('ticker', 'UNKNOWN')}."),
        ]

    def get_tools(self) -> list:
        return []  # TODO: add rag_retrieve

    def handle_tool_result(self, result: Any) -> tuple[str, list[Artifact]]:
        if isinstance(result, tuple) and len(result) == 2:
            return (str(result[0]), list(result[1]) if result[1] else [])
        return (str(result), [])

    def build_artifact_message(self, artifacts: list[Artifact]) -> HumanMessage | None:
        return None

    def parse_output(self, messages: list) -> RiskReport:
        """MOCK: Return random risk assessment instead of calling LLM."""
        level = random.choice(["low", "moderate", "high"])
        return RiskReport(
            ticker="MOCK",
            risk_factors=[
                "Mock: Competitive pressure in core markets.",
                "Mock: Regulatory scrutiny in key jurisdictions.",
                "Mock: Supply chain concentration risk.",
            ],
            risk_level=level,
            mitigants=["Mock: Strong cash position.", "Mock: Diversified revenue streams."],
            summary=f"Mock: Risk level is {level}.",
        )

    def build_result(self, output: object, artifacts: list[Artifact]) -> dict:
        result = {self.output_field: output}
        if artifacts:
            result["artifacts"] = artifacts
        return result

    def is_vision_capable(self) -> bool:
        return False
