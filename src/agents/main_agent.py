"""Main Synthesizer Agent.

Receives a natural language query, extracts ticker/date, calls sub-agent
tools (sentiment, fundamental, risk) via the ReAct loop, and synthesizes
the results into an InvestmentMemo.

Sub-agents are wrapped as tools via AgentTool — each runs its own full
ReAct loop internally when called.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import BaseAgent
from src.config import get_llm, load_prompt, load_settings
from src.schemas import InvestmentMemo
from src.artifacts import Artifact
from src.tools.base_tool import ToolResult


class MainAgent(BaseAgent):
    def __init__(self, disabled_agents: list[str] | None = None):
        super().__init__(
            agent_name="main",
            tool_names=[],  # No registry tools — uses AgentTool-wrapped sub-agents
            output_field="investment_memo",
            output_model=InvestmentMemo,
        )
        # Optional ablation flag: skip wiring in these sub-agents.
        # Used by Phase 7 ablation in the backtest pipeline.
        self.disabled_agents: set[str] = set(disabled_agents or [])

    # ------------------------------------------------------------------
    # Override implementations
    # ------------------------------------------------------------------

    def get_system_prompt(self, state: dict) -> str:
        # The rewritten prompt is fully date-agnostic — no .format()
        # substitution needed. The main agent obtains `today` at
        # runtime via the `get_today_date` tool, and the user's
        # explicit date (if any) comes through the HumanMessage query.
        return load_prompt(self.agent_name)

    def build_messages(self, state: dict) -> list:
        return [
            SystemMessage(content=self.get_system_prompt(state)),
            HumanMessage(content=state.get("query", "")),
        ]

    def get_tools(self) -> list:
        """Wire get_today_date + sub-agent AgentTools."""
        from src.agents.fundamental_agent import FundamentalAgent
        from src.agents.risk_agent import RiskAgent
        from src.agents.sentiment_agent import SentimentAgent
        from src.agents.technical_agent import TechnicalAgent
        from src.tools.agent_tool import AgentTool
        from src.tools.registry import get_tool_by_name

        # Always prepend get_today_date so the LLM can resolve
        # "today" without any date baked into the system prompt.
        tools = [get_tool_by_name("get_today_date")]

        sub_agents = {
            "sentiment": SentimentAgent,
            "fundamental": FundamentalAgent,
            "technical": TechnicalAgent,
            "risk": RiskAgent,
        }
        for name, cls in sub_agents.items():
            if name in self.disabled_agents:
                continue
            tools.append(AgentTool(cls()).to_langchain_tool())

        # Also include MCP tools if configured
        if self.mcp_servers:
            tools.extend(self._load_mcp_tools())

        return tools

    def handle_tool_result(self, result: Any) -> tuple[str, list[Artifact]]:
        if isinstance(result, tuple) and len(result) == 2:
            content, artifacts = result
            return (str(content), list(artifacts) if artifacts else [])
        if isinstance(result, ToolResult):
            return (result.content, result.artifacts)
        return (str(result), [])

    def build_artifact_message(self, artifacts: list[Artifact]) -> HumanMessage | None:
        if not self.is_vision_capable():
            return None

        image_artifacts = [a for a in artifacts if a.is_image]
        if not image_artifacts:
            return None

        content_blocks: list[dict] = [
            {"type": "text", "text": "Charts/images from the analysis:"}
        ]
        for art in image_artifacts:
            try:
                content_blocks.append(art.to_multimodal_block())
                if art.description:
                    content_blocks.append({"type": "text", "text": f"({art.description})"})
            except FileNotFoundError:
                content_blocks.append({"type": "text", "text": f"[Image not found: {art.path}]"})

        if len(content_blocks) <= 1:
            return None

        return HumanMessage(content=content_blocks)

    def parse_output(self, messages: list) -> InvestmentMemo:
        llm = get_llm().with_structured_output(self.output_model)
        messages = messages + [
            HumanMessage(
                content=(
                    "Now produce the final structured investment memo. "
                    "Call no more tools. Fill every field in the InvestmentMemo schema, "
                    "keep the recommendation consistent with the evidence, and ensure "
                    "report_markdown contains the sections: Executive Summary, "
                    "Fundamental Analysis, Technical Outlook, Sentiment Analysis, "
                    "Risk Factors, Evidence Quality & Limitations, and Investment Recommendation."
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
        cfg = load_settings()
        llm_cfg = cfg.get("llm", {})
        flag = llm_cfg.get("vision_enabled")
        if flag is not None:
            return bool(flag)
        model = llm_cfg.get("model", "").lower()
        return any(v in model for v in ["claude", "gpt-4", "gemini"])
