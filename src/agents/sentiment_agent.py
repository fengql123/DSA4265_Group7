from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import BaseAgent
from src.artifacts import Artifact
from src.config import get_llm, load_prompt
from src.schemas import SentimentReport
from src.tools.base_tool import ToolResult
from src.tools.registry import get_tools


class SentimentAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_name="sentiment",
            tool_names=["rag_retrieve", "analyze_sentiment", "plot_sentiment_timeline"],
            output_field="sentiment_report",
            output_model=SentimentReport,
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
        start_date = state.get("start_date", "UNKNOWN")
        lookback_days = state.get("lookback_days", 30)

        return [
            SystemMessage(content=self.get_system_prompt(state)),
            HumanMessage(
                content=(
                    f"Analyze sentiment for {ticker} as of {analysis_date}. "
                    f"Focus on the recent period starting around {start_date} "
                    f"and covering roughly the last {lookback_days} days. "
                    f"Use rag_retrieve to search news and earnings transcript content, "
                    f"then use analyze_sentiment on the retrieved text chunks. "
                    f"If useful, use plot_sentiment_timeline to create a chart. "
                    f"Return a structured sentiment report with overall_sentiment, "
                    f"sentiment_score, key_themes, evidence, chart_paths, and summary."
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

    def parse_output(self, messages: list) -> SentimentReport:
        llm = get_llm().with_structured_output(self.output_model)
        messages = messages + [
            HumanMessage(
                content=(
                    "Now produce your final structured sentiment report. "
                    "Set overall_sentiment to exactly one of: bullish, neutral, bearish. "
                    "Set sentiment_score between -1 and 1. "
                    "Use concise, finance-focused key themes and include supporting evidence snippets."
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