"""Sentiment Analysis Agent — STUB.

TODO: Teammates implement this agent. Replace mock implementations with real
logic using RAG retrieval, FinBERT sentiment analysis, and chart generation.

The stub returns mock data with a dummy chart image to test multimodality.
It implements all abstract methods but skips the LLM — parse_output() returns
a hardcoded SentimentReport directly.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import BaseAgent
from src.schemas import SentimentReport
from src.artifacts import Artifact, ArtifactType


class SentimentAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_name="sentiment",
            tool_names=[],  # TODO: add real tools
            output_field="sentiment_report",
            output_model=SentimentReport,
        )
        self._mock_artifacts: list[Artifact] = []

    # ------------------------------------------------------------------
    # STUB implementations — teammates replace these with real logic
    # ------------------------------------------------------------------

    def get_system_prompt(self, state: dict) -> str:
        return "You are a sentiment analyst."

    def build_messages(self, state: dict) -> list:
        return [
            SystemMessage(content=self.get_system_prompt(state)),
            HumanMessage(content=f"Analyze sentiment for {state.get('ticker', 'UNKNOWN')}."),
        ]

    def get_tools(self) -> list:
        return []  # TODO: add rag_retrieve, analyze_sentiment, plot_sentiment_timeline

    def handle_tool_result(self, result: Any) -> tuple[str, list[Artifact]]:
        if isinstance(result, tuple) and len(result) == 2:
            return (str(result[0]), list(result[1]) if result[1] else [])
        return (str(result), [])

    def build_artifact_message(self, artifacts: list[Artifact]) -> HumanMessage | None:
        return None  # TODO: implement multimodal injection

    def parse_output(self, messages: list) -> SentimentReport:
        """MOCK: Return random sentiment data instead of calling LLM."""
        # Extract ticker from the messages
        ticker = "UNKNOWN"
        for msg in messages:
            content = msg.content if hasattr(msg, "content") else str(msg)
            if isinstance(content, str) and "ticker" not in content:
                # Try to find ticker in the message
                pass

        score = round(random.uniform(-0.8, 0.8), 2)
        sentiment = "bullish" if score > 0.2 else "bearish" if score < -0.2 else "neutral"

        # Generate dummy chart for multimodal testing
        chart_path = self._generate_mock_chart("MOCK", score)
        if chart_path:
            self._mock_artifacts.append(
                Artifact(
                    artifact_type=ArtifactType.IMAGE,
                    path=str(chart_path),
                    mime_type="image/png",
                    description="Sentiment timeline (mock)",
                )
            )

        return SentimentReport(
            ticker="MOCK",
            overall_sentiment=sentiment,
            sentiment_score=score,
            key_themes=["revenue growth", "market competition", "product innovation"],
            evidence=[
                f"Mock: Sentiment is {sentiment} with score {score}.",
                "Mock: Recent earnings exceeded analyst expectations.",
            ],
            chart_paths=[str(chart_path)] if chart_path else [],
            summary=f"Mock sentiment analysis: {sentiment} (score: {score}).",
        )

    def build_result(self, output: object, artifacts: list[Artifact]) -> dict:
        # Include any mock artifacts generated in parse_output
        all_artifacts = artifacts + self._mock_artifacts
        self._mock_artifacts = []  # Reset for next run
        result = {self.output_field: output}
        if all_artifacts:
            result["artifacts"] = all_artifacts
        return result

    def is_vision_capable(self) -> bool:
        return False

    @staticmethod
    def _generate_mock_chart(ticker: str, score: float) -> Path | None:
        """Generate a small dummy sentiment chart for multimodal testing."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 3))
            dates = ["Q1", "Q2", "Q3", "Q4"]
            scores = [round(score + random.uniform(-0.3, 0.3), 2) for _ in dates]
            colors = ["#2ecc71" if s > 0 else "#e74c3c" for s in scores]
            ax.bar(dates, scores, color=colors)
            ax.set_title(f"{ticker} Sentiment (Mock)")
            ax.set_ylabel("Score")
            ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
            ax.set_ylim(-1, 1)
            fig.tight_layout()

            out_dir = Path("outputs")
            out_dir.mkdir(exist_ok=True)
            path = out_dir / f"{ticker}_mock_sentiment.png"
            fig.savefig(path, dpi=100)
            plt.close(fig)
            return path
        except Exception:
            return None
