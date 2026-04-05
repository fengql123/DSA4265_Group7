from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field

from src.artifacts import Artifact, ArtifactType
from src.tools.base_tool import BaseTool, ToolResult


@lru_cache(maxsize=1)
def _get_finbert_pipeline():
    from transformers import pipeline

    return pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        truncation=True,
        max_length=512,
    )


def _signed_score(label: str, confidence: float) -> float:
    label = label.lower()
    if label == "positive":
        return round(confidence, 4)
    if label == "negative":
        return round(-confidence, 4)
    return 0.0


class SentimentAnalysisInput(BaseModel):
    texts: list[str] = Field(
        description="List of news snippets or earnings transcript excerpts to analyze."
    )


class SentimentTimelineInput(BaseModel):
    ticker: str = Field(description="Stock ticker symbol, e.g. AAPL.")
    dates: list[str] = Field(description="Dates corresponding to sentiment scores.")
    scores: list[float] = Field(description="Sentiment scores between -1 and 1.")


class SentimentAnalysisTool(BaseTool):
    name = "analyze_sentiment"
    description = (
        "Run FinBERT sentiment analysis on a list of financial text chunks. "
        "Returns per-text labels, confidence scores, signed sentiment scores, "
        "and an aggregate summary."
    )
    input_schema = SentimentAnalysisInput

    def execute(self, texts: list[str]) -> str:
        if not texts:
            return json.dumps(
                {
                    "overall_sentiment": "neutral",
                    "average_score": 0.0,
                    "items": [],
                    "summary": "No text provided for sentiment analysis.",
                },
                indent=2,
            )

        clean_texts = [t.strip() for t in texts if t and t.strip()]
        if not clean_texts:
            return json.dumps(
                {
                    "overall_sentiment": "neutral",
                    "average_score": 0.0,
                    "items": [],
                    "summary": "No valid non-empty text provided.",
                },
                indent=2,
            )

        pipe = _get_finbert_pipeline()
        preds = pipe(clean_texts)

        items = []
        signed_scores = []

        for text, pred in zip(clean_texts, preds):
            label = pred["label"].lower()
            confidence = float(pred["score"])
            score = _signed_score(label, confidence)
            signed_scores.append(score)

            items.append(
                {
                    "text": text,
                    "label": label,
                    "confidence": round(confidence, 4),
                    "sentiment_score": score,
                }
            )

        avg_score = round(sum(signed_scores) / len(signed_scores), 4)

        if avg_score > 0.15:
            overall = "bullish"
        elif avg_score < -0.15:
            overall = "bearish"
        else:
            overall = "neutral"

        positive = sum(1 for x in items if x["label"] == "positive")
        negative = sum(1 for x in items if x["label"] == "negative")
        neutral = sum(1 for x in items if x["label"] == "neutral")

        strongest_positive = sorted(
            [x for x in items if x["label"] == "positive"],
            key=lambda x: x["confidence"],
            reverse=True,
        )[:3]

        strongest_negative = sorted(
            [x for x in items if x["label"] == "negative"],
            key=lambda x: x["confidence"],
            reverse=True,
        )[:3]

        result = {
            "overall_sentiment": overall,
            "average_score": avg_score,
            "counts": {
                "positive": positive,
                "neutral": neutral,
                "negative": negative,
            },
            "strongest_positive": strongest_positive,
            "strongest_negative": strongest_negative,
            "items": items,
            "summary": (
                f"Analyzed {len(items)} text chunks. "
                f"Overall sentiment is {overall} with average score {avg_score}."
            ),
        }

        return json.dumps(result, indent=2)


class SentimentTimelineTool(BaseTool):
    name = "plot_sentiment_timeline"
    description = (
        "Generate a sentiment timeline chart from dates and sentiment scores. "
        "Returns a chart artifact plus a short summary."
    )
    input_schema = SentimentTimelineInput

    def execute(self, ticker: str, dates: list[str], scores: list[float]) -> ToolResult:
        if not dates or not scores or len(dates) != len(scores):
            return ToolResult(
                content=(
                    "Could not generate timeline chart because dates/scores were missing "
                    "or had mismatched lengths."
                ),
                artifacts=[],
            )

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out_dir = Path("outputs")
        out_dir.mkdir(exist_ok=True)

        chart_path = out_dir / f"{ticker}_sentiment_timeline.png"

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(dates, scores, marker="o")
        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_title(f"{ticker} Sentiment Timeline")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sentiment Score")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(chart_path, dpi=120)
        plt.close(fig)

        artifact = Artifact(
            artifact_type=ArtifactType.IMAGE,
            path=str(chart_path),
            mime_type="image/png",
            description=f"{ticker} sentiment timeline",
        )

        return ToolResult(
            content=f"Saved sentiment timeline chart to {chart_path}",
            artifacts=[artifact],
        )