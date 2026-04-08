from __future__ import annotations

import hashlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field

from src.artifacts import Artifact, ArtifactType
from src.tools.base_tool import BaseTool, ToolResult


@lru_cache(maxsize=1)
def _get_finbert_pipeline():
    from transformers import pipeline

    model_name = os.getenv("SENTIMENT_MODEL_PATH", "ProsusAI/finbert")

    return pipeline(
        "sentiment-analysis",
        model=model_name,
        tokenizer=model_name,
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

    @staticmethod
    def _chart_dir() -> Path:
        out_dir = Path("outputs")
        out_dir.mkdir(exist_ok=True)
        return out_dir

    @staticmethod
    def _build_distribution_chart(items: list[dict], digest: str) -> Artifact | None:
        if not items:
            return None

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        counts = {
            "positive": sum(1 for x in items if x["label"] == "positive"),
            "neutral": sum(1 for x in items if x["label"] == "neutral"),
            "negative": sum(1 for x in items if x["label"] == "negative"),
        }

        chart_path = SentimentAnalysisTool._chart_dir() / f"sentiment_distribution_{digest}.png"

        fig, ax = plt.subplots(figsize=(6, 4))
        labels = list(counts.keys())
        values = [counts[label] for label in labels]
        colors = ["#2e8b57", "#808080", "#c0392b"]

        bars = ax.bar(labels, values, color=colors)
        ax.set_title("Sentiment Distribution")
        ax.set_xlabel("Label")
        ax.set_ylabel("Count")
        ax.set_ylim(0, max(values + [1]) + 1)

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.05,
                str(value),
                ha="center",
                va="bottom",
            )

        fig.tight_layout()
        fig.savefig(chart_path, dpi=120)
        plt.close(fig)

        return Artifact(
            artifact_type=ArtifactType.IMAGE,
            path=str(chart_path),
            mime_type="image/png",
            description="Sentiment distribution chart",
            metadata={"chart_type": "distribution"},
        )

    @staticmethod
    def _build_score_chart(items: list[dict], digest: str) -> Artifact | None:
        if not items:
            return None

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        top_items = items[:8]
        labels = []
        scores = []
        colors = []

        for idx, item in enumerate(top_items, 1):
            text = item["text"].splitlines()[0].strip()
            text = (text[:42] + "...") if len(text) > 45 else text
            labels.append(f"{idx}. {text}")
            score = item["sentiment_score"]
            scores.append(score)
            if score > 0:
                colors.append("#2e8b57")
            elif score < 0:
                colors.append("#c0392b")
            else:
                colors.append("#808080")

        chart_path = SentimentAnalysisTool._chart_dir() / f"sentiment_scores_{digest}.png"

        fig, ax = plt.subplots(figsize=(9, max(4, len(labels) * 0.7)))
        y_pos = list(range(len(labels)))
        bars = ax.barh(y_pos, scores, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.axvline(0, linestyle="--", linewidth=1, color="black")
        ax.set_xlim(-1, 1)
        ax.set_xlabel("Sentiment Score")
        ax.set_title("Sentiment Scores by Evidence Chunk")
        ax.invert_yaxis()

        for bar, value in zip(bars, scores):
            ax.text(
                value + (0.03 if value >= 0 else -0.03),
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}",
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=8,
            )

        fig.tight_layout()
        fig.savefig(chart_path, dpi=120)
        plt.close(fig)

        return Artifact(
            artifact_type=ArtifactType.IMAGE,
            path=str(chart_path),
            mime_type="image/png",
            description="Sentiment score chart",
            metadata={"chart_type": "score_by_chunk"},
        )

    @staticmethod
    def _extract_date(text: str) -> str | None:
        match = re.search(r"(?:date=|dated\s)(\d{4}-\d{2}-\d{2})", text)
        if match:
            return match.group(1)

        match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
        if match:
            return match.group(1)

        return None

    @staticmethod
    def _build_timeline_chart(items: list[dict], digest: str) -> Artifact | None:
        dated = []
        for item in items:
            item_date = SentimentAnalysisTool._extract_date(item["text"])
            if item_date:
                dated.append((item_date, item["sentiment_score"]))

        if len(dated) < 2:
            return None

        dated.sort(key=lambda x: x[0])

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        chart_path = SentimentAnalysisTool._chart_dir() / f"sentiment_timeline_{digest}.png"

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot([d[0] for d in dated], [d[1] for d in dated], marker="o", color="#1f77b4")
        ax.axhline(0, linestyle="--", linewidth=1, color="black")
        ax.set_title("Sentiment Timeline")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sentiment Score")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(chart_path, dpi=120)
        plt.close(fig)

        return Artifact(
            artifact_type=ArtifactType.IMAGE,
            path=str(chart_path),
            mime_type="image/png",
            description="Sentiment timeline chart",
            metadata={"chart_type": "timeline"},
        )

    def execute(self, texts: list[str]) -> str | ToolResult:
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

        digest = hashlib.md5(
            "||".join(item["text"][:100] for item in items[:10]).encode("utf-8")
        ).hexdigest()[:10]

        artifacts = [
            artifact
            for artifact in [
                self._build_distribution_chart(items, digest),
                self._build_score_chart(items, digest),
                self._build_timeline_chart(items, digest),
            ]
            if artifact is not None
        ]

        if not artifacts:
            return json.dumps(result, indent=2)

        result["chart_paths"] = [artifact.path for artifact in artifacts]
        return ToolResult(
            content=json.dumps(result, indent=2),
            artifacts=artifacts,
        )


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
