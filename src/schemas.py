"""Pydantic output models and LangGraph TypedDict state.

Output models define the structured data each agent produces.
PipelineState is the TypedDict that flows through the LangGraph graph.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Agent Output Schemas (Pydantic — used with LLM structured output)
# ---------------------------------------------------------------------------


class SentimentReport(BaseModel):
    """Output of the Sentiment Analysis Agent."""

    ticker: str
    overall_sentiment: Literal["bullish", "neutral", "bearish"]
    sentiment_score: float = Field(ge=-1.0, le=1.0, description="Score from -1 (bearish) to 1 (bullish)")
    key_themes: list[str] = Field(description="Major themes identified in the analyzed text")
    evidence: list[str] = Field(description="Key quotes or snippets supporting the analysis")
    chart_paths: list[str] = Field(default_factory=list, description="Paths to generated chart files")
    summary: str = Field(description="Concise summary of the sentiment analysis")


class FundamentalReport(BaseModel):
    """Output of the Fundamental Analysis Agent."""

    ticker: str
    revenue_trend: str = Field(description="Analysis of revenue trajectory")
    margin_analysis: str = Field(description="Analysis of profit margins and trends")
    valuation_assessment: str = Field(description="Assessment of current valuation (P/E, P/S, etc.)")
    macro_context: str = Field(description="Relevant macroeconomic context and its impact")
    key_metrics: dict[str, str | float] = Field(default_factory=dict, description="Key financial metrics extracted")
    summary: str = Field(description="Concise summary of the fundamental analysis")


class TechnicalReport(BaseModel):
    """Output of the Technical Analysis Agent."""

    ticker: str
    current_price: float = Field(description="Current stock price")
    fifty_two_week_high: float = Field(description="52-week high price")
    fifty_two_week_low: float = Field(description="52-week low price")
    moving_avg_50d: float = Field(description="50-day moving average")
    moving_avg_200d: float = Field(description="200-day moving average")
    beta: float = Field(default=1.0, description="Stock beta")
    technical_signal: Literal["bullish", "neutral", "bearish"] = Field(description="Overall technical signal")
    summary: str = Field(description="Concise summary of the technical analysis")


class RiskReport(BaseModel):
    """Output of the Risk Assessment Agent."""

    ticker: str
    risk_factors: list[str] = Field(description="Identified risk factors from filings and news")
    risk_level: Literal["low", "moderate", "high"] = Field(description="Overall risk assessment")
    mitigants: list[str] = Field(default_factory=list, description="Risk mitigants and positive offsets")
    summary: str = Field(description="Concise summary of the risk assessment")


class InvestmentMemo(BaseModel):
    """Output of the Main Synthesizer Agent."""

    ticker: str
    recommendation: Literal["strong_buy", "buy", "hold", "sell", "strong_sell"]
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the recommendation")
    thesis: str = Field(description="Core investment thesis")
    sentiment_summary: str = Field(description="Summary of sentiment findings")
    fundamental_summary: str = Field(description="Summary of fundamental findings")
    technical_summary: str = Field(description="Summary of technical findings")
    risk_summary: str = Field(description="Summary of risk findings")
    report_markdown: str = Field(description="Full formatted investment memo in markdown")


# ---------------------------------------------------------------------------
# LangGraph Pipeline State
# ---------------------------------------------------------------------------


class PipelineState(TypedDict, total=False):
    """State that flows through the LangGraph pipeline.

    The pipeline is: START → MainAgent → END.
    MainAgent calls sub-agents as tools within its ReAct loop.
    """

    # User's natural language query (e.g. "Should I invest in Apple?")
    query: str

    # Main agent output
    investment_memo: InvestmentMemo | None

    # Artifacts produced during analysis (charts, files)
    artifacts: Annotated[list, operator.add]

    # Error tracking
    errors: Annotated[list[str], operator.add]
