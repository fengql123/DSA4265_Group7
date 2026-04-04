"""AgentTool — wraps a BaseAgent so it can be called as a tool by other agents.

Sub-agents are both BaseAgent subclasses (with their own ReAct loops, tools,
prompts) AND callable as tools by a parent agent (e.g. MainAgent).

Usage:
    from src.tools.agent_tool import AgentTool
    from src.agents.sentiment_agent import SentimentAgent

    tool = AgentTool(SentimentAgent())
    lc_tool = tool.to_langchain_tool()
    # Now usable with llm.bind_tools([lc_tool])
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.tools.base_tool import ToolResult
from src.tools.base_tool import BaseTool


class AgentToolInput(BaseModel):
    """Input schema for sub-agent tools called by the MainAgent."""

    ticker: str = Field(description="Stock ticker symbol (e.g. 'AAPL', 'MSFT').")
    start_date: str = Field(description="Start date for analysis (YYYY-MM-DD).")
    end_date: str = Field(description="End date for analysis (YYYY-MM-DD).")


class AgentTool(BaseTool):
    """Wraps a BaseAgent so it can be called as a tool.

    When invoked, runs the wrapped agent's full run() method (which may
    include its own ReAct loop with its own tools) and returns the
    structured report as JSON text + any artifacts produced.
    """

    # These are set dynamically in __init__, not as class attributes,
    # so __init_subclass__ auto-registration is skipped (AgentTool is
    # not a concrete tool — each instance wraps a different agent).
    input_schema = AgentToolInput

    def __init__(self, agent):
        from src.agents.base import BaseAgent

        self.agent = agent
        self.name = agent.agent_name
        self.description = f"Run the {agent.agent_name} analysis agent for a stock ticker."

    def execute(self, ticker: str, start_date: str, end_date: str) -> str | ToolResult:
        """Run the wrapped agent's full pipeline and return its report."""
        # Validate inputs before running the sub-agent
        error = self._validate_inputs(ticker, start_date, end_date)
        if error:
            return error

        # Build state for the sub-agent
        from datetime import datetime

        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        lookback_days = (end - start).days

        sub_state = {
            "ticker": ticker.upper(),
            "analysis_date": end_date,
            "start_date": start_date,
            "lookback_days": lookback_days,
            "errors": [],
        }

        # Run the sub-agent (which has its own ReAct loop, tools, etc.)
        result = self.agent.run(sub_state)

        # Extract the report and artifacts
        report = result.get(self.agent.output_field)
        artifacts = result.get("artifacts", [])
        errors = result.get("errors", [])

        if report:
            content = report.model_dump_json(indent=2)
        elif errors:
            content = f"Agent {self.agent.agent_name} failed: {'; '.join(errors)}"
        else:
            content = f"Agent {self.agent.agent_name} returned no report."

        if artifacts:
            return ToolResult(content=content, artifacts=artifacts)
        return content

    @staticmethod
    def _validate_inputs(ticker: str, start_date: str, end_date: str) -> str | None:
        """Validate ticker and date range. Returns error message or None if valid."""
        from datetime import datetime

        # Validate ticker format
        ticker = ticker.strip().upper()
        if not ticker or not ticker.replace("-", "").replace(".", "").isalnum():
            return f"Invalid ticker symbol: '{ticker}'. Must be alphanumeric (e.g. AAPL, BRK-B)."

        # Validate date formats
        for label, date_str in [("start_date", start_date), ("end_date", end_date)]:
            try:
                datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                return f"Invalid {label}: '{date_str}'. Must be YYYY-MM-DD format."

        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        if start >= end:
            return f"start_date ({start_date}) must be before end_date ({end_date})."

        # Validate ticker exists via yfinance (lightweight check)
        try:
            import yfinance as yf

            stock = yf.Ticker(ticker)
            info = stock.info
            # yfinance returns an info dict even for invalid tickers,
            # but it won't have a 'shortName' or 'currentPrice'
            if not info.get("shortName") and not info.get("currentPrice"):
                return f"Ticker '{ticker}' not found. Please verify the stock symbol."
        except Exception:
            return f"Could not validate ticker '{ticker}'. Check your network connection."

        return None
