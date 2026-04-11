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
    """Input schema for sub-agent tools called by the MainAgent.

    Only `ticker` and `end_date` are passed. Each sub-agent is responsible
    for choosing its own appropriate lookback window internally based on its
    analytical needs (sentiment ~30d, technical ~1y, fundamental/risk ~2y).
    """

    ticker: str = Field(description="Stock ticker symbol (e.g. 'AAPL', 'MSFT').")
    end_date: str = Field(
        description="Analysis end date in YYYY-MM-DD format. The sub-agent picks its own lookback window."
    )


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
        self.description = (
            f"**What**: Runs the {agent.agent_name} sub-agent and returns its structured report. "
            f"**When to use**: Exactly once per investment analysis, with the analysis `end_date`. "
            f"**Input**: `ticker` (str), `end_date` (str, YYYY-MM-DD). Do NOT pass a start_date — "
            f"the sub-agent chooses its own lookback window. "
            f"**Output**: A JSON-encoded structured report specific to this sub-agent's domain, "
            f"plus any chart/image artifacts the sub-agent generated."
        )

    MAX_RETRIES = 2

    def execute(self, ticker: str, end_date: str) -> str | ToolResult:
        """Run the wrapped agent's full pipeline and return its report.

        Retries once if the sub-agent returns no report and no explicit
        error list — this handles transient structured-output parse
        failures under concurrent batch execution.
        """
        # Validate inputs before running the sub-agent
        error = self._validate_inputs(ticker, end_date)
        if error:
            return error

        report = None
        artifacts = []
        errors = []
        last_attempt_errors: list = []

        for attempt in range(self.MAX_RETRIES):
            # Build a fresh state per attempt so a failed first run
            # does not bleed error state into the retry.
            sub_state = {
                "ticker": ticker.upper(),
                "analysis_date": end_date,
                "errors": [],
            }
            try:
                result = self.agent.run(sub_state)
            except Exception as exc:
                result = {"errors": [f"{self.agent.agent_name}: {exc!s}"]}

            report = result.get(self.agent.output_field)
            artifacts = result.get("artifacts", [])
            errors = result.get("errors", [])
            if report is not None:
                break
            last_attempt_errors = list(errors)

        if report is not None:
            content = report.model_dump_json(indent=2)
        elif last_attempt_errors:
            content = f"Agent {self.agent.agent_name} failed: {'; '.join(last_attempt_errors)}"
        else:
            content = f"Agent {self.agent.agent_name} returned no report."

        if artifacts:
            return ToolResult(content=content, artifacts=artifacts)
        return content

    @staticmethod
    def _validate_inputs(ticker: str, end_date: str) -> str | None:
        """Validate ticker and end_date. Returns error message or None if valid."""
        from datetime import datetime

        # Validate ticker format
        ticker = ticker.strip().upper()
        if not ticker or not ticker.replace("-", "").replace(".", "").isalnum():
            return f"Invalid ticker symbol: '{ticker}'. Must be alphanumeric (e.g. AAPL, BRK-B)."

        # Validate end_date format
        try:
            datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            return f"Invalid end_date: '{end_date}'. Must be YYYY-MM-DD format."

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
