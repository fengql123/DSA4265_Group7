"""Market data tool using yfinance.

Provides agents with market data (prices, fundamentals, etc.).

Two modes:
- **live** (default): fetch current yfinance `.info` aggregate fields.
- **historical (as_of set)**: compute price-derived fields (current
  price, 52w range, 50/200 DMAs) from `data/market/{TICKER}.csv`
  as-of a past date, so the pipeline can be backtested without
  lookahead. Static fundamental fields (P/E, margins, analyst
  targets) are intentionally NOT served in historical mode because
  yfinance only offers current snapshots of them — the
  FundamentalAgent is expected to rely on SEC filings retrieved via
  `rag_retrieve` for historical fundamentals.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.tools.base_tool import BaseTool


MARKET_DATA_DIR = Path("data/market")


class MarketDataInput(BaseModel):
    """Input schema for the market data tool."""

    ticker: str = Field(description="Stock ticker symbol (e.g. 'AAPL', 'MSFT').")
    as_of: str | None = Field(
        default=None,
        description=(
            "Optional analysis date (YYYY-MM-DD). If provided, the tool returns a leak-free "
            "price-snapshot computed from locally stored OHLCV as-of that date, instead of live yfinance data. "
            "Always pass this when the user query mentions an analysis date or when you are analyzing a past date."
        ),
    )


class MarketDataTool(BaseTool):
    name = "get_market_data"
    description = (
        "**What**: Returns a JSON snapshot of market, valuation, and fundamental metrics for one ticker. "
        "**When to use**: Any time you need price, valuation multiples, margins, moving averages, beta, or analyst-estimate fields for a single stock. "
        "**Input**: `ticker` (str, e.g. 'AAPL'); `as_of` (optional str, YYYY-MM-DD — pass this whenever the analysis date is in the past, so the tool returns a leak-free historical snapshot). "
        "**Output (live mode, as_of omitted)**: JSON object with available live fields from "
        "`shortName, sector, industry, currentPrice, previousClose, marketCap, "
        "trailingPE, forwardPE, priceToBook, priceToSalesTrailing12Months, revenue, totalRevenue, revenueGrowth, "
        "grossMargins, operatingMargins, profitMargins, returnOnEquity, returnOnAssets, totalDebt, totalCash, debtToEquity, "
        "dividendYield, beta, fiftyTwoWeekHigh, fiftyTwoWeekLow, fiftyDayAverage, twoHundredDayAverage, "
        "earningsGrowth, targetMeanPrice, recommendationKey`. "
        "**Output (historical mode, as_of set)**: A smaller JSON object with `mode=historical_as_of`, `as_of_date`, "
        "`currentPrice` (last close on/before as_of), `previousClose`, `fiftyTwoWeekHigh`, `fiftyTwoWeekLow`, "
        "`fiftyDayAverage`, `twoHundredDayAverage`, and `beta` — all computed from `data/market/{TICKER}.csv`. "
        "Static fundamentals (P/E, margins, analyst targets) are NOT returned in historical mode; "
        "use `rag_retrieve` on SEC filings for historical fundamentals instead. "
        "**Limits**: Historical mode requires `data/market/{TICKER}.csv` to exist and to cover the requested `as_of` date. "
        "If the CSV is missing, historical mode raises an error message rather than silently falling back to live data."
    )
    input_schema = MarketDataInput

    # Fields to extract from live yfinance info dict
    KEY_FIELDS = [
        "shortName", "sector", "industry",
        "currentPrice", "previousClose", "marketCap",
        "trailingPE", "forwardPE", "priceToBook", "priceToSalesTrailing12Months",
        "revenue", "totalRevenue", "revenueGrowth", "revenuePerShare",
        "grossMargins", "operatingMargins", "profitMargins",
        "returnOnEquity", "returnOnAssets",
        "totalDebt", "totalCash", "debtToEquity",
        "dividendYield", "payoutRatio",
        "beta", "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
        "fiftyDayAverage", "twoHundredDayAverage",
        "earningsGrowth", "earningsQuarterlyGrowth",
        "targetMeanPrice", "recommendationKey",
    ]

    def execute(self, ticker: str, as_of: str | None = None) -> str:
        if as_of:
            return self._execute_historical(ticker, as_of)
        return self._execute_live(ticker)

    def _execute_live(self, ticker: str) -> str:
        import yfinance as yf

        stock = yf.Ticker(ticker)
        info = stock.info

        result = {f: info[f] for f in self.KEY_FIELDS if f in info and info[f] is not None}

        if not result:
            return f"No market data found for ticker: {ticker}"

        result["mode"] = "live"
        return json.dumps(result, indent=2, default=str)

    def _execute_historical(self, ticker: str, as_of: str) -> str:
        try:
            as_of_dt = datetime.strptime(as_of, "%Y-%m-%d")
        except ValueError:
            return f"get_market_data error: invalid as_of date '{as_of}' (expected YYYY-MM-DD)"

        # Reuse the OHLCV loader from indicators_tool for consistency.
        from src.tools.indicators_tool import load_ohlcv

        try:
            df = load_ohlcv(ticker, as_of, lookback_days=260)
        except Exception as exc:
            return f"get_market_data error: {exc}"

        if df is None or df.empty:
            return (
                f"No historical OHLCV for {ticker} on or before {as_of}. "
                f"Ensure data/market/{ticker.upper()}.csv exists and covers that date."
            )

        close = df["close"]
        last_close = float(close.iloc[-1])
        prev_close = float(close.iloc[-2]) if len(close) >= 2 else None

        # 52-week stats use the last 252 trading days ending at as_of.
        last_252 = df.tail(252)
        high_52w = float(last_252["high"].max())
        low_52w = float(last_252["low"].min())

        # Moving averages.
        last_50 = df.tail(50)["close"]
        fifty_day_avg = float(last_50.mean()) if len(last_50) == 50 else None
        last_200 = df.tail(200)["close"]
        two_hundred_day_avg = float(last_200.mean()) if len(last_200) == 200 else None

        # Beta: try the static info.json snapshot if present (best-effort only;
        # this is a static fundamental and NOT strictly leak-free).
        beta: float | None = None
        info_path = MARKET_DATA_DIR / f"{ticker.upper()}_info.json"
        if info_path.exists():
            try:
                with info_path.open() as f:
                    static_info = json.load(f)
                raw_beta = static_info.get("beta")
                if isinstance(raw_beta, (int, float)):
                    beta = float(raw_beta)
            except Exception:
                pass

        result: dict[str, Any] = {
            "mode": "historical_as_of",
            "ticker": ticker.upper(),
            "as_of_date": df.index[-1].strftime("%Y-%m-%d"),
            "currentPrice": round(last_close, 4),
        }
        if prev_close is not None:
            result["previousClose"] = round(prev_close, 4)
        result["fiftyTwoWeekHigh"] = round(high_52w, 4)
        result["fiftyTwoWeekLow"] = round(low_52w, 4)
        if fifty_day_avg is not None:
            result["fiftyDayAverage"] = round(fifty_day_avg, 4)
        if two_hundred_day_avg is not None:
            result["twoHundredDayAverage"] = round(two_hundred_day_avg, 4)
        if beta is not None:
            result["beta"] = round(beta, 4)
        result["note"] = (
            "Historical mode returns only price-derived fields from local OHLCV. "
            "For historical fundamentals (P/E, margins, guidance), use rag_retrieve on SEC filings."
        )
        return json.dumps(result, indent=2)
