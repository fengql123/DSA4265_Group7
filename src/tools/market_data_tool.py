"""Market data tool using yfinance.

Provides agents with live market data (prices, fundamentals, etc.).
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from src.tools.base_tool import BaseTool


class MarketDataInput(BaseModel):
    """Input schema for the market data tool."""

    ticker: str = Field(description="Stock ticker symbol (e.g. 'AAPL', 'MSFT').")


class MarketDataTool(BaseTool):
    name = "get_market_data"
    description = (
        "Fetch current market data and fundamentals for a stock ticker. "
        "Returns key metrics including price, P/E ratio, market cap, revenue, "
        "profit margins, and other fundamental data from Yahoo Finance."
    )
    input_schema = MarketDataInput

    # Fields to extract from yfinance info dict
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

    def execute(self, ticker: str) -> str:
        import yfinance as yf

        stock = yf.Ticker(ticker)
        info = stock.info

        result = {f: info[f] for f in self.KEY_FIELDS if f in info and info[f] is not None}

        if not result:
            return f"No market data found for ticker: {ticker}"

        return json.dumps(result, indent=2, default=str)
