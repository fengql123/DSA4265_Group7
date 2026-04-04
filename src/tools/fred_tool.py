"""FRED economic data tool.

Provides agents with macroeconomic indicators from the Federal Reserve.
"""

from __future__ import annotations

import json
import os
from typing import Any

from pydantic import BaseModel, Field

from src.tools.base_tool import BaseTool

SERIES_DESCRIPTIONS = {
    "GDP": "Gross Domestic Product",
    "UNRATE": "Unemployment Rate",
    "CPIAUCSL": "Consumer Price Index",
    "DFF": "Federal Funds Rate",
    "T10Y2Y": "10Y-2Y Treasury Yield Spread",
    "T10YIE": "10-Year Breakeven Inflation",
    "VIXCLS": "VIX Volatility Index",
    "DCOILWTICO": "Crude Oil Price (WTI)",
    "UMCSENT": "Consumer Sentiment",
    "HOUST": "Housing Starts",
}


class FredDataInput(BaseModel):
    """Input schema for the FRED data tool."""

    series_ids: list[str] = Field(
        description=(
            "List of FRED series IDs to fetch. Available: "
            "GDP, UNRATE, CPIAUCSL, DFF, T10Y2Y, T10YIE, VIXCLS, DCOILWTICO, UMCSENT, HOUST."
        )
    )
    num_observations: int = Field(
        default=12,
        description="Number of most recent observations to return per series.",
    )


class FredDataTool(BaseTool):
    name = "get_fred_data"
    description = (
        "Fetch macroeconomic data from FRED (Federal Reserve Economic Data). "
        "Use this to understand the macroeconomic environment affecting a company."
    )
    input_schema = FredDataInput

    def execute(self, series_ids: list[str], num_observations: int = 12) -> str:
        from fredapi import Fred

        api_key = os.getenv("FRED_API_KEY")
        if not api_key:
            return "Error: FRED_API_KEY not set in environment. Cannot fetch macro data."

        fred = Fred(api_key=api_key)
        results = {}

        for series_id in series_ids:
            try:
                data = fred.get_series(series_id)
                if data is not None and not data.empty:
                    recent = data.tail(num_observations)
                    results[series_id] = {
                        "description": SERIES_DESCRIPTIONS.get(series_id, series_id),
                        "latest_value": float(recent.iloc[-1]),
                        "observations": {
                            str(date.date()): float(val)
                            for date, val in recent.items()
                            if not (val != val)  # skip NaN
                        },
                    }
            except Exception as e:
                results[series_id] = {"error": str(e)}

        if not results:
            return "No FRED data retrieved. Check series IDs."

        return json.dumps(results, indent=2)
