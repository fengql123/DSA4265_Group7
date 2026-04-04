#!/usr/bin/env python3
"""Download macroeconomic indicators from FRED (Federal Reserve Economic Data).

Requires a free API key from https://fred.stlouisfed.org/docs/api/api_key.html
Set FRED_API_KEY in your .env file.

Usage:
    # Download default macro indicators
    python scripts/download_fred.py

    # Download specific series
    python scripts/download_fred.py --series GDP UNRATE CPIAUCSL

    # Download with custom date range
    python scripts/download_fred.py --start 2015-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Key macroeconomic indicators
DEFAULT_SERIES = {
    "GDP": "Gross Domestic Product",
    "UNRATE": "Unemployment Rate",
    "CPIAUCSL": "Consumer Price Index (All Urban)",
    "DFF": "Federal Funds Effective Rate",
    "T10Y2Y": "10Y-2Y Treasury Yield Spread",
    "T10YIE": "10-Year Breakeven Inflation Rate",
    "VIXCLS": "CBOE Volatility Index (VIX)",
    "DCOILWTICO": "Crude Oil Prices (WTI)",
    "UMCSENT": "University of Michigan Consumer Sentiment",
    "HOUST": "Housing Starts",
}


def download_fred_data(
    series_ids: list[str] | None = None,
    output_dir: str = "data/fred",
    start: str = "2015-01-01",
    end: str = "2024-12-31",
    api_key: str | None = None,
) -> Path:
    """Download FRED economic data series.

    Args:
        series_ids: List of FRED series IDs. Defaults to key macro indicators.
        output_dir: Output directory.
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).
        api_key: FRED API key. Defaults to FRED_API_KEY env var.

    Returns:
        Path to the output directory.
    """
    from fredapi import Fred

    api_key = api_key or os.getenv("FRED_API_KEY")
    if not api_key:
        raise ValueError(
            "FRED_API_KEY not set. Get a free key at "
            "https://fred.stlouisfed.org/docs/api/api_key.html "
            "and add it to your .env file."
        )

    fred = Fred(api_key=api_key)
    series_ids = series_ids or list(DEFAULT_SERIES.keys())
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for series_id in series_ids:
        desc = DEFAULT_SERIES.get(series_id, series_id)
        print(f"Downloading {series_id} ({desc})...")

        try:
            data = fred.get_series(series_id, observation_start=start, observation_end=end)
            if data is not None and not data.empty:
                csv_path = out_path / f"{series_id}.csv"
                data.to_csv(csv_path, header=["value"])
                print(f"  {len(data)} observations -> {csv_path.name}")
            else:
                print(f"  No data found for {series_id}")
        except Exception as e:
            print(f"  Error downloading {series_id}: {e}")

    print(f"\nDone! Saved FRED data to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Download FRED economic data.")
    parser.add_argument(
        "--series", nargs="+", default=None, help="FRED series IDs (default: key macro indicators)"
    )
    parser.add_argument(
        "--output", default="data/fred", help="Output directory"
    )
    parser.add_argument("--start", default="2015-01-01", help="Start date")
    parser.add_argument("--end", default="2024-12-31", help="End date")
    parser.add_argument("--api-key", default=None, help="FRED API key (or set FRED_API_KEY env)")

    args = parser.parse_args()
    download_fred_data(
        series_ids=args.series,
        output_dir=args.output,
        start=args.start,
        end=args.end,
        api_key=args.api_key,
    )


if __name__ == "__main__":
    main()
