#!/usr/bin/env python3
"""Download market data from yfinance.

Fetches OHLCV price history and company fundamentals for given tickers.
Saves price data as CSV and fundamentals as JSON.

Usage:
    # Download for specific tickers
    python scripts/download_market_data.py --tickers AAPL MSFT GOOGL

    # Download with custom date range
    python scripts/download_market_data.py --tickers AAPL --start 2020-01-01 --end 2024-12-31

    # Download for top 50 S&P 500 (default)
    python scripts/download_market_data.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Same default tickers as SEC filings script
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    "LLY", "V", "JPM", "UNH", "XOM", "MA", "JNJ", "PG", "COST", "HD",
    "ABBV", "MRK", "KO", "CRM", "AVGO", "PEP", "CVX", "WMT", "BAC",
    "TMO", "NFLX", "ACN", "ADBE", "AMD", "LIN", "MCD", "CSCO", "ABT",
    "ORCL", "DHR", "TXN", "CMCSA", "PM", "INTC", "WFC", "VZ", "NEE",
    "DIS", "BMY", "UPS", "RTX", "QCOM",
]


def download_market_data(
    tickers: list[str] | None = None,
    output_dir: str = "data/market",
    start: str = "2020-01-01",
    end: str = "2024-12-31",
) -> Path:
    """Download OHLCV data and fundamentals for given tickers.

    Args:
        tickers: List of stock tickers.
        output_dir: Output directory.
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).

    Returns:
        Path to the output directory.
    """
    import yfinance as yf

    tickers = tickers or DEFAULT_TICKERS
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for ticker in tickers:
        print(f"Downloading {ticker}...")
        try:
            stock = yf.Ticker(ticker)

            # Download OHLCV price history
            hist = stock.history(start=start, end=end)
            if not hist.empty:
                csv_path = out_path / f"{ticker}.csv"
                hist.to_csv(csv_path)
                print(f"  Price history: {len(hist)} rows -> {csv_path.name}")
            else:
                print(f"  No price history found for {ticker}")

            # Download company fundamentals
            info = stock.info
            if info:
                # Convert non-serializable values
                clean_info = {}
                for k, v in info.items():
                    try:
                        json.dumps(v)
                        clean_info[k] = v
                    except (TypeError, ValueError):
                        clean_info[k] = str(v)

                json_path = out_path / f"{ticker}_info.json"
                with open(json_path, "w") as f:
                    json.dump(clean_info, f, indent=2)
                print(f"  Fundamentals: {len(clean_info)} fields -> {json_path.name}")

        except Exception as e:
            print(f"  Error downloading {ticker}: {e}")

    print(f"\nDone! Saved market data to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Download market data from yfinance.")
    parser.add_argument(
        "--tickers", nargs="+", default=None, help="Stock tickers (default: top 50 S&P 500)"
    )
    parser.add_argument(
        "--output", default="data/market", help="Output directory"
    )
    parser.add_argument("--start", default="2020-01-01", help="Start date")
    parser.add_argument("--end", default="2024-12-31", help="End date")

    args = parser.parse_args()
    download_market_data(
        tickers=args.tickers,
        output_dir=args.output,
        start=args.start,
        end=args.end,
    )


if __name__ == "__main__":
    main()
