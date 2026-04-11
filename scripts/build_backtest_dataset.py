#!/usr/bin/env python3
"""Phase 1 driver: build the historical data snapshot for the backtest.

For each ticker in the universe, this script:

1. Downloads OHLCV CSV + info.json into `data/market/{TICKER}.csv`
   (plus SPY as a benchmark) over a wide date range (2020-01-01 →
   2025-10-31) so we have the lookback needed for 200DMAs and enough
   forward data for 12m forward returns.
2. Downloads full SEC 10-K / 10-Q history into `data/sec/{TICKER}/`.
3. Downloads demo news + earnings data into `data/demo/{news,earnings}`.
4. Ingests SEC filings into the `sec_filings` ChromaDB collection with
   **leak-safe `date` metadata** extracted from the filename.
5. Ingests the demo news + earnings into the `news` / `earnings`
   collections via the existing `ingest_demo.py` helpers.
6. Writes `data/backtest/universe.yaml` with the quarterly as-of dates
   (2022-01-03 through 2024-10-01, 12 dates per ticker).

Usage:
    python scripts/build_backtest_dataset.py --tickers AAPL JPM JNJ KO NFLX

All steps are idempotent — re-running the script skips work that has
already been completed.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Monthly as-of dates from 2024-10-01 (strictly after gpt-5's Sep 30,
# 2024 training cutoff) through 2025-10-01 (the last month where we
# still have 6 months of forward price data available at report
# authoring time). Exact dates don't matter because the historical
# adapter clips to the latest trading day on or before each request.
DEFAULT_AS_OF_DATES = [
    "2024-10-01", "2024-11-01", "2024-12-02",
    "2025-01-02", "2025-02-03", "2025-03-03",
    "2025-04-01", "2025-05-01", "2025-06-02",
    "2025-07-01", "2025-08-01", "2025-09-02",
    "2025-10-01",
]

DEFAULT_TICKERS = ["AAPL", "JPM", "JNJ", "KO", "NFLX"]


def _extract_date_from_filename(path: Path) -> str:
    m = re.search(r"(\d{4}-\d{2}-\d{2})", path.name)
    return m.group(1) if m else ""


def _extract_filing_type(path: Path) -> str:
    up = path.name.upper()
    if "10-K" in up or "10K" in up:
        return "10-K"
    if "10-Q" in up or "10Q" in up:
        return "10-Q"
    return "sec_filing"


def step_market_data(tickers: list[str], start: str, end: str) -> None:
    """Step 1: download OHLCV + info.json per ticker (idempotent)."""
    from scripts.download_market_data import download_market_data

    tickers_plus_spy = list(tickers) + ["SPY"]
    missing = [t for t in tickers_plus_spy if not (ROOT_DIR / "data" / "market" / f"{t}.csv").exists()]
    if not missing:
        print(f"[market] all CSVs present for {tickers_plus_spy}, skipping download")
        return

    print(f"[market] downloading {missing} over {start} .. {end}")
    download_market_data(tickers=missing, output_dir="data/market", start=start, end=end)


def step_sec_filings(tickers: list[str], start_year: int, end_year: int) -> None:
    """Step 2: download 10-K / 10-Q history per ticker (idempotent)."""
    from scripts.download_sec_filings import download_sec_filings

    print(f"[sec] downloading 10-K / 10-Q for {tickers} years {start_year}-{end_year}")
    download_sec_filings(
        tickers=tickers,
        output_dir="data/sec",
        filing_types=["10-K", "10-Q"],
        start_year=start_year,
        end_year=end_year,
    )


def step_demo_news_earnings(tickers: list[str]) -> None:
    """Step 3: download news + earnings via download_demo_data.py (idempotent)."""
    for ticker in tickers:
        demo_news = ROOT_DIR / "data" / "demo" / "news" / ticker / "news.jsonl"
        demo_earn = ROOT_DIR / "data" / "demo" / "earnings" / ticker / "transcripts.jsonl"
        if demo_news.exists() and demo_earn.exists():
            print(f"[demo] {ticker} already has news+earnings, skipping")
            continue
        print(f"[demo] downloading demo data for {ticker}")
        subprocess.run(
            [sys.executable, "scripts/download_demo_data.py", "--ticker", ticker],
            cwd=ROOT_DIR,
            check=True,
        )


def step_ingest_sec(tickers: list[str]) -> int:
    """Step 4: ingest SEC filings into the sec_filings collection with date metadata.

    Uses the project's index_files() directly with a metadata_fn that
    pulls `date` and `filing_type` out of the filename.
    """
    from src.rag.indexer import index_files

    total = 0
    for ticker in tickers:
        sec_dir = ROOT_DIR / "data" / "sec" / ticker
        if not sec_dir.exists():
            print(f"[ingest-sec] no data/sec/{ticker}, skipping")
            continue
        files = sorted(sec_dir.glob("*.txt"))
        if not files:
            print(f"[ingest-sec] {ticker}: no .txt files, skipping")
            continue

        print(f"[ingest-sec] {ticker}: ingesting {len(files)} filing(s)")

        def metadata_fn(path: Path, ticker=ticker) -> dict:
            date_str = _extract_date_from_filename(path)
            return {
                "ticker": ticker,
                "doc_type": "sec_filing",
                "source_file": path.name,
                "date": date_str,
                "filing_type": _extract_filing_type(path),
            }

        chunks = index_files(
            file_paths=files,
            collection_name="sec_filings",
            metadata_fn=metadata_fn,
        )
        print(f"[ingest-sec] {ticker}: {chunks} chunks into 'sec_filings'")
        total += chunks
    return total


def step_ingest_demo(tickers: list[str]) -> None:
    """Step 5: ingest demo news + earnings + 1 recent SEC filing via ingest_demo.py."""
    for ticker in tickers:
        print(f"[ingest-demo] {ticker}")
        subprocess.run(
            [sys.executable, "scripts/ingest_demo.py", "--ticker", ticker],
            cwd=ROOT_DIR,
            check=True,
        )


def step_write_universe(tickers: list[str], as_of_dates: list[str], output: Path) -> None:
    """Step 6: write data/backtest/universe.yaml."""
    output.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "tickers": list(tickers),
        "as_of_dates": list(as_of_dates),
        "pairs": [{"ticker": t, "as_of_date": d} for t in tickers for d in as_of_dates],
    }
    with output.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    print(f"[universe] wrote {len(data['pairs'])} (ticker, as_of_date) pairs to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the backtest dataset.")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS, help="Ticker universe")
    parser.add_argument("--market-start", default="2023-01-01", help="Market-data start date (need enough pre-window lead-in for 200DMA etc.)")
    parser.add_argument("--market-end", default="2026-04-12", help="Market-data end date (need enough post-window buffer for forward returns)")
    parser.add_argument("--sec-start-year", type=int, default=2023, help="SEC filings start year")
    parser.add_argument("--sec-end-year", type=int, default=2025, help="SEC filings end year")
    parser.add_argument(
        "--universe-out",
        default="data/backtest/universe.yaml",
        help="Path for the universe manifest",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        default=[],
        choices=["market", "sec", "demo", "ingest_sec", "ingest_demo", "universe"],
        help="Optionally skip some steps (for iteration).",
    )
    args = parser.parse_args()

    tickers = [t.upper() for t in args.tickers]
    print(f"Universe: {tickers}")
    print(f"Quarterly as-of dates: {DEFAULT_AS_OF_DATES}")

    if "market" not in args.skip:
        step_market_data(tickers, args.market_start, args.market_end)
    if "sec" not in args.skip:
        step_sec_filings(tickers, args.sec_start_year, args.sec_end_year)
    if "demo" not in args.skip:
        step_demo_news_earnings(tickers)
    if "ingest_sec" not in args.skip:
        step_ingest_sec(tickers)
    if "ingest_demo" not in args.skip:
        step_ingest_demo(tickers)
    if "universe" not in args.skip:
        step_write_universe(tickers, DEFAULT_AS_OF_DATES, ROOT_DIR / args.universe_out)

    print("\nDone.")


if __name__ == "__main__":
    main()
