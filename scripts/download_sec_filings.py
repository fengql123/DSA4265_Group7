#!/usr/bin/env python3
"""Download SEC 10-K and 10-Q filings for S&P 500 companies.

Uses edgartools to fetch filings from SEC EDGAR. Saves the text content
to data/sec/{ticker}/{filing_type}_{date}.txt.

Usage:
    # Download for default top 50 S&P 500 companies
    python scripts/download_sec_filings.py

    # Download for specific tickers
    python scripts/download_sec_filings.py --tickers AAPL MSFT GOOGL

    # Download with custom year range
    python scripts/download_sec_filings.py --start-year 2020 --end-year 2024

    # Download only 10-K filings
    python scripts/download_sec_filings.py --filing-types 10-K
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

# Top 50 S&P 500 companies by market cap (approximate)
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    "LLY", "V", "JPM", "UNH", "XOM", "MA", "JNJ", "PG", "COST", "HD",
    "ABBV", "MRK", "KO", "CRM", "AVGO", "PEP", "CVX", "WMT", "BAC",
    "TMO", "NFLX", "ACN", "ADBE", "AMD", "LIN", "MCD", "CSCO", "ABT",
    "ORCL", "DHR", "TXN", "CMCSA", "PM", "INTC", "WFC", "VZ", "NEE",
    "DIS", "BMY", "UPS", "RTX", "QCOM",
]


def download_sec_filings(
    tickers: list[str] | None = None,
    output_dir: str = "data/sec",
    filing_types: list[str] | None = None,
    start_year: int = 2020,
    end_year: int = 2024,
    identity: str = "DSA4265 Student dsa4265@example.com",
) -> Path:
    """Download SEC filings for given tickers.

    Args:
        tickers: List of stock tickers. Defaults to top 50 S&P 500.
        output_dir: Base output directory.
        filing_types: Filing types to download (default: ["10-K", "10-Q"]).
        start_year: Start year for filing search.
        end_year: End year for filing search.
        identity: Identity string for SEC EDGAR (email required by SEC).

    Returns:
        Path to the output directory.
    """
    from edgar import set_identity, Company

    tickers = tickers or DEFAULT_TICKERS
    filing_types = filing_types or ["10-K", "10-Q"]
    out_path = Path(output_dir)

    set_identity(identity)

    total_downloaded = 0

    for ticker in tickers:
        ticker_dir = out_path / ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Processing {ticker}...")

        try:
            company = Company(ticker)
        except Exception as e:
            print(f"  Error looking up {ticker}: {e}")
            continue

        for filing_type in filing_types:
            try:
                filings = company.get_filings(form=filing_type)

                for filing in filings:
                    # Filter by year
                    filing_date = str(filing.filing_date)
                    year = int(filing_date[:4])
                    if year < start_year or year > end_year:
                        continue

                    out_file = ticker_dir / f"{filing_type}_{filing_date}.txt"
                    if out_file.exists():
                        print(f"  Skipping {out_file.name} (already exists)")
                        continue

                    try:
                        # Get the filing text content
                        text = filing.text()
                        if text:
                            out_file.write_text(text, encoding="utf-8")
                            total_downloaded += 1
                            print(f"  Downloaded {out_file.name} ({len(text)} chars)")
                        else:
                            print(f"  Skipping {filing_type} {filing_date} (no text content)")
                    except Exception as e:
                        print(f"  Error downloading {filing_type} {filing_date}: {e}")

                    # Respect SEC rate limits (10 req/sec)
                    time.sleep(0.15)

            except Exception as e:
                print(f"  Error fetching {filing_type} filings for {ticker}: {e}")

    print(f"\nDone! Downloaded {total_downloaded} filings to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Download SEC filings from EDGAR.")
    parser.add_argument(
        "--tickers", nargs="+", default=None, help="Stock tickers (default: top 50 S&P 500)"
    )
    parser.add_argument(
        "--output", default="data/sec", help="Output directory (default: data/sec)"
    )
    parser.add_argument(
        "--filing-types",
        nargs="+",
        default=None,
        help="Filing types (default: 10-K 10-Q)",
    )
    parser.add_argument("--start-year", type=int, default=2020, help="Start year")
    parser.add_argument("--end-year", type=int, default=2024, help="End year")
    parser.add_argument(
        "--identity",
        default="DSA4265 Student dsa4265@example.com",
        help="SEC EDGAR identity (must include email)",
    )

    args = parser.parse_args()

    download_sec_filings(
        tickers=args.tickers,
        output_dir=args.output,
        filing_types=args.filing_types,
        start_year=args.start_year,
        end_year=args.end_year,
        identity=args.identity,
    )


if __name__ == "__main__":
    main()
