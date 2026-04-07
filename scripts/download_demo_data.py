#!/usr/bin/env python3
"""Download a small dataset for running the demo.

Downloads real financial data for a single ticker (AAPL by default):
  1. SEC 10-K filing text from SEC EDGAR
  2. Earnings call transcript from HuggingFace
  3. Recent news from yfinance
  4. Market data from yfinance

Usage:
    python scripts/download_demo_data.py
    python scripts/download_demo_data.py --ticker MSFT
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEMO_DIR = Path("data/demo")


def download_sec_filing(ticker: str) -> list[Path]:
    """Download the most recent 10-K filing for a ticker from SEC EDGAR."""
    print(f"\n[1/4] Downloading SEC 10-K filing for {ticker}...")

    from edgar import set_identity, Company

    set_identity("DSA4265 Student dsa4265@nus.edu.sg")

    out_dir = DEMO_DIR / "sec" / ticker
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    try:
        company = Company(ticker)
        filings = company.get_filings(form="10-K")

        # Get the most recent 10-K
        for filing in filings:
            filing_date = str(filing.filing_date)
            out_file = out_dir / f"10-K_{filing_date}.txt"

            if out_file.exists():
                print(f"  Already exists: {out_file.name}")
                saved.append(out_file)
                break

            try:
                text = filing.text()
                if text and len(text) > 1000:
                    out_file.write_text(text, encoding="utf-8")
                    print(f"  Saved {out_file.name} ({len(text):,} chars)")
                    saved.append(out_file)
                    break
            except Exception as e:
                print(f"  Error getting text: {e}")

            time.sleep(0.2)

    except Exception as e:
        print(f"  Error: {e}")

    if not saved:
        print("  WARNING: Could not download SEC filing. Demo will skip SEC data.")

    return saved


def download_earnings(ticker: str) -> Path | None:
    """Download earnings call transcripts from HuggingFace."""
    print(f"\n[2/4] Downloading earnings transcripts for {ticker}...")

    out_dir = DEMO_DIR / "earnings" / ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "transcripts.jsonl"

    if out_file.exists():
        print(f"  Already exists: {out_file.name}")
        return out_file

    try:
        from datasets import load_dataset

        ds = load_dataset(
            "lamini/earnings-calls-qa",
            split="train",
            streaming=True,
        )

        ticker = ticker.upper()

        rows = []
        seen = 0
        for row in ds:
            seen += 1

            row_ticker = str(row.get("ticker", "")).upper().strip()

            # Strict filter: only keep rows whose ticker exactly matches
            if row_ticker == ticker:
                rows.append(row)

            if len(rows) >= 50:
                break
            if seen >= 50000:
                break
            if seen % 10000 == 0:
                print(f"  Scanned {seen} rows, found {len(rows)} matches...")

        if rows:
            with open(out_file, "w", encoding="utf-8") as f:
                for row in rows:
                    clean = {
                        k: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v
                        for k, v in row.items()
                    }
                    f.write(json.dumps(clean) + "\n")
            print(f"  Saved {len(rows)} transcript entries -> {out_file.name}")
            return out_file
        else:
            print(f"  No transcript entries found for {ticker}")
            return None

    except Exception as e:
        print(f"  Error: {e}")
        return None


def download_market_data(ticker: str) -> Path | None:
    """Download OHLCV price data and fundamentals from yfinance."""
    print(f"\n[4/4] Downloading market data for {ticker}...")

    out_dir = DEMO_DIR / "market"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{ticker}.csv"
    json_path = out_dir / f"{ticker}_info.json"

    if csv_path.exists() and json_path.exists():
        print(f"  Already exists: {csv_path.name}, {json_path.name}")
        return csv_path

    try:
        import yfinance as yf

        stock = yf.Ticker(ticker)

        # Price history (1 year)
        hist = stock.history(period="1y")
        if not hist.empty:
            hist.to_csv(csv_path)
            print(f"  Price history: {len(hist)} rows -> {csv_path.name}")

        # Fundamentals
        info = stock.info
        if info:
            clean_info = {}
            for k, v in info.items():
                try:
                    json.dumps(v)
                    clean_info[k] = v
                except (TypeError, ValueError):
                    clean_info[k] = str(v)

            with open(json_path, "w") as f:
                json.dump(clean_info, f, indent=2)
            print(f"  Fundamentals: {len(clean_info)} fields -> {json_path.name}")

        return csv_path

    except Exception as e:
        print(f"  Error: {e}")
        return None
    
def download_news(ticker: str) -> Path | None:
    """Download recent news for a ticker using yfinance and normalize fields."""
    print(f"\n[3/4] Downloading news for {ticker}...")

    out_dir = DEMO_DIR / "news" / ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "news.jsonl"

    if out_file.exists():
        print(f"  Already exists: {out_file.name}")
        return out_file

    try:
        import yfinance as yf

        def _pick(*values):
            for v in values:
                if v not in (None, "", [], {}):
                    return v
            return ""

        def _normalize_news_item(item: dict, ticker: str) -> dict:
            # Some yfinance news payloads are flat, others keep the useful fields
            # inside nested objects such as content/provider/canonicalUrl/clickThroughUrl.
            content = item.get("content")
            if not isinstance(content, dict):
                content = {}

            provider = content.get("provider")
            if not isinstance(provider, dict):
                provider = {}

            canonical = content.get("canonicalUrl")
            if not isinstance(canonical, dict):
                canonical = {}

            clickthrough = content.get("clickThroughUrl")
            if not isinstance(clickthrough, dict):
                clickthrough = {}

            title = _pick(
                item.get("title"),
                content.get("title"),
            )

            summary = _pick(
                item.get("summary"),
                content.get("summary"),
                content.get("description"),
            )

            source = _pick(
                item.get("publisher"),
                item.get("source"),
                provider.get("displayName"),
                provider.get("sourceId"),
            )

            date = _pick(
                item.get("providerPublishTime"),
                item.get("pubDate"),
                content.get("pubDate"),
                content.get("displayTime"),
            )

            url = _pick(
                item.get("link"),
                item.get("url"),
                canonical.get("url"),
                clickthrough.get("url"),
            )

            normalized = {
                "ticker": ticker,
                "title": str(title),
                "summary": str(summary),
                "source": str(source),
                "date": str(date),
                "url": str(url),
            }

            # Keep useful optional metadata when available
            content_type = _pick(item.get("contentType"), content.get("contentType"))
            if content_type:
                normalized["content_type"] = str(content_type)

            item_id = _pick(item.get("id"), content.get("id"))
            if item_id:
                normalized["id"] = str(item_id)

            # Optional: keep raw payload for debugging / future schema updates
            normalized["raw"] = item

            return normalized

        stock = yf.Ticker(ticker)
        news_items = getattr(stock, "news", []) or []

        news_limit = int(os.getenv("PIPELINE_NEWS_LIMIT", "30"))
        cleaned = []

        for item in news_items[:news_limit]:
            if not isinstance(item, dict):
                continue

            row = _normalize_news_item(item, ticker)

            # Skip useless rows that still have no real text
            if not row["title"] and not row["summary"]:
                continue

            cleaned.append(row)

        if not cleaned:
            print(f"  No news found for {ticker}")
            return None

        with open(out_file, "w", encoding="utf-8") as f:
            for row in cleaned:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"  Saved {len(cleaned)} news entries -> {out_file.name}")
        return out_file

    except Exception as e:
        print(f"  Error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Download demo data for a single ticker.")
    parser.add_argument("--ticker", default="AAPL", help="Ticker symbol (default: AAPL)")
    args = parser.parse_args()

    ticker = args.ticker.upper()

    print("=" * 60)
    print(f"Downloading demo data for {ticker}")
    print(f"Output directory: {DEMO_DIR}")
    print("=" * 60)

    sec_files = download_sec_filing(ticker)
    earnings_file = download_earnings(ticker)
    news_file = download_news(ticker)
    market_file = download_market_data(ticker)

    print("\n" + "=" * 60)
    print("Download complete! Files:")
    for f in sec_files:
        print(f"  SEC:      {f}")
    if earnings_file:
        print(f"  Earnings: {earnings_file}")
    if news_file:
        print(f"  News:     {news_file}")
    if market_file:
        print(f"  Market:   {market_file}")
        print(f"  Info:     {market_file.with_name(f'{ticker}_info.json')}")

    print(f"\nNext: python scripts/ingest_demo.py --ticker {ticker}")
    print("=" * 60)


if __name__ == "__main__":
    main()
