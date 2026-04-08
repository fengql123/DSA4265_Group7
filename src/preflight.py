"""Preflight helpers for ensuring ticker data is available before analysis."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from src.rag.retriever import retrieve


ROOT_DIR = Path(__file__).resolve().parent.parent


def extract_ticker_from_query(query: str) -> str | None:
    """Heuristically extract a ticker from a user query."""
    upper_query = query.upper()

    # Common company-name shortcuts for demo usage.
    aliases = {
        "APPLE": "AAPL",
        "TESLA": "TSLA",
        "MICROSOFT": "MSFT",
        "NVIDIA": "NVDA",
        "GOOGLE": "GOOGL",
        "ALPHABET": "GOOGL",
        "AMAZON": "AMZN",
        "META": "META",
    }
    for name, ticker in aliases.items():
        if name in upper_query:
            return ticker

    # Fallback: look for an all-caps ticker-like token.
    match = re.search(r"\b[A-Z]{1,5}\b", upper_query)
    if match:
        return match.group(0)

    return None


def ticker_has_rag_data(ticker: str) -> bool:
    """Return True if the ticker appears to have indexed RAG documents."""
    ticker = ticker.upper()

    checks = [
        ("sec_filings", f"{ticker} risk factors"),
        ("earnings", f"{ticker} earnings call"),
        ("news", f"{ticker} company news"),
    ]

    for collection_name, query in checks:
        try:
            chunks = retrieve(
                query=query,
                collection_names=[collection_name],
                metadata_filter={"ticker": ticker},
                top_k=1,
            )
        except Exception:
            continue
        if chunks:
            return True

    return False


def ensure_ticker_data(ticker: str) -> bool:
    """Download and ingest demo data for a ticker if no indexed RAG data is found.

    Returns True if existing data was found or preparation succeeded.
    Returns False if preparation failed.
    """
    ticker = ticker.upper()

    if ticker_has_rag_data(ticker):
        return True

    print(f"\n  No indexed RAG data found for {ticker}. Downloading and ingesting demo data...")

    commands = [
        [sys.executable, "scripts/download_demo_data.py", "--ticker", ticker],
        [sys.executable, "scripts/ingest_demo.py", "--ticker", ticker],
    ]

    for cmd in commands:
        try:
            subprocess.run(cmd, cwd=ROOT_DIR, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"  Preflight failed while running: {' '.join(cmd)}")
            print(f"  Exit code: {exc.returncode}")
            return False

    return ticker_has_rag_data(ticker)
