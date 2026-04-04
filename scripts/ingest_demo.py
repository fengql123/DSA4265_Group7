#!/usr/bin/env python3
"""Ingest demo data into ChromaDB.

Run this ONCE after downloading demo data. It ingests SEC filings and
earnings transcripts into ChromaDB collections so that rag_retrieve
can find them during the demo.

Prerequisites:
    python scripts/download_demo_data.py --ticker AAPL

Usage:
    python scripts/ingest_demo.py
    python scripts/ingest_demo.py --ticker MSFT
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEMO_DIR = Path("data/demo")


def ingest_sec_filings(ticker: str) -> int:
    """Ingest SEC filing text files into ChromaDB."""
    from src.rag.indexer import index_files

    sec_dir = DEMO_DIR / "sec" / ticker
    if not sec_dir.exists():
        print(f"  No SEC data found at {sec_dir}")
        return 0

    files = sorted(sec_dir.glob("*.txt"))
    if not files:
        print(f"  No .txt files in {sec_dir}")
        return 0

    print(f"  Found {len(files)} SEC filing(s)")

    def metadata_fn(path: Path) -> dict:
        return {
            "ticker": ticker,
            "doc_type": "sec_filing",
            "source_file": path.name,
        }

    return index_files(
        file_paths=files,
        collection_name="sec_filings",
        metadata_fn=metadata_fn,
    )


def ingest_earnings(ticker: str) -> int:
    """Ingest earnings transcript JSONL files into ChromaDB."""
    from src.rag.indexer import index_documents

    earnings_dir = DEMO_DIR / "earnings" / ticker
    if not earnings_dir.exists():
        print(f"  No earnings data found at {earnings_dir}")
        return 0

    texts = []
    for f in sorted(earnings_dir.glob("*.*")):
        if f.suffix == ".jsonl":
            for line in f.read_text().strip().splitlines():
                row = json.loads(line)
                parts = [str(v) for v in row.values() if isinstance(v, str) and len(str(v)) > 20]
                if parts:
                    texts.append("\n".join(parts))
        else:
            text = f.read_text(encoding="utf-8")
            if text.strip():
                texts.append(text)

    if not texts:
        print(f"  No transcript text found in {earnings_dir}")
        return 0

    print(f"  Found {len(texts)} transcript entries")

    metadata = [{"ticker": ticker, "doc_type": "earnings_transcript"} for _ in texts]
    return index_documents(
        texts=texts,
        collection_name="earnings",
        metadata=metadata,
    )


def main():
    parser = argparse.ArgumentParser(description="Ingest demo data into ChromaDB.")
    parser.add_argument("--ticker", default="AAPL", help="Ticker symbol (default: AAPL)")
    args = parser.parse_args()
    ticker = args.ticker.upper()

    print("=" * 60)
    print(f"Ingesting demo data for {ticker} into ChromaDB")
    print("=" * 60)

    print(f"\n[1/2] SEC filings...")
    sec_chunks = ingest_sec_filings(ticker)

    print(f"\n[2/2] Earnings transcripts...")
    earnings_chunks = ingest_earnings(ticker)

    print(f"\n{'=' * 60}")
    print(f"Done! {sec_chunks} SEC chunks + {earnings_chunks} earnings chunks")
    print(f"Collections: 'sec_filings', 'earnings'")
    print(f"\nNow run: python demo/single_agent_demo.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
