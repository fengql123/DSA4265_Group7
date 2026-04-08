#!/usr/bin/env python3
"""
Pipeline Demo — MainAgent calling sub-agent tools
==================================================

Tests the full flow:
1. User submits a natural language query
2. MainAgent's LLM extracts ticker + date range
3. MainAgent calls 3 sub-agent tools (sentiment, fundamental, risk)
   - Tools may execute in parallel if the LLM issues multiple calls at once
   - Each sub-agent tool runs its own ReAct loop internally (stubs for now)
4. Sentiment stub generates a dummy chart image (tests multimodality)
5. MainAgent synthesizes into InvestmentMemo

Prerequisites:
    pip install -e .
    Set OPENROUTER_API_KEY in .env (or configure another provider)

Run:
    python demo/single_agent_demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def maybe_prepare_query_data(query: str, prepare_data: bool) -> None:
    """Prepare ticker data for a query if requested."""
    if not prepare_data:
        return

    from src.preflight import ensure_ticker_data, extract_ticker_from_query

    ticker = extract_ticker_from_query(query)
    if not ticker:
        print(f"  Could not infer ticker for query: {query}")
        return

    print(f"  Checking indexed RAG data for {ticker}...")
    ensure_ticker_data(ticker)


def run_single_query(query: str, debug: bool = False, prepare_data: bool = False) -> dict:
    """Run a single query through the pipeline."""
    from src.graph import build_graph

    maybe_prepare_query_data(query, prepare_data)
    graph = build_graph(debug=debug)
    return graph.invoke({"query": query, "errors": []})


def slugify_query(query: str) -> str:
    """Create a simple filename-safe slug from a query."""
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in query.strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")[:60] or "query"


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Full pipeline demo.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="Auto-download and ingest ticker data if indexed RAG coverage is missing",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run the parallel multi-query batch demo",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("PIPELINE DEMO — MainAgent + Sub-Agent Tools")
    print("=" * 60)

    # =========================================================================
    # STEP 1: Single query
    # =========================================================================
    query = "Should I invest in NVIDIA stock?"
    print(f"\nStep 1: Running query: '{query}'")
    print("  MainAgent will:")
    print("    - Extract ticker and date range from the query")
    print("    - Call sentiment, fundamental, risk sub-agent tools")
    print("    - Synthesize into InvestmentMemo\n")

    if args.prepare_data:
        print("  Preflight enabled: missing ticker data will be downloaded and ingested first.\n")

    result = run_single_query(query, debug=args.debug, prepare_data=args.prepare_data)

    # =========================================================================
    # STEP 2: Display results
    # =========================================================================
    print("\nStep 2: Results")
    print("-" * 40)

    errors = result.get("errors", [])
    if errors:
        print(f"\n  Errors: {errors}")

    memo = result.get("investment_memo")
    if memo:
        print(f"\n  Ticker: {memo.ticker}")
        print(f"  Recommendation: {memo.recommendation.upper()}")
        print(f"  Confidence: {memo.confidence:.0%}")
        print(f"  Thesis: {memo.thesis}")
        print(f"\n  Sentiment Summary: {memo.sentiment_summary[:150]}...")
        print(f"  Fundamental Summary: {memo.fundamental_summary[:150]}...")
        print(f"  Risk Summary: {memo.risk_summary[:150]}...")

        # Save report
        out_dir = Path("outputs")
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / "demo_report.json"
        with open(out_path, "w") as f:
            json.dump(memo.model_dump(), f, indent=2, default=str)
        print(f"\n  Report saved to {out_path}")

        md_path = out_dir / "demo_report.md"
        md_path.write_text(memo.report_markdown)
        print(f"  Markdown saved to {md_path}")
    else:
        print("\n  No memo generated. Check your API key in .env")
        print("  Make sure OPENROUTER_API_KEY is set")

    # =========================================================================
    # STEP 3: Check artifacts (multimodality test)
    # =========================================================================
    print("\n\nStep 3: Artifacts (multimodality test)")
    print("-" * 40)

    artifacts = result.get("artifacts", [])
    if artifacts:
        print(f"  {len(artifacts)} artifact(s) collected:")
        for a in artifacts:
            print(f"    - {a.path} ({a.artifact_type.value}, {a.description})")
            if a.is_image:
                b64 = a.to_base64()
                print(f"      Base64 size: {len(b64)} chars (multimodal injection works)")
    else:
        print("  No artifacts collected (sentiment stub may not have generated a chart)")

    print("\n\nStep 4: Parallel multi-query")
    print("-" * 40)

    if args.batch:
        queries = [
            "Should I invest in NVIDIA?",
            "Analyze Microsoft stock",
        ]
        print(f"  Running {len(queries)} queries in parallel via graph.batch()...")

        if args.prepare_data:
            for q in queries:
                maybe_prepare_query_data(q, prepare_data=True)

        from src.graph import build_graph

        graph = build_graph(debug=args.debug)
        states = [{"query": q, "errors": []} for q in queries]
        results = graph.batch(states)

        out_dir = Path("outputs")
        out_dir.mkdir(exist_ok=True)

        for q, r in zip(queries, results):
            m = r.get("investment_memo")
            if m:
                print(f"  '{q}' -> {m.ticker}: {m.recommendation.upper()} ({m.confidence:.0%})")

                slug = slugify_query(q)
                batch_json_path = out_dir / f"batch_{slug}.json"
                batch_md_path = out_dir / f"batch_{slug}.md"

                with open(batch_json_path, "w") as f:
                    json.dump(m.model_dump(), f, indent=2, default=str)
                batch_md_path.write_text(m.report_markdown)

                print(f"    Saved JSON: {batch_json_path}")
                print(f"    Saved Markdown: {batch_md_path}")
            else:
                errs = r.get("errors", [])
                print(f"  '{q}' -> FAILED: {errs}")
    else:
        print("  Skipped. Re-run with --batch to include the parallel multi-query test.")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
