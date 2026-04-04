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


def run_single_query(query: str, debug: bool = False) -> dict:
    """Run a single query through the pipeline."""
    from src.graph import build_graph

    graph = build_graph(debug=debug)
    return graph.invoke({"query": query, "errors": []})


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Full pipeline demo.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    print("=" * 60)
    print("PIPELINE DEMO — MainAgent + Sub-Agent Tools")
    print("=" * 60)

    # =========================================================================
    # STEP 1: Single query
    # =========================================================================
    query = "Should I invest in Apple stock?"
    print(f"\nStep 1: Running query: '{query}'")
    print("  MainAgent will:")
    print("    - Extract ticker (AAPL) and date range from the query")
    print("    - Call sentiment, fundamental, risk sub-agent tools")
    print("    - Synthesize into InvestmentMemo\n")

    result = run_single_query(query, debug=args.debug)

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

    # =========================================================================
    # STEP 4: Parallel multi-query
    # =========================================================================
    print("\n\nStep 4: Parallel multi-query")
    print("-" * 40)

    queries = [
        "Should I invest in Apple?",
        "Analyze Tesla stock",
    ]
    print(f"  Running {len(queries)} queries in parallel via graph.batch()...")

    from src.graph import build_graph

    graph = build_graph(debug=args.debug)
    states = [{"query": q, "errors": []} for q in queries]
    results = graph.batch(states)

    for q, r in zip(queries, results):
        m = r.get("investment_memo")
        if m:
            print(f"  '{q}' -> {m.ticker}: {m.recommendation.upper()} ({m.confidence:.0%})")
        else:
            errs = r.get("errors", [])
            print(f"  '{q}' -> FAILED: {errs}")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
