"""CLI entry point for running the analysis pipeline.

Accepts one or more natural language queries. Multiple queries run in parallel
via LangGraph's native batch() support.

Usage:
    # Single query
    python -m src.runner "Should I invest in Apple?"

    # Multiple queries in parallel (uses graph.batch())
    python -m src.runner "Should I invest in Apple?" "Analyze Tesla" "Is Microsoft overvalued?"

    # With custom output directory
    python -m src.runner --output-dir results "Analyze NVIDIA"
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _save_result(result: dict, query: str, output_dir: Path) -> None:
    """Save a pipeline result to disk."""
    memo = result.get("investment_memo")
    if not memo:
        print(f"\n  No memo generated for: {query}")
        errors = result.get("errors", [])
        if errors:
            print(f"  Errors: {errors}")
        return

    slug = re.sub(r"[^a-zA-Z0-9]", "_", query[:40]).strip("_").lower()

    md_path = output_dir / f"{slug}_report.md"
    md_path.write_text(memo.report_markdown)

    json_path = output_dir / f"{slug}_report.json"
    with open(json_path, "w") as f:
        json.dump(memo.model_dump(), f, indent=2, default=str)

    print(f"\n  Query: {query}")
    print(f"  Ticker: {memo.ticker}")
    print(f"  Recommendation: {memo.recommendation.upper()} (confidence: {memo.confidence:.0%})")
    print(f"  Thesis: {memo.thesis}")
    print(f"  Reports: {md_path}, {json_path}")

    artifacts = result.get("artifacts", [])
    if artifacts:
        print(f"  Artifacts: {len(artifacts)} files")
        for a in artifacts:
            print(f"    - {a.path} ({a.description})")


def main():
    parser = argparse.ArgumentParser(
        description="Run the investment analysis pipeline."
    )
    parser.add_argument(
        "queries",
        nargs="+",
        help="Natural language investment queries",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Output directory for reports (default: outputs)",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from src.graph import build_graph
    from src.preflight import ensure_ticker_data, extract_ticker_from_query

    graph = build_graph()

    print("=" * 60)
    print(f"Running {len(args.queries)} query(ies)")
    print("=" * 60)

    for query in args.queries:
        ticker = extract_ticker_from_query(query)
        if ticker:
            ensure_ticker_data(ticker)

    if len(args.queries) == 1:
        result = graph.invoke({"query": args.queries[0], "errors": []})
        _save_result(result, args.queries[0], output_dir)
    else:
        # Parallel via LangGraph's native batch()
        states = [{"query": q, "errors": []} for q in args.queries]
        results = graph.batch(states)

        for query, result in zip(args.queries, results):
            _save_result(result, query, output_dir)

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
