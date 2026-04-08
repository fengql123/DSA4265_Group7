#!/usr/bin/env python3
"""
Single Agent Demo — Test each sub-agent individually
=====================================================

Runs each sub-agent directly (not through MainAgent) to verify:
- Tools are called correctly (get_market_data, get_fred_data, rag_retrieve)
- Artifacts are produced (sentiment chart)
- Structured reports are returned
- The ReAct loop works end-to-end

Prerequisites:
    pip install -e .
    python scripts/download_demo_data.py --ticker AAPL
    python scripts/ingest_demo.py --ticker AAPL
    Set OPENROUTER_API_KEY in .env

Run:
    python demo/single_agent_demo.py
    python demo/single_agent_demo.py --agent fundamental
    python demo/single_agent_demo.py --agent sentiment
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def run_agent(agent_name: str, ticker: str, date: str, lookback_days: int, debug: bool = False) -> dict:
    """Run a single sub-agent directly and return its result."""
    from src.agents.sentiment_agent import SentimentAgent
    from src.agents.fundamental_agent import FundamentalAgent
    from src.agents.technical_agent import TechnicalAgent
    from src.agents.risk_agent import RiskAgent

    agents = {
        "sentiment": SentimentAgent,
        "fundamental": FundamentalAgent,
        "technical": TechnicalAgent,
        "risk": RiskAgent,
    }

    if agent_name not in agents:
        print(f"Unknown agent: {agent_name}. Choose from: {list(agents.keys())}")
        sys.exit(1)

    agent = agents[agent_name]()
    agent.debug = debug
    state = {
        "ticker": ticker,
        "analysis_date": date,
        "lookback_days": lookback_days,
        "errors": [],
    }

    print(f"\n  Agent: {agent}")
    print(f"  State: ticker={ticker}, date={date}, lookback_days={lookback_days}")
    print(f"  Tools: {agent.tool_names or '(none — mock)'}")
    print()

    result = agent.run(state)
    return result


def display_and_save_result(agent_name: str, result: dict):
    """Display the result and save report + artifacts to outputs/."""
    import shutil

    out_dir = Path("outputs") / "single_agent" / agent_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check errors
    errors = result.get("errors", [])
    if errors:
        print(f"  Errors: {errors}")

    # Find the report (key varies by agent)
    report = None
    report_key = None
    for key, val in result.items():
        if key not in ("errors", "artifacts") and val is not None:
            report = val
            report_key = key
            break

    if report:
        print(f"  Output field: {report_key}")
        print(f"  Report type: {type(report).__name__}")
        print()

        # Print all fields
        if hasattr(report, "model_dump"):
            data = report.model_dump()
            for field_name, value in data.items():
                if isinstance(value, str) and len(value) > 150:
                    value = value[:150] + "..."
                elif isinstance(value, list) and len(value) > 3:
                    value = value[:3] + ["..."]
                print(f"  {field_name}: {value}")

            # Save report as JSON
            json_path = out_dir / f"{agent_name}_report.json"
            with open(json_path, "w") as f:
                json.dump(report.model_dump(), f, indent=2, default=str)
            print(f"\n  Report saved: {json_path}")
        else:
            print(f"  {report}")
    else:
        print("  No report generated.")

    # Artifacts
    artifacts = result.get("artifacts", [])
    print(f"\n  Artifacts: {len(artifacts)}")
    for a in artifacts:
        print(f"    - {a.path} ({a.artifact_type.value}, {a.description})")
        if a.is_image:
            try:
                b64 = a.to_base64()
                print(f"      Base64 size: {len(b64)} chars")
            except FileNotFoundError:
                print(f"      (file not found)")

        # Copy artifact to the agent's output dir
        src_path = Path(a.path)
        if src_path.exists():
            dst_path = out_dir / src_path.name
            shutil.copy2(src_path, dst_path)
            print(f"      Saved: {dst_path}")


def main():
    parser = argparse.ArgumentParser(description="Test individual sub-agents.")
    parser.add_argument(
        "--agent",
        choices=["sentiment", "fundamental", "technical", "risk", "all"],
        default="all",
        help="Which agent to test (default: all)",
    )
    parser.add_argument("--ticker", default="AAPL", help="Ticker symbol (default: AAPL)")
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Analysis date YYYY-MM-DD (default: today)",
    )
    parser.add_argument("--lookback-days", type=int, default=180, help="Lookback period in days (default: 180)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging for agents")
    args = parser.parse_args()

    agents_to_test = (
        ["sentiment", "fundamental", "technical", "risk"]
        if args.agent == "all"
        else [args.agent]
    )

    print("=" * 60)
    print("SINGLE AGENT DEMO — Testing sub-agents individually")
    print("=" * 60)

    for agent_name in agents_to_test:
        print(f"\n{'-' * 60}")
        print(f"Testing: {agent_name}")
        print("-" * 60)

        result = run_agent(agent_name, ticker=args.ticker.upper(), date=args.date, lookback_days=args.lookback_days, debug=args.debug)
        display_and_save_result(agent_name, result)

    print(f"\n{'=' * 60}")
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
