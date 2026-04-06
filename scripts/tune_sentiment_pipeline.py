#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
from pathlib import Path


def run_cmd(cmd: list[str], env: dict | None = None) -> tuple[int, str, str]:
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return result.returncode, result.stdout, result.stderr


def load_report(report_path: Path) -> dict | None:
    if not report_path.exists():
        return None
    try:
        return json.loads(report_path.read_text())
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Tune sentiment pipeline hyperparameters.")
    parser.add_argument("--ticker", default="MSFT")
    parser.add_argument("--query", default="Analyze Microsoft stock")
    parser.add_argument("--model-path", default="", help="Optional fine-tuned sentiment model path")
    args = parser.parse_args()

    # Hyperparameter grid
    news_limits = [10, 25, 50]
    lookback_days_list = [90, 180, 365]
    top_k_news_list = [5, 10]

    results = []
    out_dir = Path("outputs/pipeline_tuning")
    out_dir.mkdir(parents=True, exist_ok=True)

    for news_limit, lookback_days, top_k_news in itertools.product(
        news_limits,
        lookback_days_list,
        top_k_news_list,
    ):
        print("=" * 80)
        print(
            f"Running: news_limit={news_limit}, "
            f"lookback_days={lookback_days}, "
            f"top_k_news={top_k_news}"
        )

        env = os.environ.copy()

        # Use fine-tuned model if provided
        if args.model_path:
            env["SENTIMENT_MODEL_PATH"] = args.model_path

        # Pass pipeline settings through environment variables
        env["PIPELINE_NEWS_LIMIT"] = str(news_limit)
        env["PIPELINE_LOOKBACK_DAYS"] = str(lookback_days)
        env["PIPELINE_TOP_K_NEWS"] = str(top_k_news)

        # Step 1: download data
        rc1, out1, err1 = run_cmd(
            ["python", "scripts/download_demo_data.py", "--ticker", args.ticker],
            env=env,
        )

        # Step 2: ingest
        rc2, out2, err2 = run_cmd(
            ["python", "scripts/ingest_demo.py", "--ticker", args.ticker],
            env=env,
        )

        # Step 3: run sentiment agent only
        rc3, out3, err3 = run_cmd(
            ["python", "demo/single_agent_demo.py", "--agent", "sentiment", "--ticker", args.ticker],
            env=env,
        )

        # Try to read the generated report if it exists
        report_path = Path(
            f"outputs/single_agent/sentiment/{args.ticker.lower()}_sentiment_report.json"
        )
        report = load_report(report_path)

        summary = {
            "news_limit": news_limit,
            "lookback_days": lookback_days,
            "top_k_news": top_k_news,
            "download_ok": rc1 == 0,
            "ingest_ok": rc2 == 0,
            "run_ok": rc3 == 0,
            "report_found": report is not None,
            "overall_sentiment": report.get("overall_sentiment") if report else None,
            "sentiment_score": report.get("sentiment_score") if report else None,
            "n_themes": len(report.get("key_themes", [])) if report else None,
            "n_evidence": len(report.get("evidence", [])) if report else None,
        }

        results.append(summary)

        print(json.dumps(summary, indent=2))

        # Optional: print errors if something fails
        if rc1 != 0:
            print("\n[download_demo_data.py stderr]")
            print(err1)
        if rc2 != 0:
            print("\n[ingest_demo.py stderr]")
            print(err2)
        if rc3 != 0:
            print("\n[single_agent_demo.py stderr]")
            print(err3)

    # Save all results
    results_path = out_dir / f"{args.ticker.lower()}_sentiment_tuning_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved results to {results_path}")


if __name__ == "__main__":
    main()