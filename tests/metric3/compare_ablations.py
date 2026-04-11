#!/usr/bin/env python3
"""Phase 7 summary: compare baseline + 4 ablation signal sets.

For each (baseline + no_sentiment + no_fundamental + no_technical + no_risk)
it runs the Phase 4/5/6 evaluation and collects headline metrics into one
CSV + JSON so you can see each sub-agent's marginal contribution.

Usage:
    python tests/metric3/compare_ablations.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tests.metric3.evaluate_performance import (
    attach_forward_returns,
    load_benchmark,
    load_prices,
    load_signals,
    long_short_report,
    per_stock_backtest,
    signal_level_report,
)

VARIANTS = {
    "baseline": "data/backtest/signals.parquet",
    "no_sentiment": "data/backtest/signals_no_sentiment.parquet",
    "no_fundamental": "data/backtest/signals_no_fundamental.parquet",
    "no_technical": "data/backtest/signals_no_technical.parquet",
    "no_risk": "data/backtest/signals_no_risk.parquet",
}


def summarize_variant(
    name: str, signals_path: Path, market_dir: Path, out_dir: Path
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not signals_path.exists():
        return {"variant": name, "error": f"missing {signals_path}"}

    signals = load_signals(signals_path)
    if signals.empty:
        return {"variant": name, "error": "empty signals"}

    tickers = sorted(signals["ticker"].unique().tolist())
    prices = load_prices(market_dir, tickers)
    spy = load_benchmark(market_dir, "SPY")

    backtest_start = pd.Timestamp("2022-01-03")
    backtest_end = pd.Timestamp("2024-12-31")

    signals_fwd = attach_forward_returns(signals, prices)
    signal_rep = signal_level_report(signals_fwd, out_dir / f"{name}_signal")
    per_stock = per_stock_backtest(
        signals, prices, backtest_start, backtest_end, out_dir / f"{name}_per_stock"
    )
    ls = long_short_report(
        signals,
        prices,
        spy,
        backtest_start,
        backtest_end,
        out_dir / f"{name}_long_short",
        top_k=2,
        n_shuffles=200,
    )

    row = {
        "variant": name,
        "n_signals": int(len(signals)),
        "rec_buy": int((signals["recommendation"] == "buy").sum()),
        "rec_hold": int((signals["recommendation"] == "hold").sum()),
        "rec_sell": int((signals["recommendation"].isin(["sell", "strong_sell"])).sum()),
        "rec_strong_buy": int((signals["recommendation"] == "strong_buy").sum()),
        "mean_confidence": float(signals["confidence"].mean()),
        "hit_rate_12m": signal_rep["horizons"].get("12m", {}).get("overall_hit_rate"),
        "IC_12m": signal_rep["horizons"].get("12m", {}).get("information_coefficient"),
        "per_stock_mean_Sharpe": per_stock["averages"]["strategy_mean_Sharpe"],
        "per_stock_mean_CR": per_stock["averages"]["strategy_mean_CR"],
        "per_stock_mean_MDD": per_stock["averages"]["strategy_mean_MDD"],
        "long_only_CR": ls["strategy"].get("long_only", {}).get("cumulative_return"),
        "long_only_Sharpe": ls["strategy"].get("long_only", {}).get("sharpe_ratio"),
        "long_only_MDD": ls["strategy"].get("long_only", {}).get("max_drawdown"),
        "long_short_Sharpe": ls["strategy"].get("long_short", {}).get("sharpe_ratio"),
    }
    return row


def main() -> None:
    out_dir = ROOT_DIR / "tests" / "metric3" / "outputs" / "ablations"
    market_dir = ROOT_DIR / "data" / "market"

    rows: list[dict] = []
    for name, path in VARIANTS.items():
        print(f"--- {name} ---")
        row = summarize_variant(name, ROOT_DIR / path, market_dir, out_dir)
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.set_index("variant")
    print()
    print("=== ABLATION SUMMARY ===")
    print(df.to_string(float_format=lambda x: f"{x:7.3f}" if isinstance(x, float) else str(x)))

    summary_csv = out_dir / "ablation_summary.csv"
    summary_json = out_dir / "ablation_summary.json"
    df.to_csv(summary_csv)
    with summary_json.open("w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\nwrote {summary_csv}")
    print(f"wrote {summary_json}")


if __name__ == "__main__":
    main()
