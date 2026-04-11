#!/usr/bin/env python3
"""Phase 3 driver: run the pipeline on each (ticker, as_of_date) pair.

Reads `data/backtest/universe.yaml`, invokes the LangGraph pipeline
for each pair via `graph.batch()` in chunks, then appends the results
to `data/backtest/signals.parquet`.

The pipeline is fully date-agnostic after the Phase 2 refactor — the
as-of date is communicated via the user's query string, and the
main agent either extracts it or falls back to `get_today_date`.

Usage:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --limit 10 --chunk 5
    python scripts/run_backtest.py --disabled-agents sentiment --signals-out data/backtest/signals_no_sentiment.parquet
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

OUTPUTS_DIR = ROOT_DIR / "outputs" / "backtest"


def _slug(ticker: str, as_of: str) -> str:
    return f"{ticker.upper()}_{as_of.replace('-', '')}"


def _save_report(result: dict, ticker: str, as_of: str) -> tuple[str, str]:
    """Save markdown + JSON reports and return their paths."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    memo = result.get("investment_memo")
    slug = _slug(ticker, as_of)

    md_path = OUTPUTS_DIR / f"{slug}_report.md"
    json_path = OUTPUTS_DIR / f"{slug}_report.json"

    if memo:
        md_path.write_text(memo.report_markdown or "")
        with json_path.open("w") as f:
            json.dump(memo.model_dump(), f, indent=2, default=str)
    else:
        md_path.write_text("")
        with json_path.open("w") as f:
            errors = result.get("errors", [])
            json.dump({"error": True, "errors": errors}, f, indent=2)

    return str(md_path.relative_to(ROOT_DIR)), str(json_path.relative_to(ROOT_DIR))


def _extract_row(query: str, result: dict, ticker: str, as_of: str) -> dict:
    memo = result.get("investment_memo")
    md_path, json_path = _save_report(result, ticker, as_of)

    if memo is None:
        return {
            "ticker": ticker,
            "as_of_date": as_of,
            "recommendation": None,
            "confidence": None,
            "report_md_path": md_path,
            "report_json_path": json_path,
            "error": True,
            "error_msg": "; ".join(result.get("errors", [])) or "no investment_memo",
        }

    return {
        "ticker": ticker,
        "as_of_date": as_of,
        "recommendation": memo.recommendation,
        "confidence": float(memo.confidence) if memo.confidence is not None else None,
        "report_md_path": md_path,
        "report_json_path": json_path,
        "error": False,
        "error_msg": "",
    }


def _load_existing_signals(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _pair_is_done(existing: pd.DataFrame | None, ticker: str, as_of: str) -> bool:
    if existing is None or existing.empty:
        return False
    mask = (existing["ticker"] == ticker) & (existing["as_of_date"] == as_of) & (~existing["error"])
    return bool(mask.any())


def run(
    universe_path: Path,
    signals_out: Path,
    chunk_size: int,
    limit: int | None,
    disabled_agents: list[str] | None,
    tickers_filter: list[str] | None,
    resume: bool,
) -> None:
    with universe_path.open() as f:
        universe = yaml.safe_load(f)
    pairs = universe.get("pairs", [])
    if tickers_filter:
        tfset = {t.upper() for t in tickers_filter}
        pairs = [p for p in pairs if p["ticker"].upper() in tfset]
    if limit is not None:
        pairs = pairs[:limit]
    if not pairs:
        print("No pairs to run.")
        return

    existing = _load_existing_signals(signals_out) if resume else None
    if existing is not None:
        before = len(pairs)
        pairs = [p for p in pairs if not _pair_is_done(existing, p["ticker"], p["as_of_date"])]
        print(f"[resume] skipping {before - len(pairs)} already-completed pairs")

    if not pairs:
        print("All pairs already completed.")
        return

    print(f"Running {len(pairs)} pair(s) in chunks of {chunk_size}")
    if disabled_agents:
        print(f"Ablation: disabled_agents={disabled_agents}")

    # Warm up the ChromaDB + embedding-model singletons before the
    # parallel batch starts. Without this, the first concurrent
    # rag_retrieve calls race on the lazy-initialization and produce
    # transient "Could not connect to tenant default_tenant" warnings.
    try:
        from src.rag.retriever import retrieve as _warmup_retrieve
        _warmup_retrieve(query="warmup", collection_names=["sec_filings"], top_k=1)
        print("[warmup] retriever + embedding model ready")
    except Exception as exc:
        print(f"[warmup] ignored error: {exc}")

    # Build graph ONCE. We temporarily monkey-patch MainAgent to
    # accept disabled_agents via a module-level flag so the existing
    # graph.py build path does not need to change.
    from src.graph import build_graph
    from src.agents import main_agent as main_agent_mod

    original_init = main_agent_mod.MainAgent.__init__

    def patched_init(self, disabled_agents=disabled_agents, **kwargs):  # type: ignore
        original_init(self, disabled_agents=disabled_agents)

    if disabled_agents:
        main_agent_mod.MainAgent.__init__ = patched_init  # type: ignore
    try:
        graph = build_graph(debug=False)
    finally:
        main_agent_mod.MainAgent.__init__ = original_init  # type: ignore

    all_rows: list[dict] = []
    if existing is not None and not existing.empty:
        all_rows.extend(existing.to_dict(orient="records"))

    for i in range(0, len(pairs), chunk_size):
        chunk = pairs[i : i + chunk_size]
        queries = [
            f"Analyze {p['ticker']} stock as of {p['as_of_date']}." for p in chunk
        ]
        states = [{"query": q, "errors": []} for q in queries]
        print(
            f"\n[chunk {i // chunk_size + 1}/{(len(pairs) + chunk_size - 1) // chunk_size}] "
            f"{len(chunk)} pair(s)"
        )
        for p, q in zip(chunk, queries):
            print(f"  - {p['ticker']} as of {p['as_of_date']}")

        chunk_results = []
        try:
            chunk_results = graph.batch(states)
        except Exception as exc:
            print(f"  batch failed: {exc}")
            traceback.print_exc()
            # Fallback: run one at a time so a single bad ticker
            # doesn't wipe the whole chunk.
            for st in states:
                try:
                    chunk_results.append(graph.invoke(st))
                except Exception as inner:
                    print(f"    single-run failed: {inner}")
                    chunk_results.append({"errors": [str(inner)]})

        for pair, query, result in zip(chunk, queries, chunk_results):
            row = _extract_row(query, result, pair["ticker"], pair["as_of_date"])
            all_rows.append(row)
            status = "OK" if not row["error"] else "ERR"
            print(
                f"    [{status}] {pair['ticker']} {pair['as_of_date']}: "
                f"{row['recommendation']} ({row['confidence']})"
            )

        # Checkpoint: write parquet after every chunk so a failure
        # partway through does not wipe earlier work.
        _write_parquet(all_rows, signals_out)

    print(f"\nDone. {len(all_rows)} rows -> {signals_out}")


def _write_parquet(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    # Dedupe on (ticker, as_of_date) — keep last (most recent run).
    df = df.drop_duplicates(subset=["ticker", "as_of_date"], keep="last")
    df.to_parquet(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the backtest pipeline across (ticker, as_of_date) pairs.")
    parser.add_argument(
        "--universe",
        default="data/backtest/universe.yaml",
        help="Path to the universe manifest.",
    )
    parser.add_argument(
        "--signals-out",
        default="data/backtest/signals.parquet",
        help="Output parquet for signals.",
    )
    parser.add_argument("--chunk", type=int, default=5, help="graph.batch chunk size.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on pairs.")
    parser.add_argument(
        "--disabled-agents",
        nargs="+",
        default=None,
        help="Sub-agents to disable for ablation (e.g. sentiment technical).",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Optional filter: only run these tickers.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-run pairs even if they already exist in the parquet.",
    )
    args = parser.parse_args()

    run(
        universe_path=ROOT_DIR / args.universe,
        signals_out=ROOT_DIR / args.signals_out,
        chunk_size=args.chunk,
        limit=args.limit,
        disabled_agents=args.disabled_agents,
        tickers_filter=args.tickers,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
