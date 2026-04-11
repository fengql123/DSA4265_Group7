#!/usr/bin/env python3
"""Metric 3 — investment-performance evaluation.

Phases 4, 5, and 6 from the plan, in one module:

- **signal_level** (Phase 4): hit rate, mean/median forward returns,
  Information Coefficient, confidence calibration, confusion matrix
  at 3m / 6m / 12m horizons. Lopez-Lira-style signal evaluation.
- **per_stock**    (Phase 5): FinMem / Li et al.-style per-ticker
  backtest. For each ticker turn the quarterly signals into a daily
  position stream, compute the return stream, then CR, SR, AV, MDD.
- **long_short**   (Phase 6): Cross-sectional long-short portfolio.
  At each rebalance, long the top-k tickers by signed_confidence and
  short the bottom-k. Plain pandas — no vectorbt, no slippage.
- **all**          runs all three and prints a combined summary.

Also reports benchmarks (buy-and-hold SPY, equal-weight universe) and
a random-shuffle baseline for significance testing on Phase 5 and 6.

Usage:
    python tests/metric3/evaluate_performance.py all
    python tests/metric3/evaluate_performance.py signal_level
    python tests/metric3/evaluate_performance.py per_stock
    python tests/metric3/evaluate_performance.py long_short --top-k 1
    python tests/metric3/evaluate_performance.py all \\
        --signals data/backtest/signals_no_sentiment.parquet \\
        --out-dir tests/metric3/outputs/ablation_no_sentiment
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RECOMMENDATION_POSITIONS = {
    "strong_buy": 1.0,
    "buy": 0.5,
    "hold": 0.0,
    "sell": -0.5,
    "strong_sell": -1.0,
}

# Long/short position for the per-stock backtest
BINARY_POSITIONS = {
    "strong_buy": 1.0,
    "buy": 1.0,
    "hold": 0.0,
    "sell": -1.0,
    "strong_sell": -1.0,
}

TRADING_DAYS_PER_YEAR = 252
HORIZONS_TRADING_DAYS = {"1m": 21, "3m": 63, "6m": 126}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_signals(signals_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(signals_path)
    # Drop rows that errored or have missing recommendation.
    df = df[~df["error"].fillna(False)]
    df = df.dropna(subset=["recommendation", "confidence"])
    df["as_of_date"] = pd.to_datetime(df["as_of_date"])
    df["ticker"] = df["ticker"].str.upper()
    df["position_fractional"] = df["recommendation"].map(RECOMMENDATION_POSITIONS)
    df["position_binary"] = df["recommendation"].map(BINARY_POSITIONS)
    df["signed_confidence"] = df["position_fractional"] * df["confidence"]
    df = df.sort_values(["ticker", "as_of_date"]).reset_index(drop=True)
    return df


def load_prices(market_dir: Path, tickers: list[str]) -> dict[str, pd.Series]:
    """Load daily close prices for each ticker. Returns dict ticker -> close Series."""
    out: dict[str, pd.Series] = {}
    for t in tickers:
        csv_path = market_dir / f"{t}.csv"
        if not csv_path.exists():
            print(f"[warn] missing price CSV for {t}: {csv_path}")
            continue
        df = pd.read_csv(csv_path, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df.sort_index()
        # Column name may be "Close" or "close".
        close_col = next((c for c in df.columns if c.lower() == "close"), None)
        if close_col is None:
            print(f"[warn] {t}: no close column found in {csv_path}")
            continue
        out[t] = df[close_col].rename(t).astype(float).dropna()
    return out


def load_benchmark(market_dir: Path, ticker: str = "SPY") -> pd.Series | None:
    prices = load_prices(market_dir, [ticker])
    return prices.get(ticker)


def align_to_trading_day(price_series: pd.Series, date: pd.Timestamp) -> pd.Timestamp | None:
    """Return the first trading day on or after `date` that exists in the series."""
    candidates = price_series.index[price_series.index >= date]
    return candidates[0] if len(candidates) > 0 else None


def forward_trading_day(price_series: pd.Series, date: pd.Timestamp, offset: int) -> pd.Timestamp | None:
    """Return the date at position `idx + offset` in the series, or None if off the end."""
    idx_array = price_series.index
    if date not in idx_array:
        return None
    idx = idx_array.get_loc(date)
    target = idx + offset
    if target >= len(idx_array):
        return None
    return idx_array[target]


# ---------------------------------------------------------------------------
# Phase 4 — Signal-level evaluation
# ---------------------------------------------------------------------------


def attach_forward_returns(
    signals: pd.DataFrame, prices: dict[str, pd.Series]
) -> pd.DataFrame:
    """Add fwd_return_3m / _6m / _12m columns per signal row."""
    signals = signals.copy()
    for horizon_name in HORIZONS_TRADING_DAYS:
        signals[f"fwd_return_{horizon_name}"] = np.nan

    for idx, row in signals.iterrows():
        ticker = row["ticker"]
        if ticker not in prices:
            continue
        series = prices[ticker]
        entry_date = align_to_trading_day(series, pd.Timestamp(row["as_of_date"]))
        if entry_date is None:
            continue
        entry_price = float(series.loc[entry_date])
        for horizon_name, offset in HORIZONS_TRADING_DAYS.items():
            fwd_date = forward_trading_day(series, entry_date, offset)
            if fwd_date is None:
                continue
            exit_price = float(series.loc[fwd_date])
            if entry_price <= 0:
                continue
            signals.at[idx, f"fwd_return_{horizon_name}"] = exit_price / entry_price - 1.0

    return signals


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, ci: float = 0.95) -> tuple[float, float]:
    if len(values) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(42)
    boots = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lo = np.quantile(boots, (1 - ci) / 2)
    hi = np.quantile(boots, 1 - (1 - ci) / 2)
    return (float(lo), float(hi))


def signal_level_report(signals_with_fwd: pd.DataFrame, out_dir: Path) -> dict:
    """Directional hit rate, per-bucket mean forward return, IC, calibration.

    Hit rate convention: only directional recommendations (buy, strong_buy,
    sell, strong_sell) are scored. A ``hold`` is not a directional bet, so
    it is excluded from both the numerator and the denominator of the hit
    rate. The per-bucket dict still reports n and mean_fwd_return for
    hold so the reader can see the opportunity-cost side of it, but the
    ``overall_hit_rate`` counts directional signals only.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"horizons": {}}

    directional_recs = {"strong_buy", "buy", "sell", "strong_sell"}

    for horizon_name in HORIZONS_TRADING_DAYS:
        col = f"fwd_return_{horizon_name}"
        df = signals_with_fwd.dropna(subset=[col]).copy()
        if df.empty:
            report["horizons"][horizon_name] = {"error": "no data"}
            continue

        # Per-bucket stats (hit_rate only defined for directional buckets).
        by_rec: dict[str, dict] = {}
        for rec, group in df.groupby("recommendation"):
            pos = RECOMMENDATION_POSITIONS[rec]
            mean_ret = float(group[col].mean())
            median_ret = float(group[col].median())
            ci_lo, ci_hi = bootstrap_ci(group[col].values)
            stats = {
                "n": int(len(group)),
                "mean_fwd_return": mean_ret,
                "median_fwd_return": median_ret,
                "fwd_return_ci95": [ci_lo, ci_hi],
            }
            if pos > 0:
                correct = (group[col] > 0).astype(int)
                stats["hit_rate"] = float(correct.mean())
            elif pos < 0:
                correct = (group[col] < 0).astype(int)
                stats["hit_rate"] = float(correct.mean())
            else:
                # hold: not a directional bet; hit_rate is undefined.
                stats["hit_rate"] = None
            by_rec[rec] = stats

        # Information Coefficient: Spearman rank corr between
        # signed_confidence and forward return, pooled across directional
        # signals only (excluding hold, whose signed_confidence is zero).
        dir_df = df[df["recommendation"].isin(directional_recs)]
        if len(dir_df) >= 2:
            ic = float(dir_df["signed_confidence"].rank().corr(dir_df[col].rank()))
        else:
            ic = float("nan")

        # Overall hit rate: directional signals only.
        dir_total = 0
        dir_hits = 0
        for rec, stats in by_rec.items():
            if rec not in directional_recs:
                continue
            dir_total += stats["n"]
            dir_hits += int(round(stats["hit_rate"] * stats["n"])) if stats["hit_rate"] is not None else 0
        overall_hit_rate = dir_hits / dir_total if dir_total else float("nan")

        # Confidence calibration (3 buckets, directional signals only).
        calib = {}
        for low, high, label in [(0.0, 0.55, "low"), (0.55, 0.75, "mid"), (0.75, 1.01, "high")]:
            bucket = dir_df[(dir_df["confidence"] >= low) & (dir_df["confidence"] < high)]
            if bucket.empty:
                calib[label] = {"n": 0}
                continue
            pos = bucket["position_fractional"]
            directional_correct = (
                ((pos > 0) & (bucket[col] > 0))
                | ((pos < 0) & (bucket[col] < 0))
            )
            calib[label] = {
                "n": int(len(bucket)),
                "hit_rate": float(directional_correct.mean()),
                "mean_fwd_return": float(bucket[col].mean()),
            }

        report["horizons"][horizon_name] = {
            "n_total": int(len(df)),
            "n_directional": int(dir_total),
            "overall_hit_rate": overall_hit_rate,
            "information_coefficient": ic,
            "by_recommendation": by_rec,
            "confidence_calibration": calib,
        }

    # Persist
    path = out_dir / "signal_level_report.json"
    with path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"[phase4] wrote {path}")
    return report


# ---------------------------------------------------------------------------
# Phase 5 — Per-stock backtest (plain pandas, no vectorbt)
# ---------------------------------------------------------------------------


def _annualize_sharpe(daily_returns: pd.Series) -> float:
    if len(daily_returns) < 2:
        return float("nan")
    mean = daily_returns.mean()
    std = daily_returns.std(ddof=0)
    if std == 0 or np.isnan(std):
        return float("nan")
    return float(mean / std * np.sqrt(TRADING_DAYS_PER_YEAR))


def _annualized_vol(daily_returns: pd.Series) -> float:
    if len(daily_returns) < 2:
        return float("nan")
    return float(daily_returns.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR))


def _max_drawdown(cum_equity: pd.Series) -> float:
    if cum_equity.empty:
        return float("nan")
    rolling_max = cum_equity.cummax()
    drawdown = (cum_equity - rolling_max) / rolling_max
    return float(drawdown.min())


def _cumulative_return(daily_returns: pd.Series) -> float:
    if daily_returns.empty:
        return float("nan")
    return float((1 + daily_returns).prod() - 1)


@dataclass
class PerformanceRow:
    name: str
    cr: float
    sharpe: float
    vol: float
    mdd: float
    n_days: int

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "cumulative_return": self.cr,
            "sharpe_ratio": self.sharpe,
            "annualized_volatility": self.vol,
            "max_drawdown": self.mdd,
            "n_days": self.n_days,
        }


def build_position_stream(
    signals_for_ticker: pd.DataFrame,
    price_index: pd.DatetimeIndex,
    position_col: str,
) -> pd.Series:
    """Turn quarterly signals into a daily step-function position series.

    Position changes on the signal date (or the first trading day on
    or after it) and holds until the next signal.
    """
    positions = pd.Series(0.0, index=price_index)
    sig = signals_for_ticker.sort_values("as_of_date")
    current_position = 0.0
    current_start_idx = 0
    for _, row in sig.iterrows():
        as_of = pd.Timestamp(row["as_of_date"])
        # Find first trading day on or after as_of.
        candidate_positions = positions.index[positions.index >= as_of]
        if len(candidate_positions) == 0:
            break
        start_day = candidate_positions[0]
        start_idx = positions.index.get_loc(start_day)
        # Fill [current_start_idx, start_idx) with current_position.
        positions.iloc[current_start_idx:start_idx] = current_position
        current_position = float(row[position_col])
        current_start_idx = start_idx
    # Fill from last change to end.
    positions.iloc[current_start_idx:] = current_position
    return positions


def per_stock_backtest(
    signals: pd.DataFrame,
    prices: dict[str, pd.Series],
    backtest_start: pd.Timestamp,
    backtest_end: pd.Timestamp,
    out_dir: Path,
    position_col: str = "position_binary",
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[PerformanceRow] = []
    per_ticker_details: dict[str, dict] = {}

    for ticker, sig in signals.groupby("ticker"):
        if ticker not in prices:
            continue
        series = prices[ticker]
        # Restrict to backtest window.
        series_win = series.loc[(series.index >= backtest_start) & (series.index <= backtest_end)]
        if series_win.empty:
            continue

        positions = build_position_stream(sig, series_win.index, position_col)
        daily_ret = series_win.pct_change().fillna(0.0)
        # Position held at end of prior bar (one-bar lag to avoid lookahead
        # on the rebalance day).
        strat_ret = daily_ret * positions.shift(1).fillna(0.0)
        cum_equity = (1 + strat_ret).cumprod()

        strat_perf = PerformanceRow(
            name=f"strategy/{ticker}",
            cr=_cumulative_return(strat_ret),
            sharpe=_annualize_sharpe(strat_ret),
            vol=_annualized_vol(strat_ret),
            mdd=_max_drawdown(cum_equity),
            n_days=int(len(strat_ret)),
        )

        # Buy-and-hold benchmark.
        bh_ret = daily_ret
        bh_equity = (1 + bh_ret).cumprod()
        bh_perf = PerformanceRow(
            name=f"buy_and_hold/{ticker}",
            cr=_cumulative_return(bh_ret),
            sharpe=_annualize_sharpe(bh_ret),
            vol=_annualized_vol(bh_ret),
            mdd=_max_drawdown(bh_equity),
            n_days=int(len(bh_ret)),
        )

        rows.append(strat_perf)
        rows.append(bh_perf)
        per_ticker_details[ticker] = {
            "strategy": strat_perf.to_dict(),
            "buy_and_hold": bh_perf.to_dict(),
            "alpha_cr": strat_perf.cr - bh_perf.cr,
            "alpha_sharpe": strat_perf.sharpe - bh_perf.sharpe,
        }

    # Cross-sectional averages.
    strategy_rows = [r for r in rows if r.name.startswith("strategy/")]
    buy_hold_rows = [r for r in rows if r.name.startswith("buy_and_hold/")]

    def _avg(rows: list[PerformanceRow], attr: str) -> float:
        vals = [getattr(r, attr) for r in rows if not np.isnan(getattr(r, attr))]
        return float(np.mean(vals)) if vals else float("nan")

    averages = {
        "strategy_mean_CR": _avg(strategy_rows, "cr"),
        "strategy_mean_Sharpe": _avg(strategy_rows, "sharpe"),
        "strategy_mean_Vol": _avg(strategy_rows, "vol"),
        "strategy_mean_MDD": _avg(strategy_rows, "mdd"),
        "buy_hold_mean_CR": _avg(buy_hold_rows, "cr"),
        "buy_hold_mean_Sharpe": _avg(buy_hold_rows, "sharpe"),
        "buy_hold_mean_Vol": _avg(buy_hold_rows, "vol"),
        "buy_hold_mean_MDD": _avg(buy_hold_rows, "mdd"),
    }

    report = {
        "position_convention": position_col,
        "averages": averages,
        "per_ticker": per_ticker_details,
    }

    with (out_dir / "per_stock_report.json").open("w") as f:
        json.dump(report, f, indent=2)
    print(f"[phase5] wrote {out_dir / 'per_stock_report.json'}")
    print(f"[phase5] strategy mean Sharpe: {averages['strategy_mean_Sharpe']:.3f}  "
          f"buy-and-hold mean Sharpe: {averages['buy_hold_mean_Sharpe']:.3f}")
    return report


# ---------------------------------------------------------------------------
# Phase 6 — Confidence-weighted portfolio
# ---------------------------------------------------------------------------


def _rec_direction(recommendation: str) -> float:
    """Map a recommendation to a directional sign.

    buy / strong_buy  -> +1
    hold              ->  0  (no position)
    sell / strong_sell -> -1  (short)
    """
    if recommendation in ("strong_buy", "buy"):
        return 1.0
    if recommendation in ("strong_sell", "sell"):
        return -1.0
    return 0.0


def build_confidence_weighted_portfolio(
    signals: pd.DataFrame,
    prices: dict[str, pd.Series],
    backtest_start: pd.Timestamp,
    backtest_end: pd.Timestamp,
) -> pd.Series:
    """Build a confidence-weighted portfolio return stream.

    At each rebalance, each ticker's raw position is
    ``direction(recommendation) * confidence``, where direction is
    +1 for buy, 0 for hold, -1 for sell. Raw positions are then
    normalized across tickers so the gross exposure
    ``sum(|w_i|) = 1`` (full use of capital), unless every ticker is
    on hold, in which case the portfolio is flat for that window.

    This construction makes every recommendation contribute in
    proportion to its confidence, eliminates the arbitrary top-k
    choice, and treats ``hold`` as "do nothing" and ``sell`` as
    "short" — which is what the five-bucket recommendation scale is
    supposed to mean. With the one-bar weight lag applied at the end,
    there is no same-day lookahead.
    """
    rebalance_dates = sorted(signals["as_of_date"].unique())
    if not rebalance_dates:
        return pd.Series(dtype=float)

    # Unified trading-day index across all tickers.
    all_indices = [s.index for s in prices.values()]
    if not all_indices:
        return pd.Series(dtype=float)
    unified_index = sorted(set().union(*[set(idx) for idx in all_indices]))
    unified_index = pd.DatetimeIndex([d for d in unified_index if backtest_start <= d <= backtest_end])
    if len(unified_index) == 0:
        return pd.Series(dtype=float)

    returns_frame = pd.DataFrame(index=unified_index)
    for t, s in prices.items():
        r = s.pct_change().fillna(0.0)
        returns_frame[t] = r.reindex(unified_index, method="ffill").fillna(0.0)

    weights = pd.DataFrame(0.0, index=unified_index, columns=returns_frame.columns)

    def pick_weights(as_of: pd.Timestamp) -> pd.Series:
        snap = signals[signals["as_of_date"] == as_of].copy()
        snap = snap[snap["ticker"].isin(weights.columns)]
        w = pd.Series(0.0, index=weights.columns)
        if snap.empty:
            return w
        # Raw position per ticker: direction * confidence.
        for _, row in snap.iterrows():
            direction = _rec_direction(row["recommendation"])
            raw = direction * float(row["confidence"])
            w[row["ticker"]] = raw
        # Normalize by total gross exposure so |w| sums to 1 (or 0).
        gross = float(w.abs().sum())
        if gross > 0:
            w = w / gross
        return w

    current_weights = pd.Series(0.0, index=weights.columns)
    rebal_idx = 0
    for day in unified_index:
        while rebal_idx < len(rebalance_dates) and pd.Timestamp(rebalance_dates[rebal_idx]) <= day:
            current_weights = pick_weights(pd.Timestamp(rebalance_dates[rebal_idx]))
            rebal_idx += 1
        weights.loc[day] = current_weights

    # One-bar lag to avoid same-day lookahead on rebalance dates.
    weights_lagged = weights.shift(1).fillna(0.0)
    port_returns = (weights_lagged * returns_frame).sum(axis=1)
    return port_returns


def confidence_weighted_report(
    signals: pd.DataFrame,
    prices: dict[str, pd.Series],
    spy: pd.Series | None,
    backtest_start: pd.Timestamp,
    backtest_end: pd.Timestamp,
    out_dir: Path,
    n_shuffles: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    port_ret = build_confidence_weighted_portfolio(signals, prices, backtest_start, backtest_end)
    if port_ret.empty:
        result_strategy = {"error": "empty portfolio (check data)"}
    else:
        equity = (1 + port_ret).cumprod()
        result_strategy = {
            "cumulative_return": _cumulative_return(port_ret),
            "sharpe_ratio": _annualize_sharpe(port_ret),
            "annualized_volatility": _annualized_vol(port_ret),
            "max_drawdown": _max_drawdown(equity),
            "n_days": int(len(port_ret)),
        }

    # Benchmarks.
    benchmarks: dict = {}
    if spy is not None:
        spy_win = spy.loc[(spy.index >= backtest_start) & (spy.index <= backtest_end)]
        if not spy_win.empty:
            spy_ret = spy_win.pct_change().fillna(0.0)
            spy_eq = (1 + spy_ret).cumprod()
            benchmarks["SPY_buy_and_hold"] = {
                "cumulative_return": _cumulative_return(spy_ret),
                "sharpe_ratio": _annualize_sharpe(spy_ret),
                "annualized_volatility": _annualized_vol(spy_ret),
                "max_drawdown": _max_drawdown(spy_eq),
                "n_days": int(len(spy_ret)),
            }

    # Equal-weight universe benchmark.
    universe_tickers = [t for t in signals["ticker"].unique() if t in prices]
    if universe_tickers:
        eq_df = pd.DataFrame({t: prices[t].pct_change() for t in universe_tickers})
        eq_df = eq_df.loc[(eq_df.index >= backtest_start) & (eq_df.index <= backtest_end)].fillna(0.0)
        eq_ret = eq_df.mean(axis=1)
        eq_equity = (1 + eq_ret).cumprod()
        benchmarks["equal_weight_universe"] = {
            "cumulative_return": _cumulative_return(eq_ret),
            "sharpe_ratio": _annualize_sharpe(eq_ret),
            "annualized_volatility": _annualized_vol(eq_ret),
            "max_drawdown": _max_drawdown(eq_equity),
            "n_days": int(len(eq_ret)),
        }

    # Random-shuffle null distribution: permute (recommendation,
    # confidence) pairs across (ticker, date) cells and re-run the
    # confidence-weighted portfolio math. No new LLM calls.
    shuffle_sharpes = []
    shuffle_crs = []
    rng = np.random.default_rng(42)
    for _ in range(n_shuffles):
        shuffled = signals.copy()
        perm = rng.permutation(len(shuffled))
        shuffled["recommendation"] = shuffled["recommendation"].values[perm]
        shuffled["confidence"] = shuffled["confidence"].values[perm]
        pr = build_confidence_weighted_portfolio(
            shuffled, prices, backtest_start, backtest_end
        )
        if not pr.empty:
            shuffle_sharpes.append(_annualize_sharpe(pr))
            shuffle_crs.append(_cumulative_return(pr))

    if shuffle_sharpes:
        benchmarks["random_shuffle"] = {
            "n_trials": len(shuffle_sharpes),
            "mean_sharpe": float(np.nanmean(shuffle_sharpes)),
            "p95_sharpe": float(np.nanpercentile(shuffle_sharpes, 95)),
            "mean_cr": float(np.nanmean(shuffle_crs)),
            "p95_cr": float(np.nanpercentile(shuffle_crs, 95)),
        }

    report = {
        "construction": "confidence_weighted",
        "backtest_window": [backtest_start.strftime("%Y-%m-%d"), backtest_end.strftime("%Y-%m-%d")],
        "strategy": result_strategy,
        "benchmarks": benchmarks,
    }

    with (out_dir / "portfolio_report.json").open("w") as f:
        json.dump(report, f, indent=2)
    print(f"[phase6] wrote {out_dir / 'portfolio_report.json'}")
    if "sharpe_ratio" in result_strategy:
        print(
            f"[phase6] confidence_weighted Sharpe="
            f"{result_strategy['sharpe_ratio']}, CR="
            f"{result_strategy['cumulative_return']}"
        )
    return report


# Backwards-compat alias so plot_backtest.py can keep calling the old name.
long_short_report = confidence_weighted_report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run_all(args: argparse.Namespace) -> None:
    signals_path = Path(args.signals)
    market_dir = Path(args.market_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Signals:  {signals_path}")
    print(f"Market:   {market_dir}")
    print(f"Out:      {out_dir}\n")

    signals = load_signals(signals_path)
    if signals.empty:
        print("No usable signals.")
        return
    print(f"Loaded {len(signals)} signals across {signals['ticker'].nunique()} tickers")

    tickers = sorted(signals["ticker"].unique().tolist())
    prices = load_prices(market_dir, tickers)
    if not prices:
        print("No price data loaded. Aborting.")
        return
    spy = load_benchmark(market_dir, "SPY")

    backtest_start = pd.Timestamp(args.backtest_start)
    backtest_end = pd.Timestamp(args.backtest_end)

    # Phase 4
    if args.command in ("all", "signal_level"):
        signals_fwd = attach_forward_returns(signals, prices)
        signal_level_report(signals_fwd, out_dir)

    # Phase 5
    if args.command in ("all", "per_stock"):
        per_stock_backtest(
            signals,
            prices,
            backtest_start,
            backtest_end,
            out_dir,
            position_col=args.position_col,
        )

    # Phase 6
    if args.command in ("all", "portfolio"):
        confidence_weighted_report(
            signals,
            prices,
            spy,
            backtest_start,
            backtest_end,
            out_dir,
            n_shuffles=args.shuffles,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Metric 3 — investment performance evaluation.")
    parser.add_argument(
        "command",
        choices=["all", "signal_level", "per_stock", "portfolio"],
        help="Which phase to run.",
    )
    parser.add_argument("--signals", default="data/backtest/signals.parquet")
    parser.add_argument("--market-dir", default="data/market")
    parser.add_argument("--out-dir", default="tests/metric3/outputs")
    parser.add_argument("--backtest-start", default="2024-10-01")
    parser.add_argument("--backtest-end", default="2026-04-11")
    parser.add_argument("--shuffles", type=int, default=500, help="Random-shuffle trials for null distribution.")
    parser.add_argument(
        "--position-col",
        choices=["position_binary", "position_fractional"],
        default="position_binary",
        help="Position convention for per-stock backtest.",
    )
    args = parser.parse_args()
    run_all(args)


if __name__ == "__main__":
    main()
