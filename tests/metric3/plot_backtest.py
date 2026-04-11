#!/usr/bin/env python3
"""Backtest plots for Phases 4, 5, and 6.

Produces the 6 core plots from the design discussion:

- Phase 4
    1. `phase4_hit_rate.png`           — grouped bar chart, hit rate per
      recommendation bucket × horizon with bootstrap CI.
    2. `phase4_forward_return_violin.png` — forward-return distribution
      per recommendation bucket at the 6m horizon.

- Phase 5
    3. `phase5_equity_grid.png`        — 5-panel equity-curve grid,
       strategy vs buy-and-hold per ticker.
    4. `phase5_alpha_bars.png`         — per-ticker alpha (strategy CR
       minus buy-and-hold CR) bar chart.

- Phase 6
    5. `phase6_cumulative_returns.png` — overlaid cumulative return
       curves: long_only, SPY, equal-weight universe, and the 5-95th
       percentile band of the random-shuffle null.
    6. `phase6_null_histogram.png`     — random-shuffle null Sharpe
       histogram with the real strategy's Sharpe marked.

Usage:
    python tests/metric3/plot_backtest.py
    python tests/metric3/plot_backtest.py --out-dir tests/metric3/outputs/plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tests.metric3.evaluate_performance import (
    HORIZONS_TRADING_DAYS,
    RECOMMENDATION_POSITIONS,
    TRADING_DAYS_PER_YEAR,
    _annualize_sharpe,
    _cumulative_return,
    _max_drawdown,
    attach_forward_returns,
    bootstrap_ci,
    build_confidence_weighted_portfolio,
    build_position_stream,
    load_benchmark,
    load_prices,
    load_signals,
)

# Consistent color palette used across every figure.
COLORS = {
    "strategy":      "#1f4e79",   # dark blue
    "buy_and_hold":  "#808080",   # gray
    "spy":           "#a11d33",   # dark red
    "equal_weight":  "#2e7d32",   # green
    "null_band":     "#c0c0c0",   # light gray
    "real":          "#1f4e79",
    "buy":           "#2e7d32",
    "hold":          "#b0b0b0",
    "sell":          "#a11d33",
    "positive":      "#2e7d32",
    "negative":      "#a11d33",
}

BUCKET_ORDER = ["strong_buy", "buy", "hold", "sell", "strong_sell"]
HORIZON_ORDER = ["1m", "3m", "6m"]


# ---------------------------------------------------------------------------
# Phase 4 plots
# ---------------------------------------------------------------------------


DIRECTIONAL_RECS = {"strong_buy", "buy", "sell", "strong_sell"}


def _directional_hit(rec: str, fwd_ret: float) -> bool | None:
    """Return True/False for directional recs, None for hold.

    Hold is not a directional bet and is excluded from hit-rate
    scoring entirely.
    """
    pos = RECOMMENDATION_POSITIONS[rec]
    if pos > 0:
        return fwd_ret > 0
    if pos < 0:
        return fwd_ret < 0
    return None  # hold


def plot_hit_rate(signals_fwd: pd.DataFrame, out_path: Path) -> None:
    """Grouped bar chart: directional hit rate per bucket × horizon
    with 95% bootstrap CI. Hold is excluded from the plot because it
    is not a directional bet."""
    buckets_present = [
        b for b in BUCKET_ORDER
        if b in DIRECTIONAL_RECS and (signals_fwd["recommendation"] == b).any()
    ]
    if not buckets_present:
        return

    x = np.arange(len(buckets_present))
    bar_width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, horizon in enumerate(HORIZON_ORDER):
        col = f"fwd_return_{horizon}"
        means = []
        errs_lo = []
        errs_hi = []
        for bucket in buckets_present:
            sub = signals_fwd[(signals_fwd["recommendation"] == bucket)].dropna(subset=[col])
            if sub.empty:
                means.append(np.nan)
                errs_lo.append(0)
                errs_hi.append(0)
                continue
            hits = sub.apply(lambda r: _directional_hit(r["recommendation"], r[col]), axis=1)
            hits = hits.dropna().astype(float).values
            if len(hits) == 0:
                means.append(np.nan)
                errs_lo.append(0)
                errs_hi.append(0)
                continue
            mean_hit = float(hits.mean())
            lo, hi = bootstrap_ci(hits, n_boot=1000)
            means.append(mean_hit)
            errs_lo.append(max(0, mean_hit - lo))
            errs_hi.append(max(0, hi - mean_hit))
        ax.bar(
            x + (i - 1) * bar_width,
            means,
            width=bar_width,
            yerr=[errs_lo, errs_hi],
            capsize=3,
            label=horizon,
            alpha=0.85,
        )

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.7, label="coin-flip")
    ax.set_xticks(x)
    ax.set_xticklabels(buckets_present)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Directional hit rate")
    ax.set_xlabel("Recommendation bucket (directional only; hold excluded)")
    ax.set_title("Phase 4 — Directional hit rate per bucket × horizon (95% bootstrap CI)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_forward_return_violin(signals_fwd: pd.DataFrame, out_path: Path, horizon: str = "6m") -> None:
    """Violin plot: forward return distribution per recommendation bucket."""
    col = f"fwd_return_{horizon}"
    buckets = [b for b in BUCKET_ORDER if (signals_fwd["recommendation"] == b).any()]
    data = []
    labels = []
    for b in buckets:
        vals = signals_fwd[signals_fwd["recommendation"] == b][col].dropna().values
        if len(vals) > 0:
            data.append(vals)
            labels.append(f"{b}\n(n={len(vals)})")

    if not data:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    parts = ax.violinplot(data, showmeans=True, showmedians=True)
    # color the bodies
    for i, pc in enumerate(parts["bodies"]):
        b = buckets[i]
        if RECOMMENDATION_POSITIONS[b] > 0:
            pc.set_facecolor(COLORS["buy"])
        elif RECOMMENDATION_POSITIONS[b] < 0:
            pc.set_facecolor(COLORS["sell"])
        else:
            pc.set_facecolor(COLORS["hold"])
        pc.set_alpha(0.65)
    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"{horizon} forward return")
    ax.set_title(f"Phase 4 — {horizon} forward return distribution per recommendation")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# Phase 5 plots
# ---------------------------------------------------------------------------


def plot_equity_grid(
    signals: pd.DataFrame,
    prices: dict[str, pd.Series],
    out_path: Path,
    backtest_start: pd.Timestamp,
    backtest_end: pd.Timestamp,
    position_col: str = "position_binary",
) -> None:
    """5-panel equity-curve grid: strategy vs buy-and-hold per ticker."""
    tickers = sorted(signals["ticker"].unique().tolist())
    tickers = [t for t in tickers if t in prices]
    n = len(tickers)
    if n == 0:
        return
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows), sharey=False)
    axes = np.array(axes).reshape(-1)

    for ax, ticker in zip(axes, tickers):
        series = prices[ticker]
        win = series.loc[(series.index >= backtest_start) & (series.index <= backtest_end)]
        if win.empty:
            ax.set_title(f"{ticker} (no data)")
            continue

        sig = signals[signals["ticker"] == ticker]
        positions = build_position_stream(sig, win.index, position_col)
        daily_ret = win.pct_change().fillna(0.0)
        strat_ret = daily_ret * positions.shift(1).fillna(0.0)

        strat_eq = (1 + strat_ret).cumprod()
        bh_eq = (1 + daily_ret).cumprod()

        ax.plot(strat_eq.index, strat_eq.values, color=COLORS["strategy"], linewidth=1.8, label="strategy")
        ax.plot(bh_eq.index, bh_eq.values, color=COLORS["buy_and_hold"], linewidth=1.4, linestyle="--", label="buy-and-hold")

        # Shade regions where position is 0 (cash).
        in_cash = positions == 0
        if in_cash.any():
            cash_mask = in_cash.astype(int).diff().fillna(0)
            starts = cash_mask[cash_mask == 1].index
            ends = cash_mask[cash_mask == -1].index
            if len(starts) > 0 and (len(ends) == 0 or starts[0] < ends[0] if len(ends) else True):
                # handle edge: already in cash at start
                pass
            # simple approach: fill_between on the cash periods
            ax.fill_between(
                positions.index,
                0,
                max(strat_eq.max(), bh_eq.max()) * 1.02,
                where=in_cash.values,
                alpha=0.08,
                color=COLORS["hold"],
            )

        ax.set_title(f"{ticker}: strat {strat_eq.iloc[-1]-1:+.1%}  BH {bh_eq.iloc[-1]-1:+.1%}", fontsize=10)
        ax.axhline(1.0, color="black", linewidth=0.6, alpha=0.4)
        ax.grid(linestyle=":", alpha=0.4)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7, loc="upper left")

        # Force a sparse, readable date axis so labels don't overlap.
        # One tick per quarter with rotated labels fits cleanly even in
        # narrow subplots.
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(30)
            lbl.set_ha("right")

    # Hide unused panels.
    for i in range(n, len(axes)):
        axes[i].axis("off")

    fig.suptitle("Phase 5 — Strategy vs Buy-and-Hold equity curves", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_alpha_bars(
    signals: pd.DataFrame,
    prices: dict[str, pd.Series],
    out_path: Path,
    backtest_start: pd.Timestamp,
    backtest_end: pd.Timestamp,
    position_col: str = "position_binary",
) -> None:
    """Bar chart: per-ticker alpha = strategy CR − buy-and-hold CR."""
    tickers = sorted(signals["ticker"].unique().tolist())
    tickers = [t for t in tickers if t in prices]

    alphas = []
    strat_crs = []
    bh_crs = []
    for t in tickers:
        win = prices[t].loc[(prices[t].index >= backtest_start) & (prices[t].index <= backtest_end)]
        if win.empty:
            alphas.append(0)
            strat_crs.append(0)
            bh_crs.append(0)
            continue
        sig = signals[signals["ticker"] == t]
        positions = build_position_stream(sig, win.index, position_col)
        daily_ret = win.pct_change().fillna(0.0)
        strat_ret = daily_ret * positions.shift(1).fillna(0.0)
        strat_cr = _cumulative_return(strat_ret)
        bh_cr = _cumulative_return(daily_ret)
        alphas.append(strat_cr - bh_cr)
        strat_crs.append(strat_cr)
        bh_crs.append(bh_cr)

    fig, ax = plt.subplots(figsize=(8, 5))
    bar_colors = [COLORS["positive"] if a >= 0 else COLORS["negative"] for a in alphas]
    bars = ax.bar(tickers, alphas, color=bar_colors, alpha=0.8)
    for bar, a, s, b in zip(bars, alphas, strat_crs, bh_crs):
        label = f"{a:+.1%}"
        height = bar.get_height()
        offset = 0.015 if height >= 0 else -0.015
        va = "bottom" if height >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, height + offset, label, ha="center", va=va, fontsize=9, fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Alpha (strategy CR − buy-and-hold CR)")
    ax.set_title("Phase 5 — Per-ticker alpha over backtest window")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# Phase 6 plots
# ---------------------------------------------------------------------------


def _portfolio_equity_curve(daily_returns: pd.Series) -> pd.Series:
    if daily_returns.empty:
        return pd.Series(dtype=float)
    return (1 + daily_returns).cumprod()


def _random_shuffle_null_curves(
    signals: pd.DataFrame,
    prices: dict[str, pd.Series],
    backtest_start: pd.Timestamp,
    backtest_end: pd.Timestamp,
    n_trials: int,
    rng_seed: int = 42,
) -> tuple[np.ndarray, list[float]]:
    """Return (matrix of cumulative equity paths, list of per-trial Sharpes)."""
    rng = np.random.default_rng(rng_seed)
    all_curves = []
    sharpes = []
    for _ in range(n_trials):
        shuffled = signals.copy()
        perm = rng.permutation(len(shuffled))
        shuffled["recommendation"] = shuffled["recommendation"].values[perm]
        shuffled["confidence"] = shuffled["confidence"].values[perm]
        pr = build_confidence_weighted_portfolio(
            shuffled, prices, backtest_start, backtest_end
        )
        if pr.empty:
            continue
        eq = _portfolio_equity_curve(pr)
        all_curves.append(eq)
        sharpes.append(_annualize_sharpe(pr))

    if not all_curves:
        return np.array([]), []

    # Align on a common index (use the first curve's index as reference).
    ref = all_curves[0].index
    aligned = np.stack([c.reindex(ref).ffill().fillna(1.0).values for c in all_curves])
    return aligned, sharpes


def plot_cumulative_returns(
    signals: pd.DataFrame,
    prices: dict[str, pd.Series],
    spy: pd.Series | None,
    out_path: Path,
    backtest_start: pd.Timestamp,
    backtest_end: pd.Timestamp,
    n_shuffles: int,
) -> tuple[pd.Series, list[float]]:
    """Overlaid equity curves: strategy, SPY, equal-weight, and null band."""
    strat_ret = build_confidence_weighted_portfolio(
        signals, prices, backtest_start, backtest_end
    )
    strat_eq = _portfolio_equity_curve(strat_ret)

    if strat_eq.empty:
        return strat_eq, []

    # SPY
    spy_eq = None
    if spy is not None:
        spy_win = spy.loc[(spy.index >= backtest_start) & (spy.index <= backtest_end)]
        if not spy_win.empty:
            spy_ret = spy_win.pct_change().fillna(0.0)
            spy_eq = _portfolio_equity_curve(spy_ret)

    # Equal-weight universe
    ew_eq = None
    universe_tickers = [t for t in signals["ticker"].unique() if t in prices]
    if universe_tickers:
        eq_df = pd.DataFrame({t: prices[t].pct_change() for t in universe_tickers})
        eq_df = eq_df.loc[(eq_df.index >= backtest_start) & (eq_df.index <= backtest_end)].fillna(0.0)
        ew_ret = eq_df.mean(axis=1)
        ew_eq = _portfolio_equity_curve(ew_ret)

    # Random-shuffle null
    null_curves, null_sharpes = _random_shuffle_null_curves(
        signals, prices, backtest_start, backtest_end, n_shuffles
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    # Null band (shaded 5-95)
    if null_curves.size:
        lo = np.percentile(null_curves, 5, axis=0)
        hi = np.percentile(null_curves, 95, axis=0)
        ref_idx = strat_eq.index[: null_curves.shape[1]]
        ax.fill_between(ref_idx, lo, hi, color=COLORS["null_band"], alpha=0.45, label=f"random-shuffle 5-95% (n={len(null_sharpes)})")

    ax.plot(strat_eq.index, strat_eq.values, color=COLORS["strategy"], linewidth=2.2, label="confidence-weighted portfolio")
    if spy_eq is not None:
        ax.plot(spy_eq.index, spy_eq.values, color=COLORS["spy"], linewidth=1.5, linestyle="--", label="SPY buy-and-hold")
    if ew_eq is not None:
        ax.plot(ew_eq.index, ew_eq.values, color=COLORS["equal_weight"], linewidth=1.5, linestyle="--", label="equal-weight universe")

    ax.axhline(1.0, color="black", linewidth=0.6, alpha=0.4)
    ax.set_ylabel("Cumulative return (indexed to 1.0)")
    ax.set_xlabel("Date")
    ax.set_title("Phase 6 — Cumulative return: confidence-weighted portfolio vs benchmarks vs null")

    # Endpoint labels
    def _ann(series, label, color):
        if series is None or series.empty:
            return
        ax.annotate(
            f"{label} {series.iloc[-1] - 1:+.1%}",
            xy=(series.index[-1], series.iloc[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=8,
            color=color,
            va="center",
        )

    _ann(strat_eq, "strategy", COLORS["strategy"])
    _ann(spy_eq, "SPY", COLORS["spy"])
    _ann(ew_eq, "EW", COLORS["equal_weight"])

    ax.legend(loc="upper left", fontsize=9)
    ax.grid(linestyle=":", alpha=0.4)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(30)
        lbl.set_ha("right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")
    return strat_ret, null_sharpes


def plot_null_histogram(
    real_sharpe: float,
    null_sharpes: list[float],
    out_path: Path,
) -> None:
    """Histogram of the random-shuffle null Sharpes with the real strategy marked."""
    if not null_sharpes:
        return
    null_arr = np.array([s for s in null_sharpes if np.isfinite(s)])
    if null_arr.size == 0:
        return
    percentile = float((null_arr < real_sharpe).mean() * 100)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(null_arr, bins=40, color=COLORS["null_band"], edgecolor="black", alpha=0.85)
    ax.axvline(real_sharpe, color=COLORS["strategy"], linewidth=2.2, label=f"real Sharpe = {real_sharpe:.3f} (pctile {percentile:.0f})")
    ax.axvline(np.median(null_arr), color="black", linewidth=1.0, linestyle="--", alpha=0.7, label=f"null median = {np.median(null_arr):.3f}")
    ax.axvline(np.percentile(null_arr, 95), color="red", linewidth=1.0, linestyle="--", alpha=0.7, label=f"null p95 = {np.percentile(null_arr, 95):.3f}")
    ax.set_xlabel("Annualized Sharpe ratio")
    ax.set_ylabel(f"Frequency (n={null_arr.size})")
    ax.set_title("Phase 6 — Random-shuffle null distribution of Sharpe")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Build backtest plots.")
    parser.add_argument("--signals", default="data/backtest/signals.parquet")
    parser.add_argument("--market-dir", default="data/market")
    parser.add_argument("--out-dir", default="tests/metric3/outputs/plots")
    parser.add_argument("--backtest-start", default="2024-10-01")
    parser.add_argument("--backtest-end", default="2026-04-11")
    parser.add_argument("--shuffles", type=int, default=500)
    args = parser.parse_args()

    out_dir = ROOT_DIR / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    signals_path = ROOT_DIR / args.signals
    market_dir = ROOT_DIR / args.market_dir

    print(f"Signals:  {signals_path}")
    print(f"Market:   {market_dir}")
    print(f"Out:      {out_dir}")
    print()

    signals = load_signals(signals_path)
    if signals.empty:
        print("No usable signals.")
        return
    print(f"Loaded {len(signals)} signals across {signals['ticker'].nunique()} tickers")

    tickers = sorted(signals["ticker"].unique().tolist())
    prices = load_prices(market_dir, tickers)
    spy = load_benchmark(market_dir, "SPY")

    backtest_start = pd.Timestamp(args.backtest_start)
    backtest_end = pd.Timestamp(args.backtest_end)

    # Phase 4
    signals_fwd = attach_forward_returns(signals, prices)
    plot_hit_rate(signals_fwd, out_dir / "phase4_hit_rate.png")
    plot_forward_return_violin(signals_fwd, out_dir / "phase4_forward_return_violin.png", horizon="6m")

    # Phase 5
    plot_equity_grid(signals, prices, out_dir / "phase5_equity_grid.png", backtest_start, backtest_end)
    plot_alpha_bars(signals, prices, out_dir / "phase5_alpha_bars.png", backtest_start, backtest_end)

    # Phase 6
    strat_ret, null_sharpes = plot_cumulative_returns(
        signals, prices, spy, out_dir / "phase6_cumulative_returns.png",
        backtest_start, backtest_end, n_shuffles=args.shuffles,
    )
    real_sharpe = _annualize_sharpe(strat_ret)
    plot_null_histogram(real_sharpe, null_sharpes, out_dir / "phase6_null_histogram.png")

    print()
    print("All plots written.")


if __name__ == "__main__":
    main()
