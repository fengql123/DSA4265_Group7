"""Technical indicators tooling.

Provides two tools for the TechnicalAgent:

- `get_price_history`: daily OHLCV bars for a ticker, optionally as-of a
  past date. Falls back from the local historical CSV (data/market) to
  live yfinance.
- `compute_technical_indicators`: latest-value snapshot of the standard
  indicator panel (RSI, MACD, Bollinger, ATR, Stochastic, ADX, OBV,
  SMAs) plus short human-readable trend descriptors.

Both tools share a single OHLCV loader so historical / backtest mode
behaves identically across them.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.tools.base_tool import BaseTool


MARKET_DATA_DIR = Path("data/market")


# ---------------------------------------------------------------------------
# Shared OHLCV loader — historical CSV preferred, yfinance fallback
# ---------------------------------------------------------------------------


def _parse_end_date(end_date: str | None) -> datetime:
    if not end_date:
        return datetime.now()
    return datetime.strptime(end_date, "%Y-%m-%d")


def _normalize_ohlcv_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a tz-naive DataFrame with columns `open, high, low, close, volume` and a plain DatetimeIndex."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = df.copy()

    # yfinance CSV has an index column named "Date" or "Datetime".
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")

    # Strip tz for comparison convenience.
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Normalize column names (case-insensitive match).
    rename_map = {}
    for col in df.columns:
        lc = col.lower()
        if lc in {"open", "high", "low", "close", "volume"}:
            rename_map[col] = lc
    df = df.rename(columns=rename_map)
    needed = ["open", "high", "low", "close", "volume"]
    df = df[[c for c in needed if c in df.columns]].dropna(how="all")
    return df.sort_index()


def _load_from_csv(ticker: str, end_date: datetime, lookback_days: int) -> pd.DataFrame | None:
    """Read OHLCV from data/market/{TICKER}.csv if available.

    Returns a slice ending at `end_date` with at most `lookback_days` bars.
    Returns `None` when the file is missing or when `end_date` falls
    outside the CSV's range (we prefer to fall back to live yfinance
    rather than truncate silently).
    """
    csv_path = MARKET_DATA_DIR / f"{ticker.upper()}.csv"
    if not csv_path.exists():
        return None

    try:
        raw = pd.read_csv(csv_path, index_col=0)
    except Exception:
        return None

    df = _normalize_ohlcv_frame(raw)
    if df.empty:
        return None

    # Slice to <= end_date.
    df = df[df.index <= pd.Timestamp(end_date)]
    if df.empty:
        return None

    # Keep at most lookback_days bars ending at end_date.
    df = df.tail(int(lookback_days))
    return df


def _load_from_yfinance(ticker: str, end_date: datetime, lookback_days: int) -> pd.DataFrame:
    import yfinance as yf

    # Buffer calendar days generously to account for weekends, holidays.
    buffer = max(lookback_days * 2, 60)
    start = (end_date - timedelta(days=buffer)).strftime("%Y-%m-%d")
    end_yf = (end_date + timedelta(days=1)).strftime("%Y-%m-%d")

    raw = yf.Ticker(ticker).history(start=start, end=end_yf, auto_adjust=False)
    df = _normalize_ohlcv_frame(raw)
    if df.empty:
        return df
    df = df[df.index <= pd.Timestamp(end_date)]
    return df.tail(int(lookback_days))


def load_ohlcv(ticker: str, end_date_str: str | None, lookback_days: int) -> pd.DataFrame:
    """Preferred loader: CSV first (deterministic backtests), else live yfinance."""
    end_dt = _parse_end_date(end_date_str)
    csv_df = _load_from_csv(ticker, end_dt, lookback_days)
    if csv_df is not None and len(csv_df) > 0:
        return csv_df
    return _load_from_yfinance(ticker, end_dt, lookback_days)


# ---------------------------------------------------------------------------
# Indicator math (plain pandas/numpy — no TA-lib dependency)
# ---------------------------------------------------------------------------


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    # Wilder smoothing via ewm alpha=1/period
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> dict[str, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return {"macd": macd_line, "signal": signal_line, "hist": hist}


def _bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0) -> dict[str, pd.Series]:
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    return {
        "middle": sma,
        "upper": sma + num_std * std,
        "lower": sma - num_std * std,
    }


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = _true_range(high, low, close)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3
) -> dict[str, pd.Series]:
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    pct_k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    pct_d = pct_k.rolling(d_period).mean()
    return {"k": pct_k, "d": pct_d}


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index
    )
    tr = _true_range(high, low, close)
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()


def _round(value: Any, digits: int = 4) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Tool: get_price_history
# ---------------------------------------------------------------------------


class GetPriceHistoryInput(BaseModel):
    ticker: str = Field(description="Stock ticker symbol (e.g. 'AAPL').")
    end_date: str = Field(
        description="Analysis end date in YYYY-MM-DD format. The series ends on or before this date."
    )
    lookback_days: int = Field(
        default=400,
        description="Number of trading-day bars to return ending at end_date. Default 400 (~18 months).",
    )


class GetPriceHistoryTool(BaseTool):
    name = "get_price_history"
    description = (
        "**What**: Returns a daily OHLCV bar series for one ticker, ending at a specific date. "
        "**When to use**: When you need raw price action for trend, support/resistance, or indicator computation over months. "
        "For only summary stats (price, margins, P/E) use `get_market_data` instead. "
        "**Input**: "
        "`ticker` (str); "
        "`end_date` (str, YYYY-MM-DD — the last bar to include, inclusive); "
        "`lookback_days` (int, default 400 ≈ 18 months of trading days). "
        "**Output**: JSON with `ticker`, `start_date`, `end_date`, `bars` (number of rows returned), "
        "`first_close`, `last_close`, `period_return_pct`, `period_high`, `period_low`, and a compact `series` "
        "of the last 10 bars (each a dict of `date, open, high, low, close, volume`). "
        "**Limits**: Uses `data/market/{TICKER}.csv` if present and the end_date falls inside its range; "
        "otherwise falls back to live yfinance. Returns an error message if no data is available for the ticker."
    )
    input_schema = GetPriceHistoryInput

    def execute(self, ticker: str, end_date: str, lookback_days: int = 400) -> str:
        try:
            df = load_ohlcv(ticker, end_date, lookback_days)
        except Exception as exc:
            return f"get_price_history error: {exc}"

        if df is None or df.empty:
            return f"No OHLCV data available for {ticker} on or before {end_date}."

        first = df.iloc[0]
        last = df.iloc[-1]
        first_close = float(first["close"])
        last_close = float(last["close"])
        period_return = (last_close / first_close - 1) * 100 if first_close else None

        tail = df.tail(10)
        series = [
            {
                "date": idx.strftime("%Y-%m-%d"),
                "open": _round(row["open"], 4),
                "high": _round(row["high"], 4),
                "low": _round(row["low"], 4),
                "close": _round(row["close"], 4),
                "volume": int(row["volume"]) if not pd.isna(row["volume"]) else None,
            }
            for idx, row in tail.iterrows()
        ]

        result = {
            "ticker": ticker.upper(),
            "start_date": df.index[0].strftime("%Y-%m-%d"),
            "end_date": df.index[-1].strftime("%Y-%m-%d"),
            "bars": int(len(df)),
            "first_close": _round(first_close),
            "last_close": _round(last_close),
            "period_return_pct": _round(period_return, 2),
            "period_high": _round(df["high"].max()),
            "period_low": _round(df["low"].min()),
            "series_tail_10": series,
        }
        return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Tool: compute_technical_indicators
# ---------------------------------------------------------------------------


class ComputeTechnicalIndicatorsInput(BaseModel):
    ticker: str = Field(description="Stock ticker symbol (e.g. 'AAPL').")
    end_date: str = Field(
        description="Analysis end date in YYYY-MM-DD format. Indicators reflect their value at this date."
    )
    lookback_days: int = Field(
        default=400,
        description="Number of trading-day bars to compute on. Default 400.",
    )


class ComputeTechnicalIndicatorsTool(BaseTool):
    name = "compute_technical_indicators"
    description = (
        "**What**: Computes a standard technical-indicator panel for one ticker ending at a specific date — "
        "RSI(14), MACD(12,26,9), Bollinger Bands(20,2), ATR(14), Stochastic(14,3), ADX(14), OBV, and 20/50/200-day SMAs. "
        "**When to use**: When you need numeric values of momentum, trend strength, volatility and moving averages to support a technical call. "
        "**Input**: "
        "`ticker` (str); "
        "`end_date` (str, YYYY-MM-DD); "
        "`lookback_days` (int, default 400 — the 200-day SMA + its lead-in requires at least 250 bars). "
        "**Output**: JSON with `ticker`, `as_of_date`, `close`, `sma20/sma50/sma200`, "
        "`rsi14` plus a regime descriptor ('oversold', 'neutral', 'overbought'), "
        "`macd` object (line, signal, histogram, crossover direction), "
        "`bollinger` object (upper, middle, lower, position relative to bands), "
        "`atr14`, `stochastic` object (k, d, regime), `adx14` plus trend strength descriptor, `obv`, "
        "and a top-level `trend_summary` string. "
        "**Limits**: Uses `data/market/{TICKER}.csv` if available, else yfinance. "
        "Fields that cannot be computed (insufficient data) are returned as `null`."
    )
    input_schema = ComputeTechnicalIndicatorsInput

    def execute(self, ticker: str, end_date: str, lookback_days: int = 400) -> str:
        try:
            df = load_ohlcv(ticker, end_date, lookback_days)
        except Exception as exc:
            return f"compute_technical_indicators error: {exc}"

        if df is None or df.empty:
            return f"No OHLCV data available for {ticker} on or before {end_date}."

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # Core indicators
        rsi = _rsi(close)
        macd = _macd(close)
        bb = _bollinger(close)
        atr = _atr(high, low, close)
        stoch = _stochastic(high, low, close)
        adx = _adx(high, low, close)
        obv = _obv(close, volume)

        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()

        last_close = float(close.iloc[-1])
        last_rsi = _round(rsi.iloc[-1], 2)
        last_macd_line = _round(macd["macd"].iloc[-1], 4)
        last_macd_signal = _round(macd["signal"].iloc[-1], 4)
        last_macd_hist = _round(macd["hist"].iloc[-1], 4)
        last_bb_upper = _round(bb["upper"].iloc[-1])
        last_bb_middle = _round(bb["middle"].iloc[-1])
        last_bb_lower = _round(bb["lower"].iloc[-1])
        last_atr = _round(atr.iloc[-1])
        last_stoch_k = _round(stoch["k"].iloc[-1], 2)
        last_stoch_d = _round(stoch["d"].iloc[-1], 2)
        last_adx = _round(adx.iloc[-1], 2)
        last_obv = _round(obv.iloc[-1], 0)
        last_sma20 = _round(sma20.iloc[-1])
        last_sma50 = _round(sma50.iloc[-1])
        last_sma200 = _round(sma200.iloc[-1])

        # Regime descriptors
        if last_rsi is None:
            rsi_regime = "unknown"
        elif last_rsi >= 70:
            rsi_regime = "overbought"
        elif last_rsi <= 30:
            rsi_regime = "oversold"
        else:
            rsi_regime = "neutral"

        if last_macd_line is None or last_macd_signal is None:
            macd_cross = "unknown"
        elif last_macd_line > last_macd_signal:
            macd_cross = "bullish (line above signal)"
        elif last_macd_line < last_macd_signal:
            macd_cross = "bearish (line below signal)"
        else:
            macd_cross = "flat"

        if None in (last_bb_upper, last_bb_lower):
            bb_position = "unknown"
        elif last_close >= last_bb_upper:
            bb_position = "above upper band (extended)"
        elif last_close <= last_bb_lower:
            bb_position = "below lower band (oversold)"
        else:
            bb_position = "inside bands"

        if last_stoch_k is None:
            stoch_regime = "unknown"
        elif last_stoch_k >= 80:
            stoch_regime = "overbought"
        elif last_stoch_k <= 20:
            stoch_regime = "oversold"
        else:
            stoch_regime = "neutral"

        if last_adx is None:
            adx_desc = "unknown"
        elif last_adx >= 25:
            adx_desc = "strong trend"
        elif last_adx >= 20:
            adx_desc = "developing trend"
        else:
            adx_desc = "weak / no clear trend"

        # High-level trend summary combining MAs and ADX
        trend_bits = []
        if last_sma50 is not None and last_sma200 is not None:
            if last_close > last_sma50 > last_sma200:
                trend_bits.append("price above 50DMA above 200DMA (uptrend)")
            elif last_close < last_sma50 < last_sma200:
                trend_bits.append("price below 50DMA below 200DMA (downtrend)")
            else:
                trend_bits.append("mixed MA stack (no clean trend)")
        if adx_desc != "unknown":
            trend_bits.append(adx_desc)
        if rsi_regime != "unknown":
            trend_bits.append(f"RSI={last_rsi} ({rsi_regime})")

        trend_summary = "; ".join(trend_bits) if trend_bits else "insufficient data for trend summary"

        result = {
            "ticker": ticker.upper(),
            "as_of_date": df.index[-1].strftime("%Y-%m-%d"),
            "bars_used": int(len(df)),
            "close": _round(last_close),
            "sma20": last_sma20,
            "sma50": last_sma50,
            "sma200": last_sma200,
            "rsi14": last_rsi,
            "rsi_regime": rsi_regime,
            "macd": {
                "line": last_macd_line,
                "signal": last_macd_signal,
                "histogram": last_macd_hist,
                "crossover": macd_cross,
            },
            "bollinger": {
                "upper": last_bb_upper,
                "middle": last_bb_middle,
                "lower": last_bb_lower,
                "position": bb_position,
            },
            "atr14": last_atr,
            "stochastic": {
                "k": last_stoch_k,
                "d": last_stoch_d,
                "regime": stoch_regime,
            },
            "adx14": last_adx,
            "adx_descriptor": adx_desc,
            "obv": last_obv,
            "trend_summary": trend_summary,
        }
        return json.dumps(result, indent=2)
