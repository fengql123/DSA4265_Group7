import pandas as pd
import numpy as np
import yfinance as yf
import json
import os
from datetime import datetime

DATA_PATH = "outputs_time_series"
START_DATE = "2023-01-01"
END_DATE = "2024-12-31"
INITIAL_CAPITAL = 10000

def parse_date(date_str):
    return datetime.strptime(date_str, "%B_%Y")

def load_time_series_signals(ticker):
    signals = []

    for file in os.listdir(DATA_PATH):
        if file.startswith(ticker):
            parts = file.replace(".json", "").split("_")
            date_str = "_".join(parts[1:])
            filepath = os.path.join(DATA_PATH, file)

            with open(filepath, "r") as f:
                data = json.load(f)

            rec = data.get("recommendation", "").lower()

            if rec == "buy":
                signal = 1
            elif rec == "sell":
                signal = -1
            else:
                signal = 0

            signals.append({
                "Date": parse_date(date_str),
                "Signal": signal
            })

    if not signals:
        return None

    df = pd.DataFrame(signals).sort_values("Date")
    df.set_index("Date", inplace=True)

    return df

# get price data
def get_price_data(ticker):
    data = yf.download(ticker, start=START_DATE, end=END_DATE)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

# apply signals
def apply_signals(price_data, signals):
    signals = signals.reindex(price_data.index, method='ffill')
    price_data['Signal'] = signals['Signal']
    price_data['Signal'].fillna(0, inplace=True)
    return price_data

# backtest engine
def run_backtest(data):
    data = data.copy()

    # daily returns
    data['Return'] = data['Close'].pct_change()

    # strategy returns
    data['Strategy_Return'] = data['Return'] * data['Signal'].shift(1)

    # equity curve
    data['Equity'] = (1 + data['Strategy_Return']).cumprod() * INITIAL_CAPITAL

    return data

def compute_metrics(data):
    returns = data['Strategy_Return'].dropna()

    # total return
    total_return = data['Equity'].iloc[-1] / INITIAL_CAPITAL - 1

    # Sharpe ratio
    if returns.std() != 0:
        sharpe = np.sqrt(252) * returns.mean() / returns.std()
    else:
        sharpe = 0

    # max drawdown
    equity = data['Equity']
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    max_dd = drawdown.min()

    return total_return, sharpe, max_dd

def buy_and_hold(prices):
    return prices['Close'].iloc[-1] / prices['Close'].iloc[0] - 1

def main():
    results = []

    tickers = list(set([f.split("_")[0] for f in os.listdir(DATA_PATH)]))

    for ticker in tickers:
        print(f"\nProcessing {ticker}...")

        signals = load_time_series_signals(ticker)
        if signals is None:
            print("No signals found")
            continue

        prices = get_price_data(ticker)
        data = apply_signals(prices, signals)

        data = run_backtest(data)

        total_return, sharpe, max_dd = compute_metrics(data)
        bh_return = buy_and_hold(prices)

        results.append({
            "Ticker": ticker,
            "Return (%)": total_return * 100,
            "Sharpe": sharpe,
            "Max Drawdown (%)": max_dd * 100,
            "Buy & Hold (%)": bh_return * 100
        })

    df = pd.DataFrame(results)

    print("\n=== TIME SERIES RESULTS ===")
    print(df)

    print("\n=== SUMMARY ===")
    print("Avg Return:", df["Return (%)"].mean())
    print("Avg Sharpe:", df["Sharpe"].mean())
    print("Avg Drawdown:", df["Max Drawdown (%)"].mean())
    print("Avg Buy & Hold:", df["Buy & Hold (%)"].mean())

    os.makedirs("results", exist_ok=True)
    df.to_csv("results/time_series_results.csv", index=False)


if __name__ == "__main__":
    main()
