import pandas as pd
import matplotlib.pyplot as plt
import os

# confif
RESULTS_PATH = "results/time_series_results.csv"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


def main():
    df = pd.read_csv(RESULTS_PATH)

    print("\nLoaded results:")
    print(df)

    plt.figure()
    df.set_index("Ticker")[["Return (%)", "Buy & Hold (%)"]].plot(kind="bar")
    plt.title("Strategy vs Buy & Hold Returns")
    plt.ylabel("Return (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/returns_comparison.png")
    plt.close()

    plt.figure()
    df.set_index("Ticker")["Sharpe"].plot(kind="bar")
    plt.title("Sharpe Ratio by Ticker")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/sharpe.png")
    plt.close()

    plt.figure()
    df.set_index("Ticker")["Max Drawdown (%)"].plot(kind="bar")
    plt.title("Max Drawdown by Ticker")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/drawdown.png")
    plt.close()

    print(f"\nPlots saved in: {PLOTS_DIR}")

if __name__ == "__main__":
    main()
