import json
import time
import subprocess
from pathlib import Path
from src.graph import build_graph

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "JPM", "JNJ", "V",
    "PG", "UNH", "HD", "MA", "DIS",
    "PYPL", "NFLX", "ADBE", "INTC", "KO"
]

DATES = [
    "January 2023",
    "April 2023",
    "July 2023",
    "October 2023",
    "January 2024",
    "April 2024",
    "July 2024",
    "October 2024"
]

OUTPUT_DIR = Path("outputs_time_series")
OUTPUT_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 3

# download and ingest data
def prepare_data():
    print("\n=== Preparing Data ===\n")

    for ticker in TICKERS:
        print(f"\nPreparing {ticker}...")

        subprocess.run([
            "python", "scripts/download_demo_data.py",
            "--ticker", ticker
        ])

        subprocess.run([
            "python", "scripts/ingest_demo.py",
            "--ticker", ticker
        ])

    print("\n=== Data Preparation Complete ===\n")

def build_queries():
    queries = []

    for ticker in TICKERS:
        for date in DATES:
            queries.append({
                "query": f"Analyze {ticker} stock in {date}",
                "ticker": ticker,
                "date": date
            })

    return queries

def run_parallel():
    graph = build_graph(debug=False)
    queries = build_queries()

    print(f"\nTotal queries: {len(queries)}")
    print(f"Running in batches of {BATCH_SIZE}...\n")

    for i in range(0, len(queries), BATCH_SIZE):
        batch = queries[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1

        print(f"\n===== Batch {batch_num} =====")

        states = [{"query": q["query"], "errors": []} for q in batch]
        results = graph.batch(states)

        for meta, result in zip(batch, results):
            ticker = meta["ticker"]
            date = meta["date"].replace(" ", "_")

            filename = f"{ticker}_{date}.json"
            filepath = OUTPUT_DIR / filename

            # Skip if already exists
            if filepath.exists():
                print(f"Skipping {filename}")
                continue

            memo = result.get("investment_memo")

            if memo:
                with open(filepath, "w") as f:
                    json.dump(memo.model_dump(), f, indent=2, default=str)

                print(f"{ticker} ({meta['date']}) → {memo.recommendation.upper()} ({memo.confidence:.0%})")

            else:
                print(f"FAILED: {meta['query']}")
                print("DEBUG:", result)

        # Prevent API overload
        time.sleep(2)

    print("\n=== All batches completed ===")
    print(f"Outputs saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    prepare_data() 
    run_parallel()
