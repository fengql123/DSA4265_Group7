#!/usr/bin/env python3
"""Generic HuggingFace dataset downloader.

Downloads any HuggingFace dataset to a local directory as parquet files.
Users process the raw data themselves for their specific use case.

Usage:
    # Basic download
    python scripts/download_hf.py --dataset "lamini/earnings-calls-qa" --output data/hf/earnings-calls-qa

    # With split and column selection
    python scripts/download_hf.py --dataset "takala/financial_phrasebank" --output data/hf/phrasebank --split train --columns sentence,label

    # Large dataset with streaming (saves to disk incrementally)
    python scripts/download_hf.py --dataset "Brianferrell787/financial-news-multisource" --output data/hf/news --stream --max-rows 100000

    # With config/subset name
    python scripts/download_hf.py --dataset "takala/financial_phrasebank" --config "sentences_allagree" --output data/hf/phrasebank

Relevant datasets for this project:
    - lamini/earnings-calls-qa                        Earnings transcripts + QA
    - kurry/sp500_earnings_transcripts                S&P 500 earnings
    - Brianferrell787/financial-news-multisource       57M+ financial news rows
    - takala/financial_phrasebank                      Sentiment training data
    - TheFinAI/fiqa-sentiment-classification           FiQA sentiment
    - nlpaueb/finer-139                                NER training data
    - PatronusAI/financebench                          Evaluation QA pairs
    - ibm/finqa                                        Numerical QA eval
    - jlh-ibm/earnings_call                            Earnings + price reactions
    - JanosAudran/financial-reports-sec                SEC filings + market labels
"""

from __future__ import annotations

import argparse
from pathlib import Path


def download_dataset(
    dataset_name: str,
    output_dir: str,
    split: str | None = None,
    config: str | None = None,
    columns: list[str] | None = None,
    stream: bool = False,
    max_rows: int | None = None,
    output_format: str = "parquet",
) -> Path:
    """Download a HuggingFace dataset to a local directory.

    Args:
        dataset_name: HuggingFace dataset identifier (e.g. "lamini/earnings-calls-qa").
        output_dir: Local directory to save the data.
        split: Dataset split to download (e.g. "train", "test"). None downloads all splits.
        config: Dataset configuration/subset name.
        columns: List of column names to keep. None keeps all columns.
        stream: If True, use streaming mode for large datasets.
        max_rows: Maximum number of rows to download (only used with stream=True).
        output_format: Output format — "parquet" or "csv".

    Returns:
        Path to the output directory.
    """
    from datasets import load_dataset

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {dataset_name}...")

    kwargs = {}
    if config:
        kwargs["name"] = config
    if split:
        kwargs["split"] = split

    if stream:
        kwargs["streaming"] = True
        ds = load_dataset(dataset_name, **kwargs)

        # For streaming, iterate and collect rows
        if split:
            _stream_to_file(ds, out_path, split, columns, max_rows, output_format)
        else:
            # When no split specified, streaming returns IterableDatasetDict
            for split_name in ds:
                _stream_to_file(
                    ds[split_name], out_path, split_name, columns, max_rows, output_format
                )
    else:
        ds = load_dataset(dataset_name, **kwargs)

        if split:
            # ds is a single Dataset
            if columns:
                ds = ds.select_columns(columns)
            _save_dataset(ds, out_path, split, output_format)
        else:
            # ds is a DatasetDict with multiple splits
            for split_name in ds:
                subset = ds[split_name]
                if columns:
                    subset = subset.select_columns(columns)
                _save_dataset(subset, out_path, split_name, output_format)

    print(f"Saved to {out_path}")
    return out_path


def _save_dataset(dataset, out_path: Path, split_name: str, fmt: str):
    """Save a Dataset to disk."""
    if fmt == "parquet":
        dataset.to_parquet(out_path / f"{split_name}.parquet")
    elif fmt == "csv":
        dataset.to_csv(out_path / f"{split_name}.csv")
    else:
        raise ValueError(f"Unknown format: {fmt}")
    print(f"  Saved {split_name}: {len(dataset)} rows")


def _stream_to_file(
    iterable_dataset,
    out_path: Path,
    split_name: str,
    columns: list[str] | None,
    max_rows: int | None,
    fmt: str,
):
    """Stream an IterableDataset to a file."""
    import pandas as pd

    rows = []
    for i, row in enumerate(iterable_dataset):
        if max_rows and i >= max_rows:
            break
        if columns:
            row = {k: v for k, v in row.items() if k in columns}
        rows.append(row)

        if (i + 1) % 10000 == 0:
            print(f"  Streamed {i + 1} rows...")

    df = pd.DataFrame(rows)
    if fmt == "parquet":
        df.to_parquet(out_path / f"{split_name}.parquet", index=False)
    elif fmt == "csv":
        df.to_csv(out_path / f"{split_name}.csv", index=False)

    print(f"  Saved {split_name}: {len(df)} rows")


def main():
    parser = argparse.ArgumentParser(
        description="Download a HuggingFace dataset to a local directory."
    )
    parser.add_argument(
        "--dataset", required=True, help="HuggingFace dataset identifier"
    )
    parser.add_argument(
        "--output", required=True, help="Output directory path"
    )
    parser.add_argument("--split", default=None, help="Dataset split (train/test/etc)")
    parser.add_argument("--config", default=None, help="Dataset config/subset name")
    parser.add_argument(
        "--columns",
        default=None,
        help="Comma-separated list of columns to keep",
    )
    parser.add_argument(
        "--stream", action="store_true", help="Use streaming mode for large datasets"
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Max rows to download (streaming only)",
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "csv"],
        default="parquet",
        help="Output file format (default: parquet)",
    )

    args = parser.parse_args()
    columns = args.columns.split(",") if args.columns else None

    download_dataset(
        dataset_name=args.dataset,
        output_dir=args.output,
        split=args.split,
        config=args.config,
        columns=columns,
        stream=args.stream,
        max_rows=args.max_rows,
        output_format=args.format,
    )


if __name__ == "__main__":
    main()
