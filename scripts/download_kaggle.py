#!/usr/bin/env python3
"""Generic Kaggle dataset downloader.

Downloads any Kaggle dataset or competition data to a local directory.
Requires ~/.kaggle/kaggle.json credentials (see https://www.kaggle.com/docs/api).

Usage:
    # Download a dataset
    python scripts/download_kaggle.py --dataset "jacksoncrow/stock-market-dataset" --output data/kaggle/stock-market

    # Download competition data
    python scripts/download_kaggle.py --competition "stock-market-prediction" --output data/kaggle/competition

    # Download specific files from a dataset
    python scripts/download_kaggle.py --dataset "owner/name" --output data/kaggle/name --files "file1.csv,file2.csv"
"""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path


def download_kaggle_dataset(
    dataset: str | None = None,
    competition: str | None = None,
    output_dir: str = "data/kaggle",
    files: list[str] | None = None,
    unzip: bool = True,
) -> Path:
    """Download a Kaggle dataset or competition data.

    Args:
        dataset: Kaggle dataset identifier (e.g. "owner/dataset-name").
        competition: Kaggle competition name (mutually exclusive with dataset).
        output_dir: Local directory to save the data.
        files: Specific files to download. None downloads all files.
        unzip: Whether to unzip downloaded files.

    Returns:
        Path to the output directory.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise ImportError(
            "kaggle package not installed. Run: pip install kaggle\n"
            "Also ensure ~/.kaggle/kaggle.json exists with your credentials."
        )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    if dataset:
        print(f"Downloading Kaggle dataset: {dataset}")
        if files:
            for f in files:
                api.dataset_download_file(dataset, f, path=str(out_path))
                print(f"  Downloaded {f}")
        else:
            api.dataset_download_files(dataset, path=str(out_path), unzip=unzip)
            print(f"  Downloaded all files")

    elif competition:
        print(f"Downloading Kaggle competition data: {competition}")
        if files:
            for f in files:
                api.competition_download_file(competition, f, path=str(out_path))
                print(f"  Downloaded {f}")
        else:
            api.competition_download_files(competition, path=str(out_path))
            print(f"  Downloaded all files")

            # competition_download_files doesn't have unzip param, do it manually
            if unzip:
                for zip_file in out_path.glob("*.zip"):
                    with zipfile.ZipFile(zip_file) as zf:
                        zf.extractall(out_path)
                    zip_file.unlink()
                    print(f"  Unzipped {zip_file.name}")
    else:
        raise ValueError("Must provide either --dataset or --competition")

    print(f"Saved to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Download a Kaggle dataset or competition data."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", help="Kaggle dataset identifier (owner/name)")
    group.add_argument("--competition", help="Kaggle competition name")
    parser.add_argument("--output", required=True, help="Output directory path")
    parser.add_argument(
        "--files",
        default=None,
        help="Comma-separated list of specific files to download",
    )
    parser.add_argument(
        "--no-unzip", action="store_true", help="Don't unzip downloaded files"
    )

    args = parser.parse_args()
    files = args.files.split(",") if args.files else None

    download_kaggle_dataset(
        dataset=args.dataset,
        competition=args.competition,
        output_dir=args.output,
        files=files,
        unzip=not args.no_unzip,
    )


if __name__ == "__main__":
    main()
