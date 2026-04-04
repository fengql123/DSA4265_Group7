#!/usr/bin/env python3
"""CLI entrypoint for ingesting data into ChromaDB.

Reads files from disk, attaches user-defined metadata, and calls
src/rag/indexer.py to chunk, embed, and upsert.

Usage:
    # Ingest a directory with explicit metadata
    python scripts/ingest.py \
        --input-dir data/sec/AAPL/ \
        --collection sec_filings \
        --metadata '{"ticker": "AAPL", "doc_type": "10-K"}'

    # Ingest with metadata extracted from directory structure
    python scripts/ingest.py \
        --input-dir data/sec/ \
        --collection sec_filings \
        --metadata '{"doc_type": "sec_filing"}' \
        --ticker-from-dir

    # Batch ingestion via YAML manifest
    python scripts/ingest.py --manifest config/ingest_manifest.yaml

    # Custom chunk size
    python scripts/ingest.py \
        --input-dir data/sec/AAPL/ \
        --collection sec_filings \
        --metadata '{"ticker": "AAPL"}' \
        --chunk-size 1024 --chunk-overlap 128
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path so we can import src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src.rag.indexer import index_files


def _collect_files(input_dir: Path, extensions: list[str] | None = None) -> list[Path]:
    """Recursively collect text files from a directory."""
    extensions = extensions or [".txt", ".md", ".text", ".html"]
    files = []
    for ext in extensions:
        files.extend(input_dir.rglob(f"*{ext}"))
    return sorted(files)


def _make_metadata_fn(
    base_metadata: dict,
    ticker_from_dir: bool = False,
    input_dir: Path | None = None,
) -> callable:
    """Create a metadata function for index_files().

    Args:
        base_metadata: Static metadata to attach to all chunks.
        ticker_from_dir: If True, extract ticker from the immediate parent directory name.
        input_dir: The base input directory (used for path-relative metadata).
    """
    def metadata_fn(file_path: Path) -> dict:
        meta = {**base_metadata, "source_file": file_path.name}

        if ticker_from_dir:
            # Use the immediate parent directory name as ticker
            meta["ticker"] = file_path.parent.name

        return meta

    return metadata_fn


def ingest_directory(
    input_dir: str,
    collection: str,
    metadata: dict | None = None,
    ticker_from_dir: bool = False,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    extensions: list[str] | None = None,
) -> int:
    """Ingest all text files from a directory into ChromaDB.

    Args:
        input_dir: Directory containing files to ingest.
        collection: ChromaDB collection name.
        metadata: Static metadata dict to attach to all chunks.
        ticker_from_dir: Extract ticker from parent directory name.
        chunk_size: Override chunk size.
        chunk_overlap: Override chunk overlap.
        extensions: File extensions to include.

    Returns:
        Number of chunks indexed.
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Error: Directory not found: {input_path}")
        return 0

    files = _collect_files(input_path, extensions)
    if not files:
        print(f"No files found in {input_path}")
        return 0

    print(f"Found {len(files)} files in {input_path}")

    metadata_fn = _make_metadata_fn(
        base_metadata=metadata or {},
        ticker_from_dir=ticker_from_dir,
        input_dir=input_path,
    )

    return index_files(
        file_paths=files,
        collection_name=collection,
        metadata_fn=metadata_fn,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def ingest_from_manifest(manifest_path: str) -> int:
    """Run batch ingestion from a YAML manifest file.

    The manifest defines multiple data sources, each mapping to a
    ChromaDB collection with specified metadata rules.

    See config/ingest_manifest.yaml for format.
    """
    manifest = Path(manifest_path)
    if not manifest.exists():
        print(f"Error: Manifest not found: {manifest}")
        return 0

    with open(manifest) as f:
        config = yaml.safe_load(f)

    total_chunks = 0
    sources = config.get("sources", [])

    for i, source in enumerate(sources):
        input_dir = source.get("input_dir")
        collection = source.get("collection")

        if not input_dir or not collection:
            print(f"Skipping source {i}: missing input_dir or collection")
            continue

        metadata = source.get("metadata_fields", {})

        # Check if we should extract ticker from directory structure
        metadata_from_path = source.get("metadata_from_path", {})
        ticker_from_dir = "ticker" in metadata_from_path and \
                          metadata_from_path["ticker"] == "{parent_dir}"

        print(f"\n{'='*60}")
        print(f"Ingesting: {input_dir} -> {collection}")

        chunks = ingest_directory(
            input_dir=input_dir,
            collection=collection,
            metadata=metadata,
            ticker_from_dir=ticker_from_dir,
        )
        total_chunks += chunks

    print(f"\n{'='*60}")
    print(f"Total: {total_chunks} chunks indexed from {len(sources)} sources")
    return total_chunks


def main():
    parser = argparse.ArgumentParser(
        description="Ingest data into ChromaDB for RAG retrieval."
    )

    # Two modes: direct or manifest
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--manifest", help="Path to YAML ingest manifest file")
    mode.add_argument("--input-dir", help="Directory containing files to ingest")

    # Direct mode options
    parser.add_argument("--collection", help="ChromaDB collection name (required with --input-dir)")
    parser.add_argument(
        "--metadata",
        default=None,
        help='JSON metadata string (e.g. \'{"ticker": "AAPL", "doc_type": "10-K"}\')',
    )
    parser.add_argument(
        "--ticker-from-dir",
        action="store_true",
        help="Extract ticker from parent directory name",
    )
    parser.add_argument("--chunk-size", type=int, default=None, help="Override chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=None, help="Override chunk overlap")
    parser.add_argument(
        "--extensions",
        default=None,
        help="Comma-separated file extensions (default: .txt,.md,.text,.html)",
    )

    args = parser.parse_args()

    if args.manifest:
        ingest_from_manifest(args.manifest)
    else:
        if not args.collection:
            parser.error("--collection is required when using --input-dir")

        metadata = json.loads(args.metadata) if args.metadata else None
        extensions = args.extensions.split(",") if args.extensions else None

        ingest_directory(
            input_dir=args.input_dir,
            collection=args.collection,
            metadata=metadata,
            ticker_from_dir=args.ticker_from_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            extensions=extensions,
        )


if __name__ == "__main__":
    main()
