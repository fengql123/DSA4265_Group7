"""Document indexing library.

Provides a DocumentIndexer class that chunks text, generates embeddings,
and upserts to ChromaDB. Owns the embedding model (loaded once, reused).

Usage:
    from src.rag.indexer import DocumentIndexer

    indexer = DocumentIndexer()

    # Index raw text with custom metadata
    indexer.index_documents(
        texts=["Revenue grew 12% YoY..."],
        collection_name="sec_filings",
        metadata=[{"ticker": "AAPL", "date": "2024-10-30", "section": "MD&A"}],
    )

    # Index files from disk
    indexer.index_files(
        file_paths=["data/sec/AAPL/10-K_2024.txt"],
        collection_name="sec_filings",
        metadata_fn=lambda path: {"ticker": "AAPL", "doc_type": "10-K"},
    )
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Callable

from src.config import get_embedding_model, get_rag_config
from src.rag.store import VectorStore, get_default_store



class DocumentIndexer:
    """Chunks text, embeds it, and upserts into ChromaDB.

    Holds a reference to the embedding model (loaded once on first use)
    and the vector store. All chunking parameters come from config but
    can be overridden per call.

    Args:
        store: VectorStore instance. Uses the default singleton if None.
        embedding_model: Pre-loaded SentenceTransformer. Loads from config if None.
        chunk_size: Default chunk size. Falls back to config if None.
        chunk_overlap: Default chunk overlap. Falls back to config if None.
    """

    def __init__(
        self,
        store: VectorStore | None = None,
        embedding_model=None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        self._store = store or get_default_store()
        self._embedding_model = embedding_model
        cfg = get_rag_config()
        self._default_chunk_size = chunk_size or cfg["chunk_size"]
        self._default_chunk_overlap = chunk_overlap or cfg["chunk_overlap"]

    @property
    def embedding_model(self):
        """Lazy-load the embedding model on first access."""
        if self._embedding_model is None:
            self._embedding_model = get_embedding_model()
        return self._embedding_model

    @property
    def store(self) -> VectorStore:
        return self._store

    @staticmethod
    def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
        """Split text into overlapping chunks using LlamaIndex SentenceSplitter."""
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.core.schema import Document

        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        doc = Document(text=text)
        nodes = splitter.get_nodes_from_documents([doc])
        return [node.get_content() for node in nodes]

    @staticmethod
    def _generate_id(doc_index: int, chunk_index: int, text: str) -> str:
        """Generate a deterministic ID for a chunk."""
        content = f"{doc_index}:{chunk_index}:{text}"
        return hashlib.md5(content.encode()).hexdigest()

    def index_documents(
        self,
        texts: list[str],
        collection_name: str,
        metadata: list[dict] | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> int:
        """Chunk texts, embed, and upsert to ChromaDB.

        Args:
            texts: List of document texts to index.
            collection_name: ChromaDB collection to upsert into.
            metadata: Per-document metadata dicts. Must match len(texts) if provided.
                Each chunk inherits its parent document's metadata.
            chunk_size: Override default chunk size.
            chunk_overlap: Override default chunk overlap.

        Returns:
            Number of chunks indexed.
        """
        chunk_size = chunk_size or self._default_chunk_size
        chunk_overlap = chunk_overlap or self._default_chunk_overlap

        if metadata and len(metadata) != len(texts):
            raise ValueError(
                f"metadata length ({len(metadata)}) must match texts length ({len(texts)})"
            )

        # Chunk all documents
        all_chunks: list[str] = []
        all_metadata: list[dict] = []
        all_ids: list[str] = []

        for i, text in enumerate(texts):
            doc_meta = metadata[i] if metadata else {}
            chunks = self._chunk_text(text, chunk_size, chunk_overlap)

            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({**doc_meta, "chunk_index": chunk_idx})
                all_ids.append(self._generate_id(i, chunk_idx, chunk))

        if not all_chunks:
            print("No chunks to index.")
            return 0

        # Generate embeddings
        embeddings = self.embedding_model.encode(
            all_chunks, show_progress_bar=len(all_chunks) > 100
        )
        embeddings_list = embeddings.tolist()

        # Upsert to ChromaDB
        collection = self._store.get_collection(collection_name)
        batch_size = 500
        for start in range(0, len(all_chunks), batch_size):
            end = min(start + batch_size, len(all_chunks))
            collection.upsert(
                ids=all_ids[start:end],
                documents=all_chunks[start:end],
                embeddings=embeddings_list[start:end],
                metadatas=all_metadata[start:end],
            )

        print(f"Indexed {len(all_chunks)} chunks into '{collection_name}'")
        return len(all_chunks)

    def index_files(
        self,
        file_paths: list[str | Path],
        collection_name: str,
        metadata_fn: Callable[[Path], dict] | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> int:
        """Read files from disk and index them.

        Args:
            file_paths: Paths to text files to index.
            collection_name: ChromaDB collection name.
            metadata_fn: Function that takes a file Path and returns metadata dict.
                If None, metadata will be {"source_file": filename}.
            chunk_size: Override default chunk size.
            chunk_overlap: Override default chunk overlap.

        Returns:
            Number of chunks indexed.
        """
        texts = []
        metadata = []

        for fp in file_paths:
            path = Path(fp)
            if not path.exists():
                print(f"  Skipping {path} (not found)")
                continue

            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = path.read_text(encoding="latin-1")

            if not text.strip():
                print(f"  Skipping {path} (empty)")
                continue

            texts.append(text)
            if metadata_fn:
                metadata.append(metadata_fn(path))
            else:
                metadata.append({"source_file": path.name})

        if not texts:
            print("No files to index.")
            return 0

        return self.index_documents(
            texts=texts,
            collection_name=collection_name,
            metadata=metadata,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )


# ---------------------------------------------------------------------------
# Module-level singleton + convenience functions (backwards compatible)
# ---------------------------------------------------------------------------

_default_indexer: DocumentIndexer | None = None


def get_default_indexer() -> DocumentIndexer:
    """Get or create the default DocumentIndexer singleton."""
    global _default_indexer
    if _default_indexer is None:
        _default_indexer = DocumentIndexer()
    return _default_indexer


def index_documents(
    texts: list[str],
    collection_name: str,
    metadata: list[dict] | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> int:
    return get_default_indexer().index_documents(
        texts, collection_name, metadata, chunk_size, chunk_overlap
    )


def index_files(
    file_paths: list[str | Path],
    collection_name: str,
    metadata_fn: Callable[[Path], dict] | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> int:
    return get_default_indexer().index_files(
        file_paths, collection_name, metadata_fn, chunk_size, chunk_overlap
    )
