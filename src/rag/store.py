"""ChromaDB vector store wrapper.

Provides a VectorStore class that manages the ChromaDB persistent client
and collection access. Used by DocumentIndexer and Retriever.

Usage:
    from src.rag.store import VectorStore

    store = VectorStore()
    collection = store.get_collection("sec_filings")
    store.list_collections()
    store.delete_collection("sec_filings")
"""

from __future__ import annotations

import chromadb

from src.config import get_rag_config


class VectorStore:
    """Wrapper around ChromaDB PersistentClient.

    Manages the database connection and provides collection access.
    """

    def __init__(self, persist_dir: str | None = None):
        cfg = get_rag_config()
        self._persist_dir = persist_dir or cfg["chroma_persist_dir"]
        self._client: chromadb.ClientAPI | None = None

    @property
    def client(self) -> chromadb.ClientAPI:
        """Lazy-initialize the ChromaDB client."""
        if self._client is None:
            self._client = chromadb.PersistentClient(path=self._persist_dir)
        return self._client

    def get_collection(self, name: str) -> chromadb.Collection:
        """Get or create a ChromaDB collection by name."""
        return self.client.get_or_create_collection(name)

    def list_collections(self) -> list[str]:
        """List all collection names in the database."""
        return [c.name for c in self.client.list_collections()]

    def delete_collection(self, name: str) -> None:
        """Delete a collection by name."""
        self.client.delete_collection(name)


# ---------------------------------------------------------------------------
# Module-level singleton + convenience functions (backwards compatible)
# ---------------------------------------------------------------------------

_default_store: VectorStore | None = None


def get_default_store() -> VectorStore:
    """Get or create the default VectorStore singleton."""
    global _default_store
    if _default_store is None:
        _default_store = VectorStore()
    return _default_store


def get_collection(name: str) -> chromadb.Collection:
    return get_default_store().get_collection(name)


def list_collections() -> list[str]:
    return get_default_store().list_collections()


def delete_collection(name: str) -> None:
    get_default_store().delete_collection(name)
