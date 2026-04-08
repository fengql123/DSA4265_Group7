"""Document retrieval from ChromaDB.

Provides a Retriever class that queries ChromaDB collections with
flexible metadata filtering. Shares the embedding model with DocumentIndexer.

Usage:
    from src.rag.retriever import Retriever

    retriever = Retriever()
    chunks = retriever.retrieve(
        query="What are Apple's key revenue drivers?",
        collection_names=["sec_filings"],
        metadata_filter={"ticker": "AAPL"},
        top_k=10,
    )
    for chunk in chunks:
        print(chunk.text, chunk.metadata, chunk.score)
"""


from __future__ import annotations

import os
from dataclasses import dataclass

from src.config import get_rag_config
from src.rag.store import VectorStore, get_default_store


@dataclass
class RetrievedChunk:
    """A single retrieved document chunk with metadata and relevance score."""

    text: str
    metadata: dict
    score: float  # Distance score (lower = more relevant for cosine)


class Retriever:
    """Queries ChromaDB collections for relevant document chunks.

    Shares the embedding model with DocumentIndexer when constructed
    from the same RAGPipeline, or loads its own if used standalone.

    Args:
        store: VectorStore instance. Uses the default singleton if None.
        embedding_model: Pre-loaded SentenceTransformer. Loads from config if None.
        top_k: Default number of results to return. Falls back to config if None.
    """

    def __init__(
        self,
        store: VectorStore | None = None,
        embedding_model=None,
        top_k: int | None = None,
    ):
        self._store = store or get_default_store()
        self._embedding_model = embedding_model
        cfg = get_rag_config()
        env_top_k = os.getenv("PIPELINE_TOP_K")
        self._default_top_k = top_k or int(env_top_k or cfg["retrieval_top_k"])

    @property
    def embedding_model(self):
        """Lazy-load the embedding model on first access."""
        if self._embedding_model is None:
            from src.config import get_embedding_model

            self._embedding_model = get_embedding_model()
        return self._embedding_model

    def retrieve(
        self,
        query: str,
        collection_names: list[str],
        metadata_filter: dict | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve relevant chunks from one or more ChromaDB collections.

        Args:
            query: Search query text.
            collection_names: Which ChromaDB collections to search.
            metadata_filter: Metadata filter dict (e.g. {"ticker": "AAPL"}).
            top_k: Number of chunks to retrieve overall.

        Returns:
            List of RetrievedChunk objects sorted by relevance (best first).
        """
        top_k = top_k or self._default_top_k

        # Embed the query
        query_embedding = self.embedding_model.encode([query]).tolist()[0]

        all_chunks: list[RetrievedChunk] = []
        per_collection_k = max(1, top_k // len(collection_names))

        for col_name in collection_names:
            try:
                collection = self._store.get_collection(col_name)
            except Exception as e:
                print(f"Warning: Could not access collection '{col_name}': {e}")
                continue

            query_kwargs = {
                "query_embeddings": [query_embedding],
                "n_results": per_collection_k,
            }

            if metadata_filter:
                where = self._build_where_clause(metadata_filter)
                if where:
                    query_kwargs["where"] = where

            try:
                results = collection.query(**query_kwargs)
            except Exception as e:
                print(f"Warning: Query failed on collection '{col_name}': {e}")
                continue

            if results and results["documents"] and results["documents"][0]:
                documents = results["documents"][0]
                metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(documents)
                distances = results["distances"][0] if results["distances"] else [0.0] * len(documents)

                for doc, meta, dist in zip(documents, metadatas, distances):
                    all_chunks.append(RetrievedChunk(text=doc, metadata=meta, score=dist))

        # Sort by score (lower distance = more relevant)
        all_chunks.sort(key=lambda c: c.score)
        return all_chunks[:top_k]

    @staticmethod
    def _build_where_clause(metadata_filter: dict) -> dict | None:
        """Convert a flat metadata filter dict to ChromaDB where clause."""
        if not metadata_filter:
            return None

        conditions = [{k: {"$eq": v}} for k, v in metadata_filter.items()]

        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}


# ---------------------------------------------------------------------------
# Module-level singleton + convenience function (backwards compatible)
# ---------------------------------------------------------------------------

_default_retriever: Retriever | None = None


def get_default_retriever() -> Retriever:
    """Get or create the default Retriever singleton."""
    global _default_retriever
    if _default_retriever is None:
        # Share the embedding model with the default indexer if it exists
        from src.rag.indexer import get_default_indexer

        indexer = get_default_indexer()
        _default_retriever = Retriever(
            store=indexer.store,
            embedding_model=indexer.embedding_model,
        )
    return _default_retriever


def retrieve(
    query: str,
    collection_names: list[str],
    metadata_filter: dict | None = None,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    return get_default_retriever().retrieve(query, collection_names, metadata_filter, top_k)
