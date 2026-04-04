"""RAG pipeline — VectorStore, DocumentIndexer, Retriever.

The three classes share an embedding model when wired together:

    from src.rag import VectorStore, DocumentIndexer, Retriever

    store = VectorStore()
    indexer = DocumentIndexer(store=store)
    retriever = Retriever(store=store, embedding_model=indexer.embedding_model)

Or use the module-level convenience functions which do this automatically:

    from src.rag.indexer import index_documents
    from src.rag.retriever import retrieve
"""

from src.rag.store import VectorStore
from src.rag.indexer import DocumentIndexer
from src.rag.retriever import Retriever, RetrievedChunk

__all__ = ["VectorStore", "DocumentIndexer", "Retriever", "RetrievedChunk"]
