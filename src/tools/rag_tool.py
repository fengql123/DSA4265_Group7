"""RAG retrieval tool.

Allows agents to search the ChromaDB vector store for relevant document chunks.
The LLM constructs its own queries — this is the key RAG interface.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.tools.base_tool import BaseTool


class RagRetrieveInput(BaseModel):
    """Input schema for the RAG retrieval tool."""

    query: str = Field(
        description=(
            "Search query describing what information to find. "
            "Be specific (e.g. 'Apple revenue growth Q4 2024' not just 'Apple')."
        )
    )
    collection_names: list[str] = Field(
        description='Which collections to search. Options: "sec_filings", "earnings", "news", "demo_sec_filings", "demo_earnings".'
    )
    ticker: str = Field(description="Stock ticker symbol to filter results (e.g. 'AAPL').")
    top_k: int = Field(default=10, description="Number of document chunks to retrieve.")


class RagRetrieveTool(BaseTool):
    name = "rag_retrieve"
    description = (
        "Retrieve relevant document chunks from the vector store. "
        "Use this to search for financial documents, SEC filings, news articles, "
        "and earnings transcripts. Construct specific, targeted queries for best results."
    )
    input_schema = RagRetrieveInput

    def execute(self, query: str, collection_names: list[str], ticker: str, top_k: int = 10) -> str:
        from src.rag.retriever import retrieve

        chunks = retrieve(
            query=query,
            collection_names=collection_names,
            metadata_filter={"ticker": ticker},
            top_k=top_k,
        )

        if not chunks:
            return f"No relevant documents found for query: '{query}' (ticker: {ticker})"

        parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.metadata.get("source_file", "unknown")
            doc_type = chunk.metadata.get("doc_type", "unknown")
            parts.append(f"[Source {i}: {source} ({doc_type})]\n{chunk.text}")

        return "\n---\n".join(parts)
