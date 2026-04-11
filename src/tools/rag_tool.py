"""RAG retrieval tool.

Allows agents to search the ChromaDB vector store for relevant document chunks.
The LLM constructs its own queries — this is the key RAG interface.
"""

from __future__ import annotations

from datetime import date
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
    start_date: str | None = Field(
        default=None,
        description="Optional inclusive start date filter in YYYY-MM-DD format.",
    )
    end_date: str | None = Field(
        default=None,
        description="Optional inclusive end date filter in YYYY-MM-DD format.",
    )
    preferred_doc_type: str | None = Field(
        default=None,
        description="Optional preferred document type for reranking (e.g. 'sec_filing', 'earnings_transcript', 'news').",
    )
    preferred_year: int | None = Field(
        default=None,
        description="Optional preferred reporting year for reranking (e.g. 2018, 2022).",
    )


class RagRetrieveTool(BaseTool):
    name = "rag_retrieve"
    description = (
        "**What**: Retrieves relevant document chunks from a ChromaDB vector store of financial documents. "
        "**When to use**: Any time you need qualitative evidence from SEC filings (10-K / 10-Q), earnings call transcripts, or news articles. "
        "For quantitative market data use `get_market_data` or `get_price_history` instead. "
        "**Input**: "
        "`query` (str — be specific, e.g. 'Apple Services revenue growth Q4 2024' not 'Apple'); "
        "`collection_names` (list[str] — choose from 'sec_filings', 'earnings', 'news', 'demo_sec_filings', 'demo_earnings'); "
        "`ticker` (str — filters results to one company); "
        "`top_k` (int, default 10); "
        "`start_date`, `end_date` (optional YYYY-MM-DD — excludes chunks outside the window; use these to avoid lookahead when running as-of a past date); "
        "`preferred_doc_type` (optional 'sec_filing' | 'earnings_transcript' | 'news' — rerank toward this type); "
        "`preferred_year` (optional int — rerank toward this reporting year). "
        "**Output**: Human-readable text block of the top-k chunks, each prefixed with `[Source i: file (doc_type) | date=… | title=…]`. "
        "If no chunks match inside the date window, returns a short explanatory message. "
        "**Limits**: Quality depends on what has been ingested into ChromaDB for the ticker. "
        "If a ticker has no data in the requested collection you will get 'No relevant documents found' — do not fabricate evidence."
    )
    input_schema = RagRetrieveInput

    @staticmethod
    def _parse_iso_date(value: str | None) -> date | None:
        if not value:
            return None
        try:
            return date.fromisoformat(str(value)[:10])
        except ValueError:
            return None

    def execute(
        self,
        query: str,
        collection_names: list[str],
        ticker: str,
        top_k: int = 10,
        start_date: str | None = None,
        end_date: str | None = None,
        preferred_doc_type: str | None = None,
        preferred_year: int | None = None,
    ) -> str:
        from src.rag.retriever import retrieve

        chunks = retrieve(
            query=query,
            collection_names=collection_names,
            metadata_filter={"ticker": ticker},
            top_k=max(top_k * 3, top_k),
            preferred_doc_type=preferred_doc_type,
            preferred_year=preferred_year,
        )

        start_bound = self._parse_iso_date(start_date)
        end_bound = self._parse_iso_date(end_date)
        filtered_chunks = []
        skipped_out_of_window = 0

        for chunk in chunks:
            chunk_date = self._parse_iso_date(chunk.metadata.get("date"))
            if chunk_date is not None:
                if start_bound and chunk_date < start_bound:
                    skipped_out_of_window += 1
                    continue
                if end_bound and chunk_date > end_bound:
                    skipped_out_of_window += 1
                    continue
            filtered_chunks.append(chunk)

        chunks = filtered_chunks[:top_k]

        if not chunks:
            if skipped_out_of_window:
                return (
                    f"No relevant in-window documents found for query: '{query}' "
                    f"(ticker: {ticker}, skipped {skipped_out_of_window} out-of-window chunk(s))"
                )
            return f"No relevant documents found for query: '{query}' (ticker: {ticker})"

        parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.metadata.get("source_file", "unknown")
            doc_type = chunk.metadata.get("doc_type", "unknown")
            date = chunk.metadata.get("date")
            title = chunk.metadata.get("title")

            header_bits = [f"Source {i}: {source} ({doc_type})"]
            if date:
                header_bits.append(f"date={date}")
            if title:
                header_bits.append(f"title={title}")

            header = " | ".join(header_bits)
            parts.append(f"[{header}]\n{chunk.text}")

        if skipped_out_of_window:
            parts.append(f"[Note] Skipped {skipped_out_of_window} retrieved chunk(s) outside {start_date} to {end_date}.")

        return "\n---\n".join(parts)
