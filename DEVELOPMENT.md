# Development Guide

## System Overview

This is a multi-agent investment analysis pipeline. A user submits a natural language query (e.g. "Should I invest in Apple?"), and the system produces a structured investment memo with a Buy/Hold/Sell recommendation.

### Architecture

```
User: "Should I invest in Apple?"
    │
    ▼
MainAgent (ReAct loop with LLM):
    1. LLM extracts ticker (AAPL) + date range from query
    2. LLM calls sub-agent tools (can be parallel via ThreadPoolExecutor):
       ├── SentimentAgent(AAPL, ...)   → SentimentReport   (stub)
       ├── FundamentalAgent(AAPL, ...) → FundamentalReport  (stub, tools wired)
       ├── TechnicalAgent(AAPL, ...)   → TechnicalReport    (stub, tools wired)
       └── RiskAgent(AAPL, ...)        → RiskReport         (stub)
    3. LLM sees all reports + any chart images (multimodal)
    4. LLM produces InvestmentMemo with recommendation
    │
    ▼
Output: InvestmentMemo (markdown report + structured JSON)
```

### Key Design Decisions

- **Sub-agents are both agents AND tools.** Each sub-agent is a `BaseAgent` subclass with its own ReAct loop, but it's also callable as a tool by the MainAgent via the `AgentTool` wrapper.
- **Parallel tool execution.** When the LLM issues multiple tool calls in one response, they run in parallel via `ThreadPoolExecutor`. Multiple user queries run in parallel via `graph.batch()`.
- **Multimodal support.** Tools can return images (via `ToolResult` + `Artifact`). If the LLM supports vision, images are embedded as base64 in the conversation.
- **All override points are abstract.** `BaseAgent` defines 8 abstract methods that every agent must implement. Only `run()` is concrete.
- **Artifacts flow through the pipeline.** Sub-agent artifacts (charts, images) are collected by `AgentTool`, passed back to MainAgent, and can be embedded in the LLM conversation for visual analysis.

---

## Project Structure

```
DSA4265-group7/
├── config/
│   ├── settings.yaml              # LLM provider, embedding model, RAG params
│   └── prompts/
│       ├── main.yaml              # MainAgent prompt template
│       ├── fundamental.yaml       # FundamentalAgent prompt
│       ├── technical.yaml         # TechnicalAgent prompt
│       ├── sentiment.yaml         # SentimentAgent prompt (for teammates)
│       └── risk.yaml              # RiskAgent prompt (for teammates)
├── src/
│   ├── config.py                  # Settings loader, get_llm(), get_embedding_model()
│   ├── schemas.py                 # Pydantic output models + PipelineState
│   ├── artifacts.py               # Artifact, ArtifactType (top-level, used everywhere)
│   ├── graph.py                   # LangGraph: START → main → END
│   ├── runner.py                  # CLI: python -m src.runner "query"
│   ├── agents/
│   │   ├── base.py                # BaseAgent (abstract) + agent registry
│   │   ├── main_agent.py          # MainAgent — orchestrates sub-agents (partial)
│   │   ├── fundamental_agent.py   # STUB — tools wired, needs refinement
│   │   ├── technical_agent.py     # STUB — tools wired, needs refinement
│   │   ├── sentiment_agent.py     # STUB — returns mock data
│   │   └── risk_agent.py          # STUB — returns mock data
│   ├── tools/
│   │   ├── base_tool.py           # BaseTool (abstract) + ToolResult + auto-registration
│   │   ├── agent_tool.py          # AgentTool — wraps BaseAgent as a tool
│   │   ├── registry.py            # Tool registry (auto-discovery)
│   │   ├── rag_tool.py            # RAG retrieval tool
│   │   ├── market_data_tool.py    # yfinance market data tool
│   │   ├── fred_tool.py           # FRED economic data tool
│   │   └── mcp_server.py          # MCP server for external tool access
│   └── rag/
│       ├── store.py               # VectorStore (ChromaDB wrapper)
│       ├── indexer.py             # DocumentIndexer (chunk + embed + upsert)
│       └── retriever.py           # Retriever (query + metadata filter)
├── scripts/
│   ├── download_hf.py             # Generic HuggingFace dataset downloader
│   ├── download_kaggle.py         # Generic Kaggle dataset downloader
│   ├── download_sec_filings.py    # SEC EDGAR 10-K/10-Q downloader
│   ├── download_market_data.py    # yfinance OHLCV downloader
│   ├── download_fred.py           # FRED macro data downloader
│   ├── download_demo_data.py      # Download small demo dataset
│   ├── ingest.py                  # CLI: ingest data into ChromaDB
│   └── ingest_demo.py             # Ingest demo data into ChromaDB
├── demo/
│   ├── single_agent_demo.py       # Test individual sub-agents + save outputs
│   └── pipeline_demo.py           # Full pipeline demo (MainAgent + all sub-agents)
└── .env                           # API keys (not tracked by git)
```

---

## Agent Architecture

### BaseAgent (`src/agents/base.py`)

All agents inherit from `BaseAgent`. It defines 8 abstract methods and one concrete method (`run()`).

#### Abstract Methods — You Must Implement All 8

| Method | Signature | Purpose |
|--------|-----------|---------|
| `get_system_prompt` | `(state: dict) -> str` | Build the system prompt. Load from YAML, format with state values. |
| `build_messages` | `(state: dict) -> list` | Construct initial `[SystemMessage, HumanMessage]`. |
| `get_tools` | `() -> list` | Return LangChain tool objects. Use `get_tools(self.tool_names)` from registry. |
| `handle_tool_result` | `(result) -> tuple[str, list[Artifact]]` | Unpack a tool's return into `(content_string, artifacts_list)`. |
| `build_artifact_message` | `(artifacts: list[Artifact]) -> HumanMessage \| None` | Create multimodal message with images, or `None`. |
| `parse_output` | `(messages: list) -> YourReport` | Produce structured Pydantic output. Call `get_llm().with_structured_output(model)`. |
| `build_result` | `(output, artifacts) -> dict` | Construct return dict: `{self.output_field: output, "artifacts": [...]}`. |
| `is_vision_capable` | `() -> bool` | Check if LLM supports vision. Read from config or return `False`. |

#### The `run()` Flow (Concrete — Don't Override)

```python
def run(self, state):
    messages = self.build_messages(state)        # 1. Initial messages
    tools = self.get_tools()                     # 2. Get tools
    collected_artifacts = []

    if tools:
        llm_with_tools = get_llm().bind_tools(tools)
        for round in range(max_tool_rounds):     # 3. ReAct loop
            response = llm_with_tools.invoke(messages)
            if not response.tool_calls:
                break                            #    LLM done gathering info

            # Execute tool calls (parallel via ThreadPoolExecutor if multiple)
            for each tool_call:
                result = tool.func(**args)       #    .func() preserves ToolResult
                content, artifacts = self.handle_tool_result(result)
                collected_artifacts.extend(artifacts)
                messages.append(ToolMessage(content))

            art_msg = self.build_artifact_message(artifacts)
            if art_msg:
                messages.append(art_msg)         #    Embed images for vision LLMs

    output = self.parse_output(messages)         # 4. Structured output
    return self.build_result(output, artifacts)   # 5. State update
```

### Constructor Parameters

```python
BaseAgent.__init__(
    agent_name="your_agent",          # Matches prompt YAML filename
    tool_names=["get_market_data"],   # Tool names from registry
    output_field="your_report",       # Field name in return dict
    output_model=YourReport,          # Pydantic model class
    mcp_servers={},                   # Optional MCP server configs
    max_tool_rounds=10,               # Max ReAct iterations
)
```

---

## Tool Architecture

### BaseTool (`src/tools/base_tool.py`)

All tools inherit from `BaseTool`. Tools auto-register when their module is imported.

```python
class MyTool(BaseTool):
    name = "my_tool"                    # Unique name
    description = "What this tool does" # LLM sees this
    input_schema = MyInput              # Pydantic model for inputs

    def execute(self, **kwargs) -> str | ToolResult:
        # Your logic here
        return "result text"
        # Or with artifacts:
        return ToolResult(content="text", artifacts=[Artifact(...)])
```

`ToolResult` is defined in `src/tools/base_tool.py` alongside `BaseTool`.

### AgentTool (`src/tools/agent_tool.py`)

Wraps any `BaseAgent` so it can be called as a tool. MainAgent uses it automatically.

```python
# MainAgent.get_tools() does this internally:
AgentTool(SentimentAgent()).to_langchain_tool()
```

When called, `AgentTool`:
1. Validates ticker (yfinance check) and date range
2. Builds a state dict for the sub-agent
3. Calls `agent.run(state)` — the sub-agent's full ReAct loop
4. Returns the report as JSON + any artifacts as `ToolResult`

### Artifacts (`src/artifacts.py`)

Top-level module for non-text outputs. Used by agents, tools, and state.

```python
from src.artifacts import Artifact, ArtifactType

artifact = Artifact(
    artifact_type=ArtifactType.IMAGE,
    path="outputs/chart.png",
    mime_type="image/png",
    description="Sentiment timeline chart",
)

# Convert to base64 for multimodal LLM messages:
artifact.to_base64()           # raw base64 string
artifact.to_multimodal_block() # {"type": "image_url", "image_url": {"url": "data:..."}}
```

---

## Output Schemas (`src/schemas.py`)

Each agent produces a Pydantic model:

| Schema | Agent | Key Fields |
|--------|-------|------------|
| `SentimentReport` | SentimentAgent | `overall_sentiment`, `sentiment_score`, `key_themes`, `evidence`, `chart_paths` |
| `FundamentalReport` | FundamentalAgent | `revenue_trend`, `margin_analysis`, `valuation_assessment`, `macro_context`, `key_metrics` |
| `TechnicalReport` | TechnicalAgent | `current_price`, `fifty_two_week_high/low`, `moving_avg_50d/200d`, `beta`, `technical_signal` |
| `RiskReport` | RiskAgent | `risk_factors`, `risk_level`, `mitigants` |
| `InvestmentMemo` | MainAgent | `recommendation`, `confidence`, `thesis`, summaries, `report_markdown` |

All have a `ticker: str` and `summary: str` field.

`PipelineState` (TypedDict) carries: `query`, `investment_memo`, `artifacts`, `errors`.

---

## Data

There are two categories of data in this project:

1. **RAG data** — text documents (SEC filings, earnings transcripts, news) that are chunked, embedded, and stored in ChromaDB for semantic retrieval by agents via the `rag_retrieve` tool.
2. **Live data** — real-time market data and macroeconomic indicators fetched on-the-fly by tools (`get_market_data`, `get_fred_data`) during the agent's ReAct loop. These are NOT stored in ChromaDB.

### RAG Architecture

```
Raw Documents (SEC filings, earnings, news)
    │
    ▼  scripts/download_*.py or scripts/download_hf.py
Downloaded to data/
    │
    ▼  scripts/ingest.py or scripts/ingest_demo.py
DocumentIndexer:
    1. Reads text files
    2. Chunks text (LlamaIndex SentenceSplitter, 512 tokens, 64 overlap)
    3. Embeds chunks (BAAI/bge-base-en-v1.5 via sentence-transformers)
    4. Upserts to ChromaDB with metadata (ticker, doc_type, etc.)
    │
    ▼  Stored in ChromaDB (data/chromadb/)
Collections: "sec_filings", "earnings", "news", etc.
    │
    ▼  Agent calls rag_retrieve tool during ReAct loop
Retriever:
    1. Embeds the query (same embedding model)
    2. Searches ChromaDB with cosine similarity + metadata filter
    3. Returns top-k RetrievedChunk objects (text + metadata + score)
```

### RAG Classes (`src/rag/`)

| Class | File | Purpose |
|-------|------|---------|
| `VectorStore` | `store.py` | ChromaDB `PersistentClient` wrapper. Manages collections. |
| `DocumentIndexer` | `indexer.py` | Chunks + embeds + upserts. Caches the embedding model. |
| `Retriever` | `retriever.py` | Semantic search with metadata filtering. Shares the embedding model with the indexer. |

All three have module-level singletons and convenience functions (`index_documents()`, `retrieve()`, etc.) so you don't need to instantiate them manually.

### Ingesting Data into ChromaDB

**From Python:**

```python
from src.rag.indexer import index_documents, index_files

# Index raw text with custom metadata
index_documents(
    texts=["Revenue grew 12% YoY..."],
    collection_name="sec_filings",
    metadata=[{"ticker": "AAPL", "doc_type": "10-K", "section": "MD&A"}],
)

# Index files from disk with a metadata function
index_files(
    file_paths=["data/sec/AAPL/10-K_2024.txt"],
    collection_name="sec_filings",
    metadata_fn=lambda path: {"ticker": path.parent.name, "doc_type": "10-K"},
)
```

**From CLI:**

```bash
# Single directory with explicit metadata
python scripts/ingest.py \
  --input-dir data/sec/AAPL/ \
  --collection sec_filings \
  --metadata '{"ticker": "AAPL", "doc_type": "10-K"}'

# Batch ingestion via YAML manifest
python scripts/ingest.py --manifest config/ingest_manifest.yaml

# Demo data (convenience script)
python scripts/ingest_demo.py --ticker AAPL
```

### Retrieving Data from ChromaDB

**From Python (direct):**

```python
from src.rag.retriever import retrieve

chunks = retrieve(
    query="What is Apple's revenue growth?",
    collection_names=["sec_filings", "earnings"],
    metadata_filter={"ticker": "AAPL"},
    top_k=10,
)
for chunk in chunks:
    print(chunk.text, chunk.metadata, chunk.score)
```

**From an agent (via tool):** Add `rag_retrieve` to your agent's `tool_names`. The LLM will construct queries and call it during the ReAct loop. The tool handles embedding, searching, and formatting results.

### Download Scripts

Generic scripts for downloading raw data from various sources. Downloaded data can be used for:

- **RAG ingestion** — text documents (filings, transcripts, news) chunked and stored in ChromaDB
- **Model training / fine-tuning** — datasets for training sentiment models, NER models, etc.
- **Evaluation** — benchmark datasets for testing pipeline quality

The scripts just fetch raw data — what you do with it (ingest, train, evaluate) is up to you.

| Script | Source | What it downloads | Example use cases |
|--------|--------|-------------------|-------------------|
| `scripts/download_hf.py` | HuggingFace | Any HF dataset (`--dataset`, `--split`, `--stream`) | Sentiment training data, NER data, eval benchmarks |
| `scripts/download_kaggle.py` | Kaggle | Any Kaggle dataset or competition data | Alternative financial datasets |
| `scripts/download_sec_filings.py` | SEC EDGAR | 10-K/10-Q filings for S&P 500 tickers | RAG ingestion |
| `scripts/download_market_data.py` | yfinance | OHLCV prices + company fundamentals | LSTM training, backtesting |
| `scripts/download_fred.py` | FRED | Macroeconomic indicators (GDP, rates, etc.) | Macro context for analysis |
| `scripts/download_demo_data.py` | Multiple | Small dataset for one ticker (demo use) | Quick demo setup |

**Relevant HuggingFace datasets for this project:**

| Dataset | HF ID | Use |
|---------|-------|-----|
| Earnings transcripts + QA | `lamini/earnings-calls-qa` | RAG ingestion |
| S&P 500 earnings | `kurry/sp500_earnings_transcripts` | RAG ingestion |
| Financial news (57M+ rows) | `Brianferrell787/financial-news-multisource` | RAG ingestion, sentiment training |
| Financial PhraseBank | `takala/financial_phrasebank` | Sentiment model training |
| FiQA sentiment | `TheFinAI/fiqa-sentiment-classification` | Sentiment model training |
| FiNER-139 NER | `nlpaueb/finer-139` | NER model training |
| FinanceBench | `PatronusAI/financebench` | RAG evaluation (150 expert QA) |
| FinQA | `ibm/finqa` | RAG evaluation (8,281 numerical QA) |
| Earnings + price reactions | `jlh-ibm/earnings_call` | Sentiment fine-tuning with market labels |

```bash
# Example: download sentiment training data
python scripts/download_hf.py --dataset "takala/financial_phrasebank" --config "sentences_allagree" --output data/hf/phrasebank

# Example: download large news dataset with streaming
python scripts/download_hf.py --dataset "Brianferrell787/financial-news-multisource" --output data/hf/news --stream --max-rows 100000
```

**Note:** Market data (`get_market_data`) and FRED data (`get_fred_data`) are also fetched live by tools during the agent's ReAct loop — they don't need to be downloaded or ingested for the pipeline to work.

### Adding a New RAG Data Source

1. Download your data (use existing scripts or write your own)
2. Ingest into a ChromaDB collection with appropriate metadata:
   ```python
   index_documents(
       texts=your_texts,
       collection_name="my_collection",
       metadata=[{"ticker": "AAPL", "doc_type": "my_type"} for _ in your_texts],
   )
   ```
3. Add `rag_retrieve` to your agent's `tool_names`
4. In your prompt, tell the LLM which `collection_names` to search

---

## Configuration

### LLM Provider (`config/settings.yaml`)

```yaml
llm:
  provider: "openrouter"          # "openrouter" | "openai" | "anthropic" | "local"
  model: "openai/gpt-5.4-nano"
  temperature: 0.1
  vision_enabled: true            # Enable multimodal image injection

  openrouter:
    api_key_env: "OPENROUTER_API_KEY"
  local:
    base_url: "http://localhost:30000/v1"  # SGLang
```

Switch providers by changing `provider` and `model`. Set the corresponding API key in `.env` (see below).

### Prompt Templates (`config/prompts/`)

YAML files with a `system_prompt` key. Use `{placeholders}` for dynamic values:

```yaml
system_prompt: |
  You are a fundamental analyst. Analyze {ticker} as of {date}.
  Use get_market_data to fetch data, then provide your analysis.
```

Loaded via `load_prompt("fundamental")` in your `get_system_prompt()`.

---

## Running the Pipeline

### Quick Start

```bash
# 1. Install
pip install -e .
# Create .env with your API keys:
cat > .env << 'EOF'
# OpenRouter (default provider) — get a key at https://openrouter.ai/keys
OPENROUTER_API_KEY=sk-or-v1-your-key-here

# Alternative providers (only needed if you change llm.provider in settings.yaml)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=

# Data source keys
FRED_API_KEY=
# Kaggle: place kaggle.json in ~/.kaggle/ instead
EOF

# 2. Download demo data (SEC filing + earnings + market data for AAPL)
python scripts/download_demo_data.py --ticker AAPL

# 3. Ingest into ChromaDB (so rag_retrieve can find documents)
python scripts/ingest_demo.py --ticker AAPL

# 4. Test individual sub-agents (outputs saved to outputs/single_agent/)
python demo/single_agent_demo.py
python demo/single_agent_demo.py --agent fundamental --ticker AAPL --date 2025-04-04

# 5. Run the full pipeline
python demo/pipeline_demo.py
```

### CLI Usage

```bash
# Single query
python -m src.runner "Should I invest in Apple?"

# Multiple queries in parallel (uses graph.batch())
python -m src.runner "Should I invest in Apple?" "Analyze Tesla"
```

### Demos

**`demo/single_agent_demo.py`** — Runs each sub-agent individually (not through MainAgent). Tests tools, artifacts, and structured reports. Outputs saved to `outputs/single_agent/{agent_name}/`.

```bash
python demo/single_agent_demo.py                                          # all agents, defaults
python demo/single_agent_demo.py --agent fundamental                      # just one agent
python demo/single_agent_demo.py --ticker MSFT --date 2025-03-01          # custom ticker + date
python demo/single_agent_demo.py --ticker AAPL --lookback-days 180        # custom lookback
python demo/single_agent_demo.py --agent fundamental --debug              # with debug logging
```

**`demo/pipeline_demo.py`** — Runs the full pipeline: user query → MainAgent → sub-agent tools → InvestmentMemo. Also tests parallel multi-query via `graph.batch()`. Outputs saved to `outputs/`.

---

## How to Add a New Agent

### Step 1: Define your output schema

In `src/schemas.py`, add a Pydantic model:

```python
class MyReport(BaseModel):
    ticker: str
    my_field: str = Field(description="What this field contains")
    summary: str = Field(description="Concise summary")
```

### Step 2: Write your prompt template

Create `config/prompts/my_agent.yaml`:

```yaml
system_prompt: |
  You are a specialist analyst. Analyze {ticker} as of {date}.
  Use your tools to gather data, then provide analysis.
```

### Step 3: Create your agent class

Create `src/agents/my_agent.py`:

```python
from src.agents.base import BaseAgent
from src.artifacts import Artifact
from src.config import get_llm, load_prompt
from src.schemas import MyReport
from src.tools.base_tool import ToolResult
from src.tools.registry import get_tools
from langchain_core.messages import HumanMessage, SystemMessage


class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_name="my_agent",
            tool_names=["rag_retrieve"],
            output_field="my_report",
            output_model=MyReport,
        )

    def get_system_prompt(self, state: dict) -> str:
        template = load_prompt(self.agent_name)
        return template.format(
            ticker=state.get("ticker", "UNKNOWN"),
            date=state.get("analysis_date", "UNKNOWN"),
        )

    def build_messages(self, state: dict) -> list:
        return [
            SystemMessage(content=self.get_system_prompt(state)),
            HumanMessage(content=f"Analyze {state.get('ticker')} as of {state.get('analysis_date')}."),
        ]

    def get_tools(self) -> list:
        return get_tools(self.tool_names)

    def handle_tool_result(self, result):
        if isinstance(result, ToolResult):
            return (result.content, result.artifacts)
        if isinstance(result, tuple) and len(result) == 2:
            return (str(result[0]), list(result[1]) if result[1] else [])
        return (str(result), [])

    def build_artifact_message(self, artifacts):
        return None

    def parse_output(self, messages: list) -> MyReport:
        llm = get_llm().with_structured_output(self.output_model)
        messages = messages + [
            HumanMessage(content="Now produce your final structured report.")
        ]
        return llm.invoke(messages)

    def build_result(self, output, artifacts):
        result = {self.output_field: output}
        if artifacts:
            result["artifacts"] = artifacts
        return result

    def is_vision_capable(self) -> bool:
        return False
```

### Step 4: Wire it into MainAgent

In `src/agents/main_agent.py`, add your agent to `get_tools()`:

```python
from src.agents.my_agent import MyAgent

tools = [
    ...,
    AgentTool(MyAgent()).to_langchain_tool(),
]
```

Update `config/prompts/main.yaml` to tell the LLM about the new tool.

---

## How to Add a New Tool

### Step 1: Define input schema + tool class

Create `src/tools/my_tool.py`:

```python
from pydantic import BaseModel, Field
from src.tools.base_tool import BaseTool


class MyToolInput(BaseModel):
    query: str = Field(description="What to search for")
    limit: int = Field(default=10, description="Max results")


class MyTool(BaseTool):
    name = "my_tool"
    description = "Search for something useful"
    input_schema = MyToolInput

    def execute(self, query: str, limit: int = 10) -> str:
        return f"Found {limit} results for '{query}'"
```

### Step 2: Register it

Add your module to `_TOOL_MODULES` in `src/tools/registry.py`:

```python
_TOOL_MODULES = [
    ...,
    "src.tools.my_tool",
]
```

### Step 3: Use it in an agent

In your agent's `__init__`:

```python
tool_names=["rag_retrieve", "my_tool"]
```

---

## Current Status

| Agent | Status | Uses Tools | Notes |
|-------|--------|-----------|-------|
| MainAgent | **Partial** | Sub-agent tools via AgentTool | Orchestration works; prompt and output parsing need refinement |
| FundamentalAgent | **Stub** | `get_market_data`, `get_fred_data`, `rag_retrieve` | Tools wired with ReAct loop; prompt and parsing need work |
| TechnicalAgent | **Stub** | `get_market_data` | Tool wired with ReAct loop; prompt and parsing need work |
| SentimentAgent | **Stub** | None (returns mock data) | Generates dummy chart for multimodal test |
| RiskAgent | **Stub** | None (returns mock data) | Returns random risk factors |

| Tool | Status | Notes |
|------|--------|-------|
| `rag_retrieve` | **Implemented** | Searches ChromaDB with metadata filtering |
| `get_market_data` | **Implemented** | Live yfinance data |
| `get_fred_data` | **Implemented** | FRED macro indicators (needs `FRED_API_KEY`) |
| MCP Server | **Implemented** | Exposes tools via Model Context Protocol |

| Component | Status | Notes |
|-----------|--------|-------|
| `Artifact` / `ArtifactType` | **Implemented** | `src/artifacts.py` — top-level module |
| `ToolResult` | **Implemented** | `src/tools/base_tool.py` — tool return type |
| `AgentTool` | **Implemented** | Wraps any BaseAgent as a callable tool |
| RAG Pipeline | **Implemented** | VectorStore + DocumentIndexer + Retriever |
| Parallel tool execution | **Implemented** | ThreadPoolExecutor in BaseAgent.run() |
| Parallel multi-query | **Implemented** | graph.batch() in runner |
| Multimodal artifacts | **Implemented** | Base64 image injection in build_artifact_message() |
