# SEC Filing Analyst

Quick run guide for the current branch.

## Setup

Use Python 3.11.

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install -e .
```

Create a `.env` file in the repo root. Minimum useful keys:

```env
OPENAI_API_KEY=
OPENROUTER_API_KEY=
FRED_API_KEY=
```

Notes:
- `FRED_API_KEY` is needed for macro context in the fundamental agent.
- If Chroma telemetry warnings appear, make sure `posthog<6` is installed in the active venv.

## Main Ways To Run

### 1. Normal single-query run

Best for quick testing.

```bash
python -m src.runner "Analyze Microsoft stock"
```

This saves markdown and JSON reports into `outputs/`.

### 2. Demo script

Quick single-query demo:

```bash
python demo/pipeline_demo.py
```

Single-query demo with automatic ticker data preparation:

```bash
python demo/pipeline_demo.py --prepare-data
```

Demo with batch mode:

```bash
python demo/pipeline_demo.py --batch
```

Full demo with batch mode and automatic data preparation:

```bash
python demo/pipeline_demo.py --prepare-data --batch
```

## What `--prepare-data` Does

Some parts of the pipeline use RAG-backed document retrieval:
- SEC filings
- earnings transcripts
- news

If a ticker has not been indexed yet, `--prepare-data` will:
1. check whether indexed RAG data exists for that ticker
2. download demo data for that ticker if needed
3. ingest it into ChromaDB
4. run the analysis

First-time runs for a new ticker can therefore be much slower.
Later runs should be faster once the data is already indexed.

## What `--batch` Does

`--batch` runs the parallel multi-query part of the demo. This is heavier than a single query because each query triggers the full multi-agent pipeline.

## Useful Commands

Check indexed collections:

```bash
python - <<'PY'
from src.rag.store import list_collections
print(list_collections())
PY
```

Test a single sub-agent:

```bash
python demo/single_agent_demo.py --agent sentiment --ticker AAPL
python demo/single_agent_demo.py --agent fundamental --ticker AAPL
python demo/single_agent_demo.py --agent risk --ticker AAPL
```

## Fine-Tuned Sentiment Model

By default, sentiment inference uses `ProsusAI/finbert`.

To train the fine-tuned sentiment checkpoint:

```bash
python scripts/train_sentiment.py \
  --dataset "TheFinAI/fiqa-sentiment-classification" \
  --label-field score \
  --output-dir models/finbert-fiqa-full \
  --epochs 3
```

To use the fine-tuned checkpoint for inference:

```bash
export SENTIMENT_MODEL_PATH="models/finbert-fiqa-full"
python demo/single_agent_demo.py --agent sentiment --ticker TSLA
```

To switch back to the baseline model:

```bash
unset SENTIMENT_MODEL_PATH
python demo/single_agent_demo.py --agent sentiment --ticker TSLA
```

`SENTIMENT_MODEL_PATH` also affects:
- `python demo/pipeline_demo.py`
- `python -m src.runner "Analyze Tesla stock"`

## Sentiment Images

Sentiment runs can now generate chart artifacts automatically, including:
- sentiment distribution
- sentiment scores by evidence chunk
- sentiment timeline (when dated evidence is available)

These images are typically saved in:
- `outputs/`
- `outputs/single_agent/sentiment/`

## Output Files

Main demo output:
- `outputs/demo_report.json`
- `outputs/demo_report.md`

Batch demo output:
- `outputs/batch_*.json`
- `outputs/batch_*.md`

Runner output:
- `outputs/*_report.json`
- `outputs/*_report.md`
