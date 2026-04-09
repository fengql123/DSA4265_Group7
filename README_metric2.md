# Metric 2 Reproduction Notes

This branch contains the Metric 2 evaluation code and the small curated 3M FinanceBench corpus used during debugging.

## Files Included

- `scripts/evaluate_rag.py`
- `src/rag/retriever.py`
- `src/tools/rag_tool.py`
- `scripts/ingest_demo.py`
- `data/benchmarks/financebench/sec/MMM/3M_2018_10K.txt`
- `data/benchmarks/financebench/sec/MMM/3M_2022_10K.txt`
- `data/benchmarks/financebench/sec/MMM/3M_2023Q2_10Q.txt`

## Important Note

- FinQA `gold_program` mode is an oracle-style upper bound because it executes FinQA's provided `program_re`.
- FinQA `llm` mode is the pure LLM baseline.
- FinQA `hybrid_program` mode uses the LLM to infer a short program, then executes it in Python.

## Commands

### FinanceBench Reference Contexts

```bash
python /Users/clarakoh/DSA4265_Group7/scripts/evaluate_rag.py \
  --benchmark financebench \
  --dataset-path /Users/clarakoh/DSA4265_Group7/data/benchmarks/financebench/train.parquet \
  --split train \
  --sample-size 5 \
  --use-reference-contexts
```

### FinanceBench Original Retrieval

```bash
python /Users/clarakoh/DSA4265_Group7/scripts/evaluate_rag.py \
  --benchmark financebench \
  --dataset-path /Users/clarakoh/DSA4265_Group7/data/benchmarks/financebench/train.parquet \
  --split train \
  --sample-size 5 \
  --collection-names sec_filings,earnings,news
```

### FinanceBench Curated 3M Retrieval

```bash
python /Users/clarakoh/DSA4265_Group7/scripts/evaluate_rag.py \
  --benchmark financebench \
  --dataset-path /Users/clarakoh/DSA4265_Group7/data/benchmarks/financebench/train.parquet \
  --split train \
  --sample-size 5 \
  --collection-names financebench_sec_filings
```

### FinQA LLM-Only

```bash
python /Users/clarakoh/DSA4265_Group7/scripts/evaluate_rag.py \
  --benchmark finqa \
  --dataset-id ibm/finqa \
  --split test \
  --sample-size 20 \
  --use-reference-contexts
```

### FinQA Hybrid Program

```bash
python /Users/clarakoh/DSA4265_Group7/scripts/evaluate_rag.py \
  --benchmark finqa \
  --dataset-id ibm/finqa \
  --split test \
  --sample-size 20 \
  --use-reference-contexts \
  --finqa-solver hybrid_program
```

### FinQA Gold Program Upper Bound

```bash
python /Users/clarakoh/DSA4265_Group7/scripts/evaluate_rag.py \
  --benchmark finqa \
  --dataset-id ibm/finqa \
  --split test \
  --sample-size 20 \
  --use-reference-contexts \
  --finqa-solver gold_program
```
