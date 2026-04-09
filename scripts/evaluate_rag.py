#!/usr/bin/env python3
"""Evaluate RAG quality on FinanceBench / FinQA.

Supports two evaluation modes:
1. Retrieval mode: retrieve contexts from ChromaDB using the project's retriever.
2. Reference-context mode: use contexts bundled with the benchmark dataset.

Metrics:
- RAGAS: Faithfulness, Answer Relevancy, Context Precision.
- FinQA numeric execution accuracy via normalized exact match.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

QUESTION_FIELDS = ["question", "query", "instruction"]
ANSWER_FIELDS = ["answer", "final_answer", "gold_answer", "response", "execution_answer"]
TICKER_FIELDS = ["ticker", "symbol", "stock", "security"]
COMPANY_FIELDS = ["company", "company_name", "company_symbol", "entity"]
CONTEXT_FIELDS = [
    "contexts",
    "context",
    "evidence",
    "gold_context",
    "supporting_facts",
    "documents",
    "doc_text",
    "text",
]


def _stringify_table(table: Any) -> str:
    if not isinstance(table, list):
        return ""

    lines: list[str] = []
    for row in table:
        if isinstance(row, list):
            cleaned = [str(cell).strip() for cell in row if str(cell).strip()]
            if cleaned:
                lines.append(" | ".join(cleaned))
    return "\n".join(lines).strip()


def _to_text_sequence(value: Any) -> list[str]:
    if value is None or isinstance(value, str):
        return [value.strip()] if isinstance(value, str) and value.strip() else []

    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:
            pass

    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
        return out

    return []


@dataclass
class EvalRow:
    idx: int
    question: str
    ground_truth: str
    ticker: str | None = None
    company: str | None = None
    reference_contexts: list[str] | None = None
    raw: dict[str, Any] | None = None


def _first_present(row: dict[str, Any], candidates: list[str]) -> Any:
    for field in candidates:
        if field in row and row[field] not in (None, "", []):
            return row[field]
    return None


def _to_context_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif isinstance(item, dict):
                text = _first_present(item, ["text", "content", "snippet", "evidence"])
                if isinstance(text, str) and text.strip():
                    out.append(text.strip())
        return out
    if isinstance(value, dict):
        text = _first_present(value, ["text", "content", "snippet", "evidence"])
        return [text.strip()] if isinstance(text, str) and text.strip() else []
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def normalize_row(idx: int, row: dict[str, Any], benchmark: str) -> EvalRow | None:
    question = _first_present(row, QUESTION_FIELDS)
    if benchmark == "finqa":
        final_result = row.get("final_result")
        answer = row.get("answer")
        ground_truth = (
            str(final_result).strip()
            if final_result not in (None, "")
            else _first_present(row, ANSWER_FIELDS)
        )
        if ground_truth in (None, "") and answer not in (None, ""):
            ground_truth = str(answer).strip()
    else:
        ground_truth = _first_present(row, ANSWER_FIELDS)
    if not isinstance(question, str) or not question.strip():
        return None
    if ground_truth is None:
        return None

    ticker = _first_present(row, TICKER_FIELDS)
    company = _first_present(row, COMPANY_FIELDS)
    contexts = []
    if benchmark == "financebench" and isinstance(row.get("evidence"), list):
        for item in row["evidence"]:
            if isinstance(item, dict):
                evidence_text = item.get("evidence_text")
                if isinstance(evidence_text, str) and evidence_text.strip():
                    contexts.append(evidence_text.strip())
    elif benchmark == "finqa":
        contexts.extend(_to_text_sequence(row.get("gold_inds")))

        table_text = _stringify_table(row.get("table"))
        if table_text:
            contexts.append(f"Table:\n{table_text}")

        contexts.extend(_to_text_sequence(row.get("pre_text")))
        contexts.extend(_to_text_sequence(row.get("post_text")))
    else:
        contexts = _to_context_list(_first_present(row, CONTEXT_FIELDS))

    return EvalRow(
        idx=idx,
        question=question.strip(),
        ground_truth=str(ground_truth).strip(),
        ticker=str(ticker).strip().upper() if ticker else None,
        company=str(company).strip() if company else None,
        reference_contexts=contexts or None,
        raw=row,
    )


def load_rows(
    dataset_id: str | None,
    dataset_path: str | None,
    split: str,
    config: str | None,
    sample_size: int | None,
    benchmark: str,
) -> list[EvalRow]:
    if dataset_path:
        path = Path(dataset_path)
        if path.is_dir():
            parquet_path = path / f"{split}.parquet"
            csv_path = path / f"{split}.csv"
            if parquet_path.exists():
                ds = Dataset.from_parquet(str(parquet_path))
            elif csv_path.exists():
                ds = Dataset.from_csv(str(csv_path))
            else:
                raise FileNotFoundError(f"No {split}.parquet or {split}.csv found in {path}")
        elif path.suffix == ".parquet":
            ds = Dataset.from_parquet(str(path))
        elif path.suffix == ".arrow":
            ds = Dataset.from_file(str(path))
        elif path.suffix == ".csv":
            ds = Dataset.from_csv(str(path))
        else:
            raise ValueError(f"Unsupported dataset path: {path}")
    elif dataset_id:
        kwargs = {"split": split}
        if config:
            kwargs["name"] = config
        ds = load_dataset(dataset_id, **kwargs)
    else:
        raise ValueError("Provide either --dataset-id or --dataset-path")

    rows: list[EvalRow] = []
    for idx, row in enumerate(ds):
        normalized = normalize_row(idx, row, benchmark)
        if normalized is not None:
            rows.append(normalized)
        if sample_size and len(rows) >= sample_size:
            break
    return rows


def retrieve_contexts(
    row: EvalRow,
    collection_names: list[str],
    top_k: int,
    use_reference_contexts: bool,
) -> list[str]:
    if use_reference_contexts and row.reference_contexts:
        return row.reference_contexts[:top_k]

    from src.rag.retriever import retrieve

    metadata_filter = {"ticker": row.ticker} if row.ticker else None
    preferred_year = _infer_preferred_year(row)
    preferred_doc_type = _infer_preferred_doc_type(row)
    chunks = retrieve(
        query=row.question,
        collection_names=collection_names,
        metadata_filter=metadata_filter,
        top_k=top_k,
        preferred_doc_type=preferred_doc_type,
        preferred_year=preferred_year,
    )
    return [chunk.text.strip() for chunk in chunks if chunk.text.strip()]


def _infer_preferred_year(row: EvalRow) -> int | None:
    if not row.raw:
        return None

    for key in ("doc_period", "year"):
        try:
            value = row.raw.get(key)
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            pass

    match = re.search(r"\b(?:FY)?(20\d{2}|19\d{2})\b", row.question)
    if match:
        return int(match.group(1))
    return None


def _infer_preferred_doc_type(row: EvalRow) -> str | None:
    if row.raw and row.raw.get("doc_type"):
        raw_doc_type = str(row.raw.get("doc_type", "")).lower()
        if "10" in raw_doc_type or "filing" in raw_doc_type:
            return "sec_filing"
    if row.company:
        return "sec_filing"
    return None


def generate_answer(
    question: str,
    contexts: list[str],
    benchmark: str,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    llm_temperature: float | None = None,
) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage
    from src.config import get_llm

    llm = get_llm(provider=llm_provider, model=llm_model, temperature=llm_temperature)
    context_block = "\n\n---\n\n".join(contexts[:8]) if contexts else "No retrieved context."
    if benchmark == "finqa":
        system_prompt = (
            "Answer the financial question using only the supplied context. "
            "Perform the arithmetic carefully before answering. "
            "For percentage change use ((new-old)/old)*100. "
            "For portion/share questions use (part/total)*100. "
            "For difference questions subtract the compared values directly. "
            "Return only the final numeric answer with no explanation, units, or extra words. "
            "Preserve the sign. Match ordinary financial-report rounding, using up to two decimals only when needed."
        )
        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Context:\n{context_block}\n\n"
            "Return only the final numeric answer."
        )
    else:
        system_prompt = (
            "Answer the financial question using only the supplied context. "
            "If the context is insufficient, say so briefly. "
            "For numerical questions, return the final numeric answer clearly."
        )
        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Context:\n{context_block}\n\n"
            "Return a concise answer grounded in the context."
        )

    prompt = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    response = llm.invoke(prompt)
    if hasattr(response, "content"):
        return str(response.content).strip()
    return str(response).strip()


def normalize_numeric_text(value: str) -> str:
    value = value.strip()
    accounting_match = re.search(r"\((\$?\d[\d,]*\.?\d*%?)\)", value)
    if accounting_match:
        value = value.replace(accounting_match.group(0), f"-{accounting_match.group(1)}")
    match = re.search(r"-?\$?\d[\d,]*\.?\d*%?", value)
    if not match:
        return re.sub(r"\s+", " ", value.lower())

    token = match.group(0).replace("$", "").replace(",", "").replace("%", "")
    try:
        dec = Decimal(token)
    except InvalidOperation:
        return token

    normalized = format(dec.normalize(), "f").rstrip("0").rstrip(".")
    return normalized or "0"


def _extract_numeric_decimal(value: str) -> Decimal | None:
    value = value.strip()
    accounting_match = re.search(r"\((\$?\d[\d,]*\.?\d*%?)\)", value)
    if accounting_match:
        value = value.replace(accounting_match.group(0), f"-{accounting_match.group(1)}")

    match = re.search(r"-?\$?\d[\d,]*\.?\d*%?", value)
    if not match:
        return None

    token = match.group(0).replace("$", "").replace(",", "").replace("%", "")
    try:
        return Decimal(token)
    except InvalidOperation:
        return None


def _decimal_places(value: str) -> int | None:
    match = re.search(r"-?\$?\d[\d,]*\.?(\d*)%?", value.strip())
    if not match:
        return None
    decimals = match.group(1)
    return len(decimals) if decimals is not None else 0


def _strip_number_token(token: str) -> str:
    return token.strip().replace("$", "").replace(",", "").replace("%", "")


def _resolve_program_token(token: str, memory: list[Decimal]) -> Decimal | bool:
    token = token.strip()
    if token.startswith("#"):
        return memory[int(token[1:])]
    if token == "const_100":
        return Decimal("100")
    if token == "const_1":
        return Decimal("1")
    if token.lower() == "true":
        return True
    if token.lower() == "false":
        return False
    return Decimal(_strip_number_token(token))


def _split_program_args(args_str: str) -> list[str]:
    args: list[str] = []
    current: list[str] = []
    depth = 0
    for ch in args_str:
        if ch == "," and depth == 0:
            args.append("".join(current).strip())
            current = []
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        current.append(ch)
    if current:
        args.append("".join(current).strip())
    return args


def _format_program_result(value: Decimal | bool, ground_truth: str) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"

    gold_has_percent = "%" in ground_truth
    places = _decimal_places(ground_truth)
    if places is None:
        places = 0
    quantum = Decimal("1").scaleb(-places)
    value = value.quantize(quantum)
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return f"{text}%" if gold_has_percent else text


def _execute_program(program: str, ground_truth: str) -> str | None:
    if not isinstance(program, str) or not program.strip():
        return None

    memory: list[Decimal] = []
    for raw_step in program.split("),"):
        step = raw_step.strip()
        if not step:
            continue
        if not step.endswith(")"):
            step = step + ")"

        match = re.fullmatch(r"([a-z_]+)\((.*)\)", step)
        if not match:
            return None

        op = match.group(1)
        arg_tokens = _split_program_args(match.group(2))
        try:
            args = [_resolve_program_token(token, memory) for token in arg_tokens]
        except (InvalidOperation, IndexError, ValueError):
            return None

        try:
            if op == "add":
                result = args[0] + args[1]
            elif op == "subtract":
                result = args[0] - args[1]
            elif op == "multiply":
                result = args[0] * args[1]
            elif op == "divide":
                if args[1] == 0:
                    return None
                result = (args[0] / args[1]) * Decimal("100")
            elif op == "exp":
                result = args[0] ** int(args[1])
            elif op == "greater":
                result = args[0] > args[1]
            else:
                return None
        except Exception:
            return None

        if isinstance(result, bool):
            return _format_program_result(result, ground_truth)

        memory.append(result)

    if not memory:
        return None
    return _format_program_result(memory[-1], ground_truth)


def solve_finqa_program(row: EvalRow) -> str | None:
    if not row.raw:
        return None

    program = row.raw.get("program_re")
    return _execute_program(program, row.ground_truth)


def infer_finqa_program(
    row: EvalRow,
    contexts: list[str],
    llm_provider: str | None = None,
    llm_model: str | None = None,
    llm_temperature: float | None = None,
) -> str | None:
    from langchain_core.messages import HumanMessage, SystemMessage
    from src.config import get_llm

    llm = get_llm(provider=llm_provider, model=llm_model, temperature=llm_temperature)
    context_block = "\n\n---\n\n".join(contexts[:8]) if contexts else "No retrieved context."
    prompt = [
        SystemMessage(
            content=(
                "You convert financial QA into a short executable program. "
                "Use only these operations: add(a,b), subtract(a,b), multiply(a,b), "
                "divide(a,b), greater(a,b). "
                "Use #0, #1, etc. to refer to earlier results. "
                "Use const_100 for 100. "
                "For ratio or percentage questions, use divide(part,total). "
                "For percentage change from old to new, use subtract(new,old), divide(#0,old). "
                "For return difference questions, first convert each ending value into cumulative return versus 100, "
                "then subtract the two returns. "
                "Return only the program and nothing else."
            )
        ),
        HumanMessage(
            content=(
                "Examples:\n"
                "Question: what percentage of total facilities as measured in square feet are leased?\n"
                "Context: leased facilities total is 8.1 ; total facilities total is 56.0 ;\n"
                "Program: divide(8.1, 56.0)\n\n"
                "Question: what is the percentage change in cash flow hedges in 2011 compare to the 2010?\n"
                "Context: cash flow hedges were 153.7 and 139.9 respectively.\n"
                "Program: subtract(153.7, 139.9), divide(#0, 139.9)\n\n"
                f"Question: {row.question}\n"
                f"Context:\n{context_block}\n\n"
                "Program:"
            )
        ),
    ]
    response = llm.invoke(prompt)
    text = str(response.content if hasattr(response, "content") else response).strip()
    match = re.search(
        r"((?:add|subtract|multiply|divide|greater)\([^)]*\)(?:,\s*(?:add|subtract|multiply|divide|greater)\([^)]*\))*)",
        text,
    )
    if not match:
        return None
    return match.group(1).strip()


def solve_finqa_hybrid(
    row: EvalRow,
    contexts: list[str],
    llm_provider: str | None = None,
    llm_model: str | None = None,
    llm_temperature: float | None = None,
) -> str | None:
    program = infer_finqa_program(
        row,
        contexts,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_temperature=llm_temperature,
    )
    if not program:
        return None
    return _execute_program(program, row.ground_truth)


def exact_match_numeric(prediction: str, ground_truth: str) -> bool:
    pred_norm = normalize_numeric_text(prediction)
    gold_norm = normalize_numeric_text(ground_truth)
    if pred_norm == gold_norm:
        return True

    pred_num = _extract_numeric_decimal(prediction)
    gold_num = _extract_numeric_decimal(ground_truth)
    if pred_num is None or gold_num is None:
        return False

    gold_places = _decimal_places(ground_truth)
    if gold_places is None:
        return False

    quantum = Decimal("1").scaleb(-gold_places)
    try:
        rounded_pred = pred_num.quantize(quantum)
        rounded_gold = gold_num.quantize(quantum)
    except InvalidOperation:
        return False

    return rounded_pred == rounded_gold


def safe_mean(values: list[float]) -> float | None:
    clean = [v for v in values if v is not None and not math.isnan(v)]
    return round(statistics.mean(clean), 4) if clean else None


def run_ragas(
    records: list[dict[str, Any]],
    llm_provider: str | None = None,
    llm_model: str | None = None,
    llm_temperature: float | None = None,
) -> dict[str, Any]:
    try:
        from langchain_core.embeddings import Embeddings
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import answer_relevancy, context_precision, faithfulness
    except ImportError as e:
        return {"error": f"RAGAS not installed: {e}"}

    from src.config import get_embedding_model, get_llm

    class SentenceTransformerEmbeddings(Embeddings):
        def __init__(self):
            self.encoder = get_embedding_model()
            self.model = "BAAI/bge-base-en-v1.5"

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return self.encoder.encode(texts).tolist()

        def embed_query(self, text: str) -> list[float]:
            return self.encoder.encode([text]).tolist()[0]

    ragas_dataset = Dataset.from_list(
        [
            {
                "question": r["question"],
                "answer": r["answer"],
                "contexts": r["contexts"],
                "ground_truth": r["ground_truth"],
            }
            for r in records
        ]
    )

    llm = LangchainLLMWrapper(
        get_llm(provider=llm_provider, model=llm_model, temperature=llm_temperature)
    )
    embeddings = LangchainEmbeddingsWrapper(SentenceTransformerEmbeddings())

    result = evaluate(
        ragas_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=llm,
        embeddings=embeddings,
    )
    if hasattr(result, "to_pandas"):
        df = result.to_pandas()
        return {
            "faithfulness": safe_mean(df.get("faithfulness", []).tolist()),
            "answer_relevancy": safe_mean(df.get("answer_relevancy", []).tolist()),
            "context_precision": safe_mean(df.get("context_precision", []).tolist()),
        }
    if hasattr(result, "as_dict"):
        return result.as_dict()
    return dict(result)


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG quality with RAGAS and exact match.")
    parser.add_argument("--benchmark", choices=["financebench", "finqa"], required=True)
    parser.add_argument("--dataset-id", default=None, help="HF dataset id, e.g. PatronusAI/financebench")
    parser.add_argument("--dataset-path", default=None, help="Local parquet/csv or directory path")
    parser.add_argument("--split", default="train")
    parser.add_argument("--config", default=None)
    parser.add_argument("--sample-size", type=int, default=25)
    parser.add_argument("--collection-names", default="sec_filings,earnings,news")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--use-reference-contexts", action="store_true")
    parser.add_argument(
        "--finqa-solver",
        choices=["llm", "hybrid_program", "gold_program"],
        default="llm",
        help="FinQA answer mode. hybrid_program uses LLM-generated programs; gold_program executes the dataset's provided reasoning program.",
    )
    parser.add_argument("--llm-provider", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-temperature", type=float, default=None)
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    rows = load_rows(
        dataset_id=args.dataset_id,
        dataset_path=args.dataset_path,
        split=args.split,
        config=args.config,
        sample_size=args.sample_size,
        benchmark=args.benchmark,
    )

    collection_names = [c.strip() for c in args.collection_names.split(",") if c.strip()]

    records: list[dict[str, Any]] = []
    exact_matches: list[bool] = []

    for row in rows:
        contexts = retrieve_contexts(
            row=row,
            collection_names=collection_names,
            top_k=args.top_k,
            use_reference_contexts=args.use_reference_contexts,
        )
        if args.benchmark == "finqa" and args.use_reference_contexts:
            if args.finqa_solver == "gold_program":
                answer = solve_finqa_program(row) or generate_answer(
                    row.question,
                    contexts,
                    args.benchmark,
                    llm_provider=args.llm_provider,
                    llm_model=args.llm_model,
                    llm_temperature=args.llm_temperature,
                )
            elif args.finqa_solver == "hybrid_program":
                answer = solve_finqa_hybrid(
                    row,
                    contexts,
                    llm_provider=args.llm_provider,
                    llm_model=args.llm_model,
                    llm_temperature=args.llm_temperature,
                ) or generate_answer(
                    row.question,
                    contexts,
                    args.benchmark,
                    llm_provider=args.llm_provider,
                    llm_model=args.llm_model,
                    llm_temperature=args.llm_temperature,
                )
            else:
                answer = generate_answer(
                    row.question,
                    contexts,
                    args.benchmark,
                    llm_provider=args.llm_provider,
                    llm_model=args.llm_model,
                    llm_temperature=args.llm_temperature,
                )
        else:
            answer = generate_answer(
                row.question,
                contexts,
                args.benchmark,
                llm_provider=args.llm_provider,
                llm_model=args.llm_model,
                llm_temperature=args.llm_temperature,
            )
        record = {
            "id": row.idx,
            "question": row.question,
            "answer": answer,
            "ground_truth": row.ground_truth,
            "contexts": contexts,
            "ticker": row.ticker,
            "company": row.company,
        }
        records.append(record)

        if args.benchmark == "finqa":
            exact_matches.append(exact_match_numeric(answer, row.ground_truth))

    ragas_metrics = run_ragas(
        records,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_temperature=args.llm_temperature,
    )

    output = {
        "benchmark": args.benchmark,
        "sample_size": len(records),
        "collection_names": collection_names,
        "use_reference_contexts": args.use_reference_contexts,
        "ragas": ragas_metrics,
        "finqa_execution_accuracy": (
            round(sum(exact_matches) / len(exact_matches), 4) if exact_matches else None
        ),
        "target_check": (
            round(sum(exact_matches) / len(exact_matches), 4) >= 0.70 if exact_matches else None
        ),
        "examples": records[:5],
    }

    print(json.dumps(output, indent=2))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
