#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import pipeline


def map_label(x):
    s = str(x).strip().lower()
    if s in {"0", "-1"} or "neg" in s:
        return "negative"
    if s in {"1"} or "neu" in s:
        return "neutral"
    if s in {"2"} or "pos" in s:
        return "positive"
    raise ValueError(f"Unknown label format: {x}")


def pred_label(s):
    s = s.lower()
    if "neg" in s:
        return "negative"
    if "neu" in s:
        return "neutral"
    if "pos" in s:
        return "positive"
    raise ValueError(f"Unknown model label: {s}")


def eval_model(texts, y_true, model_name, batch_size=32):
    clf = pipeline("text-classification", model=model_name, tokenizer=model_name, truncation=True, max_length=256)
    y_pred = [pred_label(p["label"]) for p in clf(texts, batch_size=batch_size)]
    acc = accuracy_score(y_true, y_pred)
    f1w = f1_score(y_true, y_pred, average="weighted")
    return float(acc), float(f1w)


def load_phrasebank(config_name, split, text_col, label_col, max_rows=None):
    ds = load_dataset("takala/financial_phrasebank", config_name, split=split)
    if max_rows:
        ds = ds.select(range(min(max_rows, len(ds))))
    return [str(x) for x in ds[text_col]], [map_label(x) for x in ds[label_col]]


def load_fiqa(split, text_col, label_col, max_rows=None):
    ds = load_dataset("TheFinAI/fiqa-sentiment-classification", split=split)
    if max_rows:
        ds = ds.select(range(min(max_rows, len(ds))))
    return [str(x) for x in ds[text_col]], [map_label(x) for x in ds[label_col]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-model", default="ProsusAI/finbert")
    ap.add_argument("--finetuned-model", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-col", default="sentence")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()

    phrase_cfgs = ["sentences_allagree", "sentences_75agree", "sentences_66agree", "sentences_50agree"]

    print("=== Metric 1: Sentiment Classification Accuracy ===")
    for cfg in phrase_cfgs:
        texts, y = load_phrasebank(cfg, args.split, args.text_col, args.label_col, args.max_rows)
        b_acc, b_f1 = eval_model(texts, y, args.baseline_model, args.batch_size)
        t_acc, t_f1 = eval_model(texts, y, args.finetuned_model, args.batch_size)
        print(f"[PhraseBank:{cfg}] n={len(texts)}")
        print(f"  baseline  acc={b_acc:.4f} f1w={b_f1:.4f}")
        print(f"  finetuned acc={t_acc:.4f} f1w={t_f1:.4f}")
        print(f"  delta     acc={t_acc-b_acc:+.4f} f1w={t_f1-b_f1:+.4f}")

    texts, y = load_fiqa(args.split, args.text_col, args.label_col, args.max_rows)
    b_acc, b_f1 = eval_model(texts, y, args.baseline_model, args.batch_size)
    t_acc, t_f1 = eval_model(texts, y, args.finetuned_model, args.batch_size)

    print(f"\n[FiQA-SA] n={len(texts)}")
    print(f"  baseline  acc={b_acc:.4f} f1w={b_f1:.4f}")
    print(f"  finetuned acc={t_acc:.4f} f1w={t_f1:.4f}")
    print(f"  delta     acc={t_acc-b_acc:+.4f} f1w={t_f1-b_f1:+.4f}")
    print("  target: finetuned FiQA weighted F1 >= 0.88")


if __name__ == "__main__":
    main()
