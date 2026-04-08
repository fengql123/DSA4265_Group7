#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import Features, Value, concatenate_datasets, load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def normalize_label(value):
    if isinstance(value, str):
        value = value.strip().lower()
        return LABEL2ID[value]

    value = float(value)
    if value < 0:
        return LABEL2ID["negative"]
    if value > 0:
        return LABEL2ID["positive"]
    return LABEL2ID["neutral"]


def standardize_dataset(ds):
    features = Features({
        "text": Value("string"),
        "label": Value("int64"),
    })
    return ds.cast(features)


def load_fiqa():
    ds = load_dataset("TheFinAI/fiqa-sentiment-classification")
    train_ds = ds["train"]
    val_ds = ds["valid"]
    test_ds = ds["test"]

    def map_row(row):
        return {
            "text": str(row["sentence"]),
            "label": int(normalize_label(row["score"])),
        }

    return (
        standardize_dataset(train_ds.map(map_row, remove_columns=train_ds.column_names)),
        standardize_dataset(val_ds.map(map_row, remove_columns=val_ds.column_names)),
        standardize_dataset(test_ds.map(map_row, remove_columns=test_ds.column_names)),
    )


def load_phrasebank(config_name="sentences_allagree", seed=42):
    ds = load_dataset(
        "takala/financial_phrasebank",
        config_name,
        trust_remote_code=True,
    )["train"]

    split = ds.train_test_split(test_size=0.2, seed=seed)
    train_ds = split["train"]
    test_ds = split["test"]

    def map_row(row):
        return {
            "text": str(row["sentence"]),
            "label": int(row["label"]),
        }

    mapped_train = train_ds.map(map_row, remove_columns=train_ds.column_names)
    mapped_test = test_ds.map(map_row, remove_columns=test_ds.column_names)

    return standardize_dataset(mapped_train), standardize_dataset(mapped_test)


def tokenize_dataset(ds, tokenizer, max_length=256):
    def tok(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    return ds.map(tok, batched=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="models/finbert-mixed")
    parser.add_argument("--model-name", default="ProsusAI/finbert")
    parser.add_argument("--phrasebank-config", default="sentences_allagree")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    args = parser.parse_args()

    fiqa_train, fiqa_val, fiqa_test = load_fiqa()
    pb_train, pb_test = load_phrasebank(args.phrasebank_config)

    train_ds = concatenate_datasets([fiqa_train, pb_train]).shuffle(seed=42)
    val_ds = concatenate_datasets([fiqa_val, pb_test]).shuffle(seed=42)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_tok = tokenize_dataset(train_ds, tokenizer)
    val_tok = tokenize_dataset(val_ds, tokenizer)
    fiqa_test_tok = tokenize_dataset(fiqa_test, tokenizer)
    pb_test_tok = tokenize_dataset(pb_test, tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="weighted_f1",
        greater_is_better=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    fiqa_metrics = trainer.evaluate(fiqa_test_tok)
    pb_metrics = trainer.evaluate(pb_test_tok)

    summary = {
        "model": args.model_name,
        "output_dir": args.output_dir,
        "phrasebank_config": args.phrasebank_config,
        "fiqa_test": fiqa_metrics,
        "phrasebank_test": pb_metrics,
    }

    out_path = Path(args.output_dir) / "mixed_training_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
