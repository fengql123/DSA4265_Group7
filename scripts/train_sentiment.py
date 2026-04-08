#!/usr/bin/env python3
"""Fine-tune FinBERT for financial sentiment classification.

Examples:
    python scripts/train_sentiment.py \
      --dataset "TheFinAI/fiqa-sentiment-classification" \
      --output-dir models/finbert-fiqa

    python scripts/train_sentiment.py \
      --dataset "takala/financial_phrasebank" \
      --config "sentences_allagree" \
      --text-field sentence \
      --label-field label \
      --output-dir models/finbert-phrasebank
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
from datasets import ClassLabel, DatasetDict, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


LABEL_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune FinBERT for sentiment classification.")
    parser.add_argument(
        "--dataset",
        default="TheFinAI/fiqa-sentiment-classification",
        help="Hugging Face dataset id",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional Hugging Face dataset config name",
    )
    parser.add_argument(
        "--text-field",
        default="sentence",
        help="Column containing the input text",
    )
    parser.add_argument(
        "--label-field",
        default="label",
        help="Column containing the class label",
    )
    parser.add_argument(
        "--model-name",
        default="ProsusAI/finbert",
        help="Base model checkpoint to fine-tune",
    )
    parser.add_argument(
        "--output-dir",
        default="models/finbert-finetuned",
        help="Directory for the saved model",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap for training samples (useful for quick tests)",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def normalize_label(value) -> int:
    if isinstance(value, str):
        key = value.strip().lower()
        if key not in LABEL_TO_ID:
            raise ValueError(f"Unsupported label string: {value}")
        return LABEL_TO_ID[key]

    if isinstance(value, (int, float, np.integer, np.floating)):
        fvalue = float(value)
        if fvalue < 0:
            return LABEL_TO_ID["negative"]
        if fvalue > 0:
            return LABEL_TO_ID["positive"]
        return LABEL_TO_ID["neutral"]

    if isinstance(value, (int, np.integer)):
        ivalue = int(value)
        if ivalue in ID_TO_LABEL:
            return ivalue
        raise ValueError(f"Unsupported integer label: {value}")

    raise ValueError(f"Unsupported label type: {type(value)}")


def ensure_splits(dataset) -> DatasetDict:
    if isinstance(dataset, DatasetDict):
        if "train" in dataset and "validation" in dataset:
            return dataset
        if "train" in dataset and "test" in dataset:
            dataset["validation"] = dataset["test"]
            return dataset
        if "train" in dataset:
            split = dataset["train"].train_test_split(test_size=0.2, seed=42)
            return DatasetDict(
                {
                    "train": split["train"],
                    "validation": split["test"],
                }
            )
        raise ValueError("DatasetDict does not contain a usable train split.")

    split = dataset.train_test_split(test_size=0.2, seed=42)
    return DatasetDict({"train": split["train"], "validation": split["test"]})


def load_and_prepare_dataset(args: argparse.Namespace) -> DatasetDict:
    ds = load_dataset(args.dataset, args.config) if args.config else load_dataset(args.dataset)
    ds = ensure_splits(ds)

    if args.max_train_samples:
        ds["train"] = ds["train"].select(range(min(args.max_train_samples, len(ds["train"]))))

    if args.text_field not in ds["train"].column_names:
        raise ValueError(f"Text field '{args.text_field}' not found in dataset columns: {ds['train'].column_names}")
    if args.label_field not in ds["train"].column_names:
        raise ValueError(f"Label field '{args.label_field}' not found in dataset columns: {ds['train'].column_names}")

    def remap(example):
        label_value = example[args.label_field]
        if isinstance(ds["train"].features.get(args.label_field), ClassLabel):
            label_value = ds["train"].features[args.label_field].int2str(label_value)
        return {
            "text": str(example[args.text_field]),
            "labels": normalize_label(label_value),
        }

    keep_cols = [args.text_field, args.label_field]
    ds = ds.map(remap, remove_columns=[c for c in ds["train"].column_names if c not in keep_cols])
    return ds


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=-1)

    accuracy = float((preds == labels).mean())

    f1s = []
    for class_id in sorted(ID_TO_LABEL):
        tp = np.sum((preds == class_id) & (labels == class_id))
        fp = np.sum((preds == class_id) & (labels != class_id))
        fn = np.sum((preds != class_id) & (labels == class_id))

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        support = np.sum(labels == class_id)
        f1s.append((f1, support))

    total_support = sum(s for _, s in f1s) or 1
    weighted_f1 = sum(f1 * support for f1, support in f1s) / total_support

    return {
        "accuracy": round(accuracy, 4),
        "weighted_f1": round(float(weighted_f1), 4),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=3,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    dataset = load_and_prepare_dataset(args)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
        )

    tokenized = dataset.map(tokenize, batched=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="weighted_f1",
        greater_is_better=True,
        report_to="none",
        save_total_limit=2,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metadata = {
        "base_model": args.model_name,
        "dataset": args.dataset,
        "dataset_config": args.config,
        "text_field": args.text_field,
        "label_field": args.label_field,
        "metrics": metrics,
    }
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2))

    print("\nTraining complete.")
    print(f"Saved fine-tuned model to: {output_dir}")
    print(f"Validation metrics: {json.dumps(metrics, indent=2)}")
    print("\nTo use this model for inference, set:")
    print(f'  export SENTIMENT_MODEL_PATH="{output_dir}"')


if __name__ == "__main__":
    main()
