#!/usr/bin/env python3
"""
bert_finetune.py
───────────────────────
Fine-tune a BERT-like encoder for binary relevance:
(question, response) → {true|false}.

Input format (JSONL), one example per line:
{
  "question": "...",
  "response": "...",
  "label": "true" | "false" | 1 | 0
}

Outputs a standard HuggingFace checkpoints directory.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


# ─────────────────────────── I/O ───────────────────────────
def _parse_label(x: Any) -> int:
    if isinstance(x, bool):
        return 1 if x else 0
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y", "relevant"}:
        return 1
    if s in {"0", "false", "no", "n", "irrelevant", "not_relevant"}:
        return 0
    raise ValueError(f"Unrecognized label: {x}")

def load_jsonl_pairs(path: Path) -> Dataset:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            rows.append({
                "question": item["question"],
                "response": item["response"],
                "label_id": _parse_label(item["label"]),
            })
    return Dataset.from_list(rows)


# ─────────────────────── tokenization ───────────────────────
def make_tokenize_fn(tokenizer, max_len: int):
    def tokenize_fn(batch):
        enc = tokenizer(
            batch["question"],
            batch["response"],
            truncation=True,
            max_length=max_len,
        )
        enc["labels"] = batch["label_id"]  # HF expects the key "labels"
        return enc
    return tokenize_fn


# ──────────────────────── metrics ──────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    try:
        probs = torch.tensor(logits).softmax(-1)[:, 1].numpy()
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = float("nan")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}


# ────────────────────────── CLI ────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser(description="Fine-tune BERT for question–response relevance (binary).")

    # Data
    ap.add_argument("--train-file", type=Path, required=True, help="JSONL with question/response/label.")
    ap.add_argument("--valid-file", type=Path, help="Optional JSONL validation split.")

    # Model/tokenizer
    ap.add_argument("--model-name", default="google-bert/bert-base-uncased",
                    help="Base encoder (e.g., 'google-bert/bert-base-uncased', 'castorini/monobert-large-msmarco').")
    ap.add_argument("--hf-token", default=None, help="Hugging Face token if the repo requires authentication.")
    ap.add_argument("--max-len", type=int, default=384, help="Max sequence length.")

    # Training
    ap.add_argument("--output-dir", type=Path, required=True, help="Where to save checkpoints.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true", help="Enable fp16 if supported.")
    ap.add_argument("--bf16", action="store_true", help="Enable bf16 if supported.")

    # Eval/ckpt/logging
    ap.add_argument("--eval-steps", type=int, default=1000)
    ap.add_argument("--save-steps", type=int, default=1000)
    ap.add_argument("--save-total-limit", type=int, default=2)
    ap.add_argument("--logging-steps", type=int, default=200)
    ap.add_argument("--early-stopping", type=int, default=4,
                    help="Early stopping patience (eval steps). Set 0 to disable.")
    ap.add_argument("--report-to", nargs="*", default=["none"], help="e.g., ['wandb']")

    return ap.parse_args()


# ───────────────────────── main ────────────────────────────
def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Data
    print("🔹 Loading data")
    train_ds_raw = load_jsonl_pairs(args.train_file)
    valid_ds_raw = load_jsonl_pairs(args.valid_file) if args.valid_file and args.valid_file.exists() else None

    # Tokenizer & model
    print("🔹 Tokenizer & model")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=args.hf_token, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, token=args.hf_token, num_labels=2
    )

    tokenize_fn = make_tokenize_fn(tokenizer, args.max_len)
    train_ds = train_ds_raw.map(tokenize_fn, batched=True, remove_columns=["question", "response", "label_id"])
    eval_ds = (valid_ds_raw.map(tokenize_fn, batched=True, remove_columns=["question", "response", "label_id"])
               if valid_ds_raw is not None else None)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # TrainingArguments
    print("🔹 TrainingArguments")
    evaluation_strategy = "steps" if eval_ds is not None else "no"
    args_hf = TrainingArguments(
        output_dir=str(args.output_dir),
        overwrite_output_dir=True,

        # training
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,

        # evaluation/checkpointing
        evaluation_strategy=evaluation_strategy,
        eval_steps=args.eval_steps if eval_ds is not None else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=(eval_ds is not None),
        metric_for_best_model="eval_f1",  # key from compute_metrics (prefixed by "eval_")
        greater_is_better=True,

        # logging
        logging_steps=args.logging_steps,
        report_to=args.report_to,

        # misc
        seed=args.seed,
        fp16=args.fp16,
        bf16=args.bf16,
    )

    callbacks = []
    if eval_ds is not None and args.early_stopping > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping))

    print("🔹 Trainer")
    trainer = Trainer(
        model=model,
        args=args_hf,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics if eval_ds is not None else None,
        callbacks=callbacks,
    )

    print("🚀 Starting fine-tuning …")
    trainer.train()
    print("✅ Training finished.")

    # Save final artifacts
    print(f"💾 Saving to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("🏁 Done.")

if __name__ == "__main__":
    main()


"""
Example usage:
python bert_finetune.py \
  --train-file /data/all_triplets-train.jsonl \
  --valid-file /data/all_triplets-valid.jsonl \
  --model-name castorini/monobert-large-msmarco \
  --hf-token $HF_TOKEN \
  --output-dir ./models/bert/bert-relevance-all \
  --max-len 384 \
  --batch-size 32 \
  --epochs 3 \
  --lr 3e-5 \
  --eval-steps 1000 \
  --save-steps 1000 \
  --early-stopping 4 \
  --fp16
"""