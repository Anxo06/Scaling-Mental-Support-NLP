#!/usr/bin/env python3
"""
finetune_t5.py
────────────────
Finetune a (Mono)T5-style model for binary relevance:
Input  : "Query: <question> Document: <response> Relevant:"
Target : "true" | "false"

Expected data formats (train/valid)
-----------------------------------
- JSONL: one object per line with fields: {"question": str, "response": str, "label": (1|0|true|false|"true"|"false")}
- JSON : {"data":[ {...}, {...}, ... ]} with the same fields per item
- CSV/TSV: header must include question,response,label

Labels are mapped to {"true","false"} with case-insensitive parsing of 1/0/true/false/yes/no.
"""

import argparse
import json
import csv
from pathlib import Path
from typing import Dict, List, Any

import datasets
from datasets import Dataset
import numpy as np
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# ────────────────────────────── IO helpers ──────────────────────────────
def _parse_label(x) -> str:
    if isinstance(x, bool):
        return "true" if x else "false"
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y", "relevant"}:
        return "true"
    if s in {"0", "false", "no", "n", "not_relevant", "irrelevant"}:
        return "false"
    raise ValueError(f"Unrecognized label: {x}")

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append({
                "question": obj["question"],
                "response": obj["response"],
                "label": _parse_label(obj["label"]),
            })
    return rows

def _read_json(path: Path) -> List[Dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    data = obj["data"] if "data" in obj else obj
    rows = []
    for r in data:
        rows.append({
            "question": r["question"],
            "response": r["response"],
            "label": _parse_label(r["label"]),
        })
    return rows

def _read_table(path: Path, delimiter: str) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for r in reader:
            rows.append({
                "question": r["question"],
                "response": r["response"],
                "label": _parse_label(r["label"]),
            })
    return rows

def load_split(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _read_jsonl(path)
    if suffix == ".json":
        return _read_json(path)
    if suffix == ".csv":
        return _read_table(path, ",")
    if suffix in {".tsv", ".tab"}:
        return _read_table(path, "\t")
    raise ValueError(f"Unsupported file format: {path}")

# ───────────────────────────── tokenization ─────────────────────────────
def build_input_text(question: str, response: str, template: str) -> str:
    # default template is MonoT5-like
    return template.format(question=question, response=response)

def prepare_dataset(examples: List[Dict[str, Any]], tokenizer, max_source_len: int, max_target_len: int, template: str) -> Dataset:
    texts = [build_input_text(ex["question"], ex["response"], template) for ex in examples]
    labels = [ex["label"] for ex in examples]  # "true"/"false"
    model_inputs = tokenizer(texts, max_length=max_source_len, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels_tokenized = tokenizer(labels, max_length=max_target_len, truncation=True)

    model_inputs["labels"] = labels_tokenized["input_ids"]
    return Dataset.from_dict(model_inputs)

# ────────────────────────────── metrics ──────────────────────────────
def compute_metrics_fn(tokenizer):
    true_id = tokenizer.convert_tokens_to_ids("true")
    false_id = tokenizer.convert_tokens_to_ids("false")

    def postprocess(pred_ids: np.ndarray) -> List[str]:
        # Take the first generated token; fall back to argmax over sequence if needed
        outs = []
        for row in pred_ids:
            tok = row[0] if len(row) > 0 else -1
            if tok == true_id:
                outs.append("true")
            elif tok == false_id:
                outs.append("false")
            else:
                # fallback: find first occurrence of either token
                if true_id in row:
                    outs.append("true")
                elif false_id in row:
                    outs.append("false")
                else:
                    # final fallback: treat as false
                    outs.append("false")
        return outs

    def _compute(eval_pred):
        pred_ids, label_ids = eval_pred
        # Convert label ids back to strings for exact match on "true"/"false"
        labels = []
        for row in label_ids:
            # Drop padding (-100) then decode
            toks = [t for t in row if t != -100]
            text = tokenizer.decode(toks, skip_special_tokens=True).strip().lower()
            labels.append("true" if "true" in text and "false" not in text else ("false" if "false" in text else text))

        preds = postprocess(pred_ids)
        y_true = np.array([1 if x == "true" else 0 for x in labels])
        y_pred = np.array([1 if x == "true" else 0 for x in preds])

        acc = (y_true == y_pred).mean().item()
        # precision/recall/f1 (binary, positive="true")
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    return _compute

# ──────────────────────────────── CLI ────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser(description="Finetune a T5/MonoT5-style model for binary relevance.")
    # Data
    ap.add_argument("--train-file", type=Path, required=True, help="Path to train split (jsonl/json/csv/tsv).")
    ap.add_argument("--valid-file", type=Path, required=True, help="Path to validation split (jsonl/json/csv/tsv).")

    # Model & tokenization
    ap.add_argument("--model-name", default="castorini/monot5-base-msmarco-10k", help="Base T5/MonoT5 model.")
    ap.add_argument("--max-source-len", type=int, default=512)
    ap.add_argument("--max-target-len", type=int, default=2)
    ap.add_argument("--template", default="Query: {question} Document: {response} Relevant:",
                    help="Input template. Use {question} and {response} placeholders.")

    # Training
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")

    # Logging / eval / saving
    ap.add_argument("--eval-steps", type=int, default=500)
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--logging-steps", type=int, default=50)
    ap.add_argument("--save-total-limit", type=int, default=3)
    ap.add_argument("--load-best", action="store_true", help="Load best ckpt at end (metric=f1).")
    ap.add_argument("--push-to-hub", action="store_true")
    return ap.parse_args()

# ───────────────────────────────── main ─────────────────────────────────
def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_rows = load_split(args.train_file)
    valid_rows = load_split(args.valid_file)

    # Tokenizer / model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)

    # Prepare datasets
    train_ds = prepare_dataset(train_rows, tokenizer, args.max_source_len, args.max_target_len, args.template)
    valid_ds = prepare_dataset(valid_rows, tokenizer, args.max_source_len, args.max_target_len, args.template)

    # Collator
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest")

    # Training args
    targs = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad-accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.load_best,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=args.fp16,
        bf16=args.bf16,
        seed=args.seed,
        report_to=["none"],  # set to ["wandb"] etc. if desired
        predict_with_generate=True,
    )

    compute_metrics = compute_metrics_fn(tokenizer)

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"✅ Training finished. Model saved to: {args.output_dir}")

if __name__ == "__main__":
    main()


"""
Example usage:
Basic run (MonoT5 base, fp16, 3 epochs):
python finetune_t5.py \
  --train-file ./data/train.jsonl \
  --valid-file ./data/valid.jsonl \
  --model-name castorini/monot5-base-msmarco-10k \
  --output-dir ./experiments/t5/monot5-relevance-all \
  --batch-size 16 \
  --grad-accum 2 \
  --epochs 3 \
  --lr 3e-5 \
  --fp16
"""