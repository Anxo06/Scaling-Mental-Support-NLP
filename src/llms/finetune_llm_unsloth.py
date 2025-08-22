#!/usr/bin/env python3
"""
finetune_llm.py
───────────────────────
Finetune a chat LLM (e.g., Llama-3.x) with UnsLoTH + LoRA for binary relevance.
Task: Given (QUESTION, RESPONSE), generate exactly: "true" or "false".

Input format (JSONL), one example per line:
{
  "question": "...",
  "response": "...",
  "label": "true" | "false" | 1 | 0
}

Notes
-----
- Fixed prompt (same logic you use at inference).
- Uses UnsLoTH 4-bit loading by default + LoRA.
- Writes a standard HF checkpoint (merged adapters optional).

Dependencies
------------
pip install unsloth torch datasets transformers trl
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

from datasets import Dataset
import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments

# ------------------------- Fixed prompt (as requested) -------------------------
SYSTEM_PROMPT = (
    "You are an experienced mental health professional. "
    "Given a QUESTION and a single RESPONSE, decide whether the response is relevant to the mental health question."
)

def build_zeroshot_messages(question: str, response: str) -> List[Dict[str, str]]:
    user_prompt = (
        "Given a question and a single response, decide whether the response is MENTAL-HEALTH-RELEVANT to the question.\n\n"
        "Guidelines for RELEVANT (True):\n"
        "- Addresses or acknowledges the user's mental health concern or question.\n"
        "- Offers empathic, supportive, or informational guidance appropriate to mental health / counseling.\n"
        "- Provides actionable advice or resources that are relevant to the user's issue.\n\n"
        "Guidelines for NOT RELEVANT (False):\n"
        "- Off-topic, nonsense, spam, purely social chit-chat unrelated to the question.\n"
        "- Talks about unrelated activities without linking to mental health.\n"
        "- Generic motivational lines that do not meaningfully connect to the user's issue.\n\n"
        f"Question: {question}\n"
        f"Response: {response}\n"
        "Is the response relevant?\n"
        "Respond with a single token: true or false. Do not include anything else."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]

# ------------------------------- Data loading ---------------------------------
def _parse_label(x) -> str:
    if isinstance(x, bool):
        return "true" if x else "false"
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y", "relevant"}:
        return "true"
    if s in {"0", "false", "no", "n", "irrelevant", "not_relevant"}:
        return "false"
    raise ValueError(f"Unrecognized label: {x}")

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append({
                "question": obj["question"],
                "response": obj["response"],
                "label": _parse_label(obj["label"]),  # "true"/"false"
            })
    return rows

def make_supervised_texts(rows, tokenizer, chat_template: str):
    """Convert (question,response,label) into a single 'text' per example:
       chat(template(messages)) + assistant turn with the gold label."""
    tokenizer = get_chat_template(
        tokenizer,
        chat_template=chat_template,
        mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
    )
    texts = []
    for r in rows:
        messages = build_zeroshot_messages(r["question"], r["response"])
        # Render chat and add assistant answer (gold label)
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        rendered += r["label"]  # append the gold label as assistant content
        texts.append(rendered)
    return texts

# ---------------------------------- CLI ---------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Finetune an LLM with UnsLoTH + LoRA for True/False relevance.")
    # Data
    ap.add_argument("--train-file", type=Path, required=True, help="Training JSONL.")
    ap.add_argument("--valid-file", type=Path, help="Validation JSONL (optional).")

    # Model / LoRA / quantization
    ap.add_argument("--base-model", default="unsloth/Llama-3.1-8B-Instruct-bnb-4bit",
                    help="Base model or UnsLoTH 4-bit variant.")
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--target-modules", nargs="*", default=None,
                    help="Optional list of target modules; if omitted, UnsLoTH chooses.")
    ap.add_argument("--use-4bit", action="store_true", help="Force 4-bit loading (usually auto for UnsLoTH variants).")

    # Training
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=4, help="per-device batch size")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)  # LoRA often uses a higher LR
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")

    # Logging / eval / save
    ap.add_argument("--logging-steps", type=int, default=50)
    ap.add_argument("--eval-steps", type=int, default=500)
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--save-total-limit", type=int, default=2)
    ap.add_argument("--load-best", action="store_true", help="Load best ckpt (metric=eval_loss) at end.")
    ap.add_argument("--report-to", nargs="*", default=["none"], help="['wandb'] etc.")
    ap.add_argument("--chat-template", default="llama-3", help="UnsLoTH chat template id.")
    ap.add_argument("--packing", action="store_true", help="Pack multiple samples per sequence.")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--merge-and-save", action="store_true", help="Merge LoRA into base weights at the end.")
    return ap.parse_args()

# --------------------------------- main ---------------------------------------
def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔹 Device: {device}")

    # 1) Load base model (4-bit if UnsLoTH variant)
    print("🔹 Loading base model")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_len,
        load_in_4bit=args.use_4bit or "4bit" in args.base_model.lower(),  # heuristic
        dtype=None,
    )

    # 2) Convert to LoRA PEFT
    print("🔹 Applying LoRA")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=args.target_modules,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing=args.gradient_checkpointing,
        random_state=args.seed,
    )

    # 3) Build training/validation datasets (rendered chat strings)
    print("🔹 Preparing datasets")
    train_rows = load_jsonl(args.train_file)
    train_texts = make_supervised_texts(train_rows, tokenizer, args.chat_template)
    train_ds = Dataset.from_dict({"text": train_texts})

    eval_ds = None
    if args.valid_file and args.valid_file.exists():
        valid_rows = load_jsonl(args.valid_file)
        valid_texts = make_supervised_texts(valid_rows, tokenizer, args.chat_template)
        eval_ds = Dataset.from_dict({"text": valid_texts})

    # 4) TrainingArguments
    targs = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        evaluation_strategy=("steps" if eval_ds is not None else "no"),
        eval_steps=(args.eval_steps if eval_ds is not None else None),
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=(args.load_best and eval_ds is not None),
        metric_for_best_model="loss",
        greater_is_better=False,
        report_to=args.report_to,
        fp16=args.fp16,
        bf16=args.bf16,
        seed=args.seed,
    )

    # 5) Trainer (SFT style on concatenated chat + gold label)
    print("🔹 Starting SFT")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
        packing=args.packing,  # when True packs multiple samples into max_seq_len
        max_seq_length=args.max_seq_len,
    )

    trainer.train()
    print("✅ Training finished.")

    # 6) Save adapters (and optionally merge)
    print("💾 Saving adapters to", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.merge_and_save:
        print("🔁 Merging LoRA adapters into base model …")
        # Reload base in full precision for merge (optional optimization)
        base_model, _ = FastLanguageModel.from_pretrained(
            model_name=args.base_model,
            max_seq_length=args.max_seq_len,
            load_in_4bit=False,
            dtype=torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32),
        )
        merged = FastLanguageModel.merge_lora(base_model, model)
        merged.save_pretrained(args.output_dir / "merged")
        print("✅ Merged model saved to:", args.output_dir / "merged")

    print("🏁 Done.")

if __name__ == "__main__":
    main()


"""
Example usage:
Standard LoRA SFT (4-bit base, fp16, packing on)
python finetune_llm_unsloth.py \
  --train-file ./data/relevance_train.jsonl \
  --valid-file ./data/relevance_valid.jsonl \
  --base-model unsloth/Llama-3.1-8B-Instruct-bnb-4bit \
  --output-dir ./experiments/llm/llama3-8b-relevance-all \
  --epochs 2 \
  --batch-size 4 \
  --grad-accum 8 \
  --lr 2e-4 \
  --max-seq-len 2048 \
  --packing \
  --fp16
"""