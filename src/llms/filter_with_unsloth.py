#!/usr/bin/env python3
# filter_with_unsloth.py
# ─────────────────────────────────────────────────────────────
# Re-rank a retrieval list with your UnsLoTH-finetuned model:
#   • Trues first (stable within-bucket), then falses.
#   • Batched inference.
#   • Optional zero-shot / few-shot prompting.
#
# Expected input JSON (from dump_initial_retrieval.py)
# {
#   "qid-1": {
#     "question": "...",
#     "cluster-id": "...",      # optional
#     "results": [
#       {"response": "...", "score": 12.3, ...},
#       ...
#     ]
#   },
#   ...
# }
# ─────────────────────────────────────────────────────────────

from __future__ import annotations
from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Callable

import argparse
import json
import re
from pathlib import Path

import torch
from tqdm import tqdm
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import LogitsProcessor, LogitsProcessorList

SYSTEM_PROMPT = (
    "You are an experienced mental health professional. "
    "Given a QUESTION and a single RESPONSE, decide whether the response is relevant to the mental health question."
)

# ───────────────────────── prompt builders ─────────────────────────
def _build_prompt(question: str, response: str) -> List[Dict[str, str]]:
    """Zero-shot prompt."""
    user_prompt = (
        "Given a question and a single response, decide whether the response is MENTAL-HEALTH-RELEVANT to the question.\n\n"
        "Guidelines for RELEVANT (True):\n"
        "- Addresses or acknowledges the user's mental health concern or question.\n"
        "- Offers empathic, supportive, or informational guidance appropriate to mental health / counseling.\n"
        "- Provides actionable advice or resources that are relevant to the user's issue.\n\n"
        "Guidelines for NOT RELEVANT (False):\n"
        "- Off-topic, nonsense, spam, purely social chat unrelated to the question.\n"
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

def _build_prompt_with_fewshot_examples(question: str, response: str) -> List[Dict[str, str]]:
    """Tiny few-shot prompt (replace with your real examples later)."""
    examples = [
        # (question, response, label)
        ("I can't sleep and feel anxious at night.", 
         "Try a simple sleep routine and consider limiting caffeine. If anxiety persists, a therapist can help.", 
         "true"),
        ("Any movie recommendations for the weekend?", 
         "You should watch the latest action film, it's awesome!", 
         "false"),
    ]
    msgs: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for q, r, lab in examples:
        msgs.append({"role": "user",
                     "content": f"Question: {q}\nResponse: {r}\nIs the response relevant?\nRespond with a single token: true or false."})
        msgs.append({"role": "assistant", "content": lab})
    # final example to classify
    msgs.append({"role": "user",
                 "content": f"Question: {question}\nResponse: {response}\nIs the response relevant?\nRespond with a single token: true or false."})
    return msgs

# ───────────────────────── core inference ─────────────────────────
@torch.no_grad()
def classify_batch(
    model,
    tokenizer,
    batch_pairs: List[Tuple[str, str]],
    build_prompt_fn: Callable[[str, str], List[Dict[str, str]]],
    logits_processor: LogitsProcessorList,
    true_id: int,
    false_id: int,
    device: str = "cuda",
) -> List[bool]:
    """Fast batched classification for a list of (question, response) pairs."""
    # 1) Build messages
    batch_messages = [build_prompt_fn(q, r) for q, r in batch_pairs]
    # 2) Render to strings (no tokenization yet)
    prompts = [
        tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )
        for message in batch_messages
    ]
    # 3) Tokenize
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=1,
        use_cache=True,
        do_sample=False,
        logits_processor=logits_processor,
    )

    last_tok = outputs[:, -1].tolist()
    return [(tid == true_id) for tid in last_tok]

def filter_with_unsloth(
    all_results: Dict,
    prompt_mode: str,
    finetuned_model: str,
    cache_dir: Optional[str],
    batch_size: int,
    max_seq_len: int = 2048,
    show_progress: bool = True,
) -> Dict:
    """Reorder each question's results: trues first (stable), then falses."""
    build_fn = _build_prompt if prompt_mode == "zeroshot" else _build_prompt_with_fewshot_examples

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=finetuned_model,
        cache_dir=cache_dir,
        max_seq_length=max_seq_len,
        dtype=None,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3",
        mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
    )

    # Allow only the next token to be " true" or " false"
    def get_allow_ids(tok):
        true_ids = tok.encode(" true", add_special_tokens=False)
        false_ids = tok.encode(" false", add_special_tokens=False)
        return true_ids[-1], false_ids[-1]

    class AllowOnly(LogitsProcessor):
        def __init__(self, allowed_ids: List[int]):
            super().__init__()
            self.allowed = torch.tensor(allowed_ids)

        def __call__(self, input_ids, scores):
            mask = torch.full_like(scores, float("-inf"))
            idx = self.allowed.to(scores.device).unsqueeze(0).expand(scores.size(0), -1)
            mask.scatter_(1, idx, scores.gather(1, idx))
            return mask

    true_id, false_id = get_allow_ids(tokenizer)
    logit_processors = LogitsProcessorList([AllowOnly([true_id, false_id])])
    FastLanguageModel.for_inference(model)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ranked = deepcopy(all_results)
    total_questions = len(ranked)

    for qi, (qid, bundle) in enumerate(ranked.items(), start=1):
        question = bundle["question"]
        responses = bundle["results"]

        if show_progress:
            print(f"[{qi}/{total_questions}] scoring {len(responses)} responses for qid={qid}")

        true_bucket, false_bucket = [], []

        # Batched loop
        for start in range(0, len(responses), batch_size):
            chunk = responses[start : start + batch_size]
            pairs = [(question, r["response"]) for r in chunk]
            preds = classify_batch(model, tokenizer, pairs, build_fn, logit_processors, true_id, false_id, device=device)
            # stable partition preserving order within the chunk
            for r, pred in zip(chunk, preds):
                (true_bucket if pred else false_bucket).append(r)

        bundle["results"] = true_bucket + false_bucket

    return ranked

# ────────────────────────── I/O + CLI ──────────────────────────
def load_json(fp: Path):
    with fp.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, fp: Path):
    fp.parent.mkdir(parents=True, exist_ok=True)
    with fp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)

def truncate_top_k(all_results: Dict, top_k: int) -> Dict:
    out = {}
    for qid, bundle in all_results.items():
        out[qid] = {
            "question": bundle["question"],
            "cluster-id": bundle.get("cluster-id"),
            "results": bundle["results"][:top_k],
        }
    return out

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Re-rank initial retrievals with an UnsLoTH-finetuned model.")
    p.add_argument("--input-dir", type=Path, default=Path("/mnt/gpu-fastdata/anxo/Scaling-Mental-Support/results/initial-retrieval"),
                   help="Directory containing initial retrieval JSONs.")
    p.add_argument("--glob", default="*.json", help="Glob for selecting input JSONs within --input-dir.")
    p.add_argument("--output-dir", type=Path, default=Path("/mnt/gpu-fastdata/anxo/Scaling-Mental-Support/results/logits/finetunecounselchat-zeroshot-unsloth/"),
                   help="Directory to write re-ranked JSONs.")
    p.add_argument("--finetuned-model", default="/mnt/gpu-fastdata/hf-cache/hub/llama3-8b-ScalingMentalHealth-Support-counselchat-relevance-lora-custom",
                   help="Path or HF id for the finetuned (merged) model.")
    p.add_argument("--cache-dir", default="/mnt/gpu-fastdata/hf-cache/hub", help="HF cache directory.")
    p.add_argument("--prompt-mode", choices=["zeroshot", "fewshot"], default="zeroshot",
                   help="Prompting strategy.")
    p.add_argument("--batch-size", type=int, default=1, help="Batch size for model inference.")
    p.add_argument("--max-seq-len", type=int, default=2048, help="Max sequence length for the tokenizer/model.")
    p.add_argument("--top-k", type=int, default=150, help="Truncate to top-K per question before re-ranking.")
    return p.parse_args()

def main():
    args = parse_args()

    inputs = sorted((args.input_dir).glob(args.glob))
    if not inputs:
        raise SystemExit(f"No JSONs found in {args.input_dir} matching '{args.glob}'")

    print(f"Found {len(inputs)} cached retrieval files.")
    print(f"Finetuned model: {args.finetuned_model}")
    print(f"Prompt mode    : {args.prompt_mode}")
    print(f"Batch size     : {args.batch_size}")
    print(f"Top-K          : {args.top_k}")

    for fp in inputs:
        print(f"\n🔹 Processing: {fp}")
        all_results_full = load_json(fp)
        all_results_trunc = truncate_top_k(all_results_full, args.top_k)

        reranked = filter_with_unsloth(
            all_results=all_results_trunc,
            prompt_mode=args.prompt_mode,
            finetuned_model=args.finetuned_model,
            cache_dir=args.cache_dir,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            show_progress=True,
        )

        # Name like: <stem>__{prompt}UnslothFilter_pool{K}_reranked.json
        suffix = f"__{args.prompt_mode}UnslothFilter_pool{args.top_k}_reranked.json"
        out_fp = args.output_dir / (fp.stem + suffix)
        save_json(reranked, out_fp)
        print(f"✅ Saved → {out_fp}")

if __name__ == "__main__":
    main()
