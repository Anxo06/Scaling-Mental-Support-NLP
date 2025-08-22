# rerank_with_monot5_simple.py
# ────────────────────────────
from copy import deepcopy
from typing import Dict, List

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration


def load_monot5(model_dir: str, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok    = T5Tokenizer.from_pretrained(model_dir, cache_dir="/mnt/gpu-fastdata/hf-cache/hub")
    model  = T5ForConditionalGeneration.from_pretrained(model_dir, cache_dir="/mnt/gpu-fastdata/hf-cache/hub").to(device).eval()
    return tok, model, device


@torch.no_grad()
def batch_is_true(tok, model, device, prompts: List[str]) -> List[bool]:
    """
    Return True/False for each prompt depending on whether the first generated
    token is "true".
    """
    enc = tok(prompts, padding=True, truncation=True, return_tensors="pt").to(device)
    outs = model.generate(
        **enc,
        do_sample=False,
        num_beams=1,
        min_new_tokens=1,
        max_new_tokens=3,
    )
    decoded = tok.batch_decode(outs, skip_special_tokens=True)
    return [d.lower().startswith("true") for d in decoded]


def filter_with_t5(
    all_results: Dict,
    model_dir: str,
    batch_size: int = 32,
) -> Dict:
    """
    Re-order each question's `results` so that responses classified "true"
    (relevant) stay at the top, preserving original order; "false" go to the end.
    """
    tok, model, device = load_monot5(model_dir)
    ranked = deepcopy(all_results)            # don't touch original

    for qid, bundle in ranked.items():
        question  = bundle["question"]
        responses = bundle["results"]

        true_bucket, false_bucket = [], []

        # walk through responses in batches
        for start in range(0, len(responses), batch_size):
            batch = responses[start : start + batch_size]
            prompts = [
                f"Query: {question} Document: {resp['response']}" for resp in batch
            ]

            preds = batch_is_true(tok, model, device, prompts)

            # distribute into the two buckets, keeping relative order
            for resp, is_true in zip(batch, preds):
                (true_bucket if is_true else false_bucket).append(resp)

        bundle["results"] = true_bucket + false_bucket

    return ranked


# ─────────── quick demo ───────────
if __name__ == "__main__":
    mock = {
        "q1": {
            "question": "How can I manage my anxiety at work?",
            "cluster-id": "1149",
            "results": [
                {"response": "Try breathing exercises.", "score": 12.3},
                {"response": "Cats are cute and fluffy.", "score": 11.0},
                {"response": "To manage anxiety at work, prioritize workload management by breaking down tasks, setting realistic deadlines, and communicating with your manager about workload concerns. Practice healthy habits like getting enough sleep, eating well, and exercising.", "score": 10.9},
                {"response": "She always had an interesting perspective on why the world must be flat.", "score": 2.9},
                {"response": "The pet shop stocks everything you need to keep your anaconda happy.", "score": 2.5},
                {"response": "To manage anxiety at work, prioritize workload management by breaking down tasks, setting realistic deadlines, and communicating with your manager about workload concerns. Practice healthy habits like getting enough sleep, eating well, and exercising.", "score": 1.9},
                {"response": "The gloves protect my feet from excess work.", "score": 1.9},

            ],
        }
    }

    MODEL_PATH = "/mnt/gpu-fastdata/anxo/Scaling-Mental-Support/python-project/experiments/t5ranking/monot5-relevance-counselchat/checkpoint-228"   # <- put your checkpoint here


    cleaned = filter_with_t5(mock, MODEL_PATH, batch_size=16)
    from pprint import pprint
    pprint(cleaned)