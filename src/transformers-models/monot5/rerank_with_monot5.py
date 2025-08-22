"""
rerank_with_monot5_confidence.py
────────────────────────────────
• For each <question, response> pair MonoT5 generates "true"/"false".
• We keep ALL responses but re-order them:
      – TRUEs first,   sorted by     probability ↓
      – FALSEs last,   sorted by     probability ↑
"""

from copy import deepcopy
from typing import Dict, List, Tuple
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration


# ───────────────────────── loader ───────────────────────────────
def load_monot5(model_dir: str, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok    = T5Tokenizer.from_pretrained(model_dir, cache_dir="/mnt/gpu-fastdata/hf-cache/hub")
    model  = T5ForConditionalGeneration.from_pretrained(model_dir, cache_dir="/mnt/gpu-fastdata/hf-cache/hub").to(device).eval()
    return tok, model, device


@torch.no_grad()
def batch_predict(tok, model, device, prompts: List[str]) -> List[Tuple[bool, float]]:
    """
    Returns list of (is_true, prob) for each prompt.
      • prob  is exp(sequence_log_prob)  ∈ (0,1]
    """
    enc = tok(prompts, padding=True, truncation=True, return_tensors="pt").to(device)

    outs = model.generate(
        **enc,
        do_sample=False,
        num_beams=4,
        min_new_tokens=1,
        max_new_tokens=3,
        return_dict_in_generate=True,
        output_scores=True,
    )

    decoded = tok.batch_decode(outs.sequences, skip_special_tokens=True)
    probs   = torch.exp(outs.sequences_scores).tolist()  # log-prob → prob

    results = []
    for txt, p in zip(decoded, probs):
        is_true = txt.lower().startswith("true")
        results.append((is_true, p))
    return results


# ─────────────────────────── ranking ────────────────────────────
def confidence_rerank_t5(
    all_results: Dict,
    model_dir: str,
    batch_size: int = 32,
) -> Dict:
    """
    Re-order results and attach `t5_score` (probability that MonoT5 said "true").

    • TRUEs first, high → low by `t5_score`
    • FALSEs next, low → high by `t5_score`
    """
    tok, model, device = load_monot5(model_dir)
    ranked = deepcopy(all_results)

    for qid, bundle in ranked.items():
        question  = bundle["question"]
        responses = bundle["results"]

        # build prompts
        prompts = [
            f"Query: {question} Document: {r['response']}"
            for r in responses
        ]

        # run MonoT5 in batches and collect (is_true, prob)
        preds = []
        for start in range(0, len(prompts), batch_size):
            preds.extend(batch_predict(
                tok, model, device, prompts[start : start + batch_size])
            )

        # attach helper fields
        for resp, (is_true, prob) in zip(responses, preds):
            resp["_t5_is_true"] = is_true
            resp["_t5_prob"]    = prob
            resp["t5_output"]    = is_true


        # split buckets
        trues  = [r for r in responses if r["_t5_is_true"]]
        falses = [r for r in responses if not r["_t5_is_true"]]

        # sort inside each bucket
        trues.sort(key=lambda r: r["_t5_prob"], reverse=True)   # high → low
        falses.sort(key=lambda r: r["_t5_prob"])                # low → high

        # merge + keep `t5_score`
        bundle["results"] = []
        for r in (trues + falses):
            clean = {k: v for k, v in r.items() if not k.startswith("_t5_")}
            clean["t5_score"] = r["_t5_prob"]
            clean["t5_output"] = r["t5_output"]
            bundle["results"].append(clean)

    return ranked


# ───────────────────── demo run ─────────────────────
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

    new_rank = confidence_rerank_t5(mock, MODEL_PATH, batch_size=16)

    from pprint import pprint
    pprint(new_rank)
