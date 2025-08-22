# rerank_with_bert_simple.py
# ────────────────────────────
from copy import deepcopy
from typing import Dict, List, Tuple

import torch
from transformers import BertTokenizer, BertForSequenceClassification


def load_bert(model_dir: str, device: str | None = None, cache_dir: str | None = None):
    """
    Load a fine-tuned BertForSequenceClassification (2 classes: false/true).
    Assumes label mapping is either in config.id2label or default {0:false, 1:true}.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok    = BertTokenizer.from_pretrained(model_dir, cache_dir=cache_dir, token="hf_token")
    model  = BertForSequenceClassification.from_pretrained(model_dir, cache_dir=cache_dir, token="hf_token").to(device).eval()

    # read mapping if present (fallback to 0:false, 1:true)
    id2label = getattr(model.config, "id2label", None)
    if isinstance(id2label, dict) and len(id2label) == 2:
        id2label = {int(k): str(v).lower() for k, v in id2label.items()}
    else:
        id2label = {0: "false", 1: "true"}

    return tok, model, device, id2label


@torch.no_grad()
def batch_is_true(
    tok: BertTokenizer,
    model: BertForSequenceClassification,
    device: str,
    pairs: List[Tuple[str, str]],
    max_length: int = 512,
) -> List[bool]:
    """
    Return True/False for each (question, response) pair depending on the predicted class.
    """
    questions, responses = zip(*pairs) if pairs else ([], [])
    enc = tok(
        list(questions),
        list(responses),
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    ).to(device)

    logits = model(**enc).logits              # [B, 2]
    preds  = torch.argmax(logits, dim=-1)     # [B]
    return [bool(p.item()) for p in preds]    # 0 -> False, 1 -> True


def filter_with_bert(
    all_results: Dict,
    model_dir: str,
    batch_size: int = 32,
    cache_dir: str | None = None,
    max_length: int = 512,
) -> Dict:
    """
    Re-order each question's `results` so that responses classified "true"
    (relevant) stay at the top, preserving original order; "false" go to the end.
    """
    tok, model, device, _ = load_bert(model_dir, cache_dir=cache_dir)
    ranked = deepcopy(all_results)            # don't touch original

    for qid, bundle in ranked.items():
        question  = bundle["question"]
        responses = bundle["results"]

        true_bucket, false_bucket = [], []

        # walk through responses in batches
        for start in range(0, len(responses), batch_size):
            batch = responses[start : start + batch_size]
            pairs = [(question, resp["response"]) for resp in batch]

            preds = batch_is_true(tok, model, device, pairs, max_length=max_length)

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
                {"response": "Break tasks into smaller steps, take mindful breaks.", "score": 10.9},
                {"response": "The pet shop stocks everything you need to keep your anaconda happy.", "score": 2.5},
            ],
        }
    }

    # <- put your BERT checkpoint here (folder with config.json + model weights)
    MODEL_PATH = "/mnt/gpu-fastdata/anxo/Scaling-Mental-Support/python-project/experiments/bertranking/models/bert-relevance-all/checkpoint-11727"  # <- change me


    cleaned = filter_with_bert(
        mock,
        MODEL_PATH,
        batch_size=16,
        cache_dir="/mnt/gpu-fastdata/hf-cache/hub",   # optional
        max_length=512,
    )

    from pprint import pprint
    pprint(cleaned)
