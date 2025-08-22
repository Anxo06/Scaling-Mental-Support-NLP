# rerank_with_bert_confidence.py
# ─────────────────────────────────
# • For each <question, response> pair a fine-tuned BERT classifier returns:
#       logits → softmax → P(true)
# • We keep ALL responses but re-order them:
#       – TRUEs first,   sorted by  P(true) ↓
#       – FALSEs last,   sorted by  P(false) ↓  (equivalently P(true) ↑)
#
# Expected model: BertForSequenceClassification with 2 labels
# Label mapping: uses config.id2label if available, else {0:false, 1:true}.

from copy import deepcopy
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification


def _resolve_true_index(model) -> int:
    """
    Return the label index that corresponds to 'true' (case-insensitive).
    Fallback to class 1 if no mapping is found.
    """
    id2label = getattr(model.config, "id2label", None)
    if isinstance(id2label, dict) and len(id2label) >= 2:
        # normalize mapping: ensure keys are int, values lowercased
        norm = {int(k): str(v).lower() for k, v in id2label.items()}
        # try to find 'true'
        for k, v in norm.items():
            if v.strip() == "true":
                return k
        # common alt: "LABEL_1" vs "LABEL_0": prefer 1 as true
        return 1 if 1 in norm else max(norm.keys())
    # default binary mapping
    return 1


def load_bert(model_dir: str, device: str | None = None, cache_dir: str | None = None):
    """
    Load a fine-tuned BertForSequenceClassification (binary).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok    = BertTokenizer.from_pretrained(model_dir, cache_dir=cache_dir)
    model  = BertForSequenceClassification.from_pretrained(model_dir, cache_dir=cache_dir).to(device).eval()
    true_idx = _resolve_true_index(model)
    return tok, model, device, true_idx


@torch.no_grad()
def batch_predict_proba(
    tok: BertTokenizer,
    model: BertForSequenceClassification,
    device: str,
    pairs: List[Tuple[str, str]],
    max_length: int = 512,
    true_idx: int = 1,
) -> List[Tuple[bool, float]]:
    """
    For each (question, response) return (is_true, p_true).
    """
    if not pairs:
        return []

    qs, rs = zip(*pairs)
    enc = tok(
        list(qs),
        list(rs),
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    ).to(device)

    logits = model(**enc).logits           # [B, C]
    probs  = F.softmax(logits, dim=-1)     # [B, C]
    p_true = probs[:, true_idx]            # [B]
    pred   = torch.argmax(logits, dim=-1)  # [B]

    out = []
    for pt, pr in zip(p_true.tolist(), pred.tolist()):
        out.append((pr == true_idx, float(pt)))
    return out


def confidence_rerank_bert(
    all_results: Dict,
    model_dir: str,
    batch_size: int = 32,
    cache_dir: str | None = None,
    max_length: int = 512,
) -> Dict:
    """
    Re-order results and attach `bert_score` (P(true)) & `bert_output` ("true"/"false").

    • TRUEs first,  sorted by  P(true) DESC
    • FALSEs next,  sorted by  P(false) DESC  (i.e., P(true) ASC)
    """
    tok, model, device, true_idx = load_bert(model_dir, cache_dir=cache_dir)
    ranked = deepcopy(all_results)

    for qid, bundle in ranked.items():
        question  = bundle["question"]
        responses = bundle["results"]

        # build batch pairs
        pairs = [(question, r["response"]) for r in responses]

        # predict in mini-batches to avoid OOM
        preds: List[Tuple[bool, float]] = []
        for start in range(0, len(pairs), batch_size):
            preds.extend(batch_predict_proba(
                tok, model, device,
                pairs[start : start + batch_size],
                max_length=max_length,
                true_idx=true_idx,
            ))

        # attach helper fields
        for resp, (is_true, p_true) in zip(responses, preds):
            resp["_bert_is_true"] = is_true
            resp["_bert_p_true"]  = p_true
            resp["bert_output"]   = "true" if is_true else "false"
            resp["bert_score"]    = p_true  # probability of TRUE

        # split buckets
        trues  = [r for r in responses if r["_bert_is_true"]]
        falses = [r for r in responses if not r["_bert_is_true"]]

        # sort:
        #  - TRUEs:  P(true) high → low
        #  - FALSEs: P(false) high → low  == P(true) low → high
        trues.sort(key=lambda r: r["_bert_p_true"], reverse=True)
        falses.sort(key=lambda r: r["_bert_p_true"])  # ascending p_true

        # merge and strip temp keys
        merged = []
        for r in (trues + falses):
            clean = {k: v for k, v in r.items() if not k.startswith("_bert_")}
            merged.append(clean)

        bundle["results"] = merged

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
                {"response": "Break tasks into smaller steps, take mindful breaks.", "score": 10.9},
                {"response": "The pet shop stocks everything you need to keep your anaconda happy.", "score": 2.5},
            ],
        }
    }

    MODEL_PATH = "/mnt/gpu-fastdata/anxo/Scaling-Mental-Support/python-project/experiments/bertranking/models/bert-relevance-all/checkpoint-11727"

    new_rank = confidence_rerank_bert(
        mock,
        MODEL_PATH,
        batch_size=16,
        cache_dir="/mnt/gpu-fastdata/hf-cache/hub",  # optional
        max_length=512,
    )

    from pprint import pprint
    pprint(new_rank)
