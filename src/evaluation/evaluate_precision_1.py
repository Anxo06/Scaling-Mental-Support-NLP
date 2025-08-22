#!/usr/bin/env python3
"""
evaluate_precision_1.py
───────────────────────
Compute mean P@1 for retrieval results, EXCLUDING queries missing in results.

JSON STRUCTURES
---------------
Ground truth (questions with their relevant responses):
{
  "questions": [
    {
      "identifier": "Q12345",              # str
      "relevant-responses": [              # list[str]
        "Short answer text 1...",
        "Short answer text 2..."
      ]
    },
    ...
  ]
}

Results (per question, an ordered list of retrieved responses):
{
  "Q12345": {
    "question": "How can I ...?",          # str (not used for scoring)
    "cluster-id": "C7",                    # str or null (ignored)
    "results": [                           # ranked list (top-1 first)
      {
        "response": "Short answer text 2...",   # str (compared to ground truth)
        "question-identifier": "Q987",          # str (ignored)
        "cluster-id": "C3",                     # str or "null" (ignored)
        "score": 12.34                          # float (ignored)
      },
      ...
    ]
  },
  ...
}

Scoring
-------
P@1 per query = 1.0 if top-1 response ∈ relevant-responses else 0.0.
Queries absent from `results` are EXCLUDED from the average.
Queries with zero relevant responses ⇒ P@1 = 0.0 (standard IR convention).
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


# ───────────────────────────── IO ─────────────────────────────
def load_json(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ────────────────────────── METRIC ───────────────────────────
def precision_at_1(relevant_responses: set, retrieved_responses: List[str]) -> float:
    """P@1 for a single query."""
    if not relevant_responses:
        return 0.0
    if not retrieved_responses:
        return 0.0
    return 1.0 if retrieved_responses[0] in relevant_responses else 0.0


def mean_precision_at_1_excluding_missing(ground_truth: dict, results: dict) -> Tuple[float, Dict[int, List[float]], int, int]:
    """
    Mean P@1 over queries present in `results`.
    Returns: (mean_p1, grouped_by_num_relevant, num_evaluated, num_missing)
    """
    p1_scores: List[float] = []
    grouped: Dict[int, List[float]] = {}
    missing = 0

    for q in ground_truth.get("questions", []):
        qid = q["identifier"]
        rel_set = set(q.get("relevant-responses", []))

        if qid not in results:
            missing += 1
            continue

        retrieved = [r["response"] for r in results[qid].get("results", [])]
        p1 = precision_at_1(rel_set, retrieved)
        p1_scores.append(p1)

        n_rel = len(rel_set)
        grouped.setdefault(n_rel, []).append(p1)

    mean_p1 = float(np.mean(p1_scores)) if p1_scores else 0.0
    return mean_p1, grouped, len(p1_scores), missing


# ────────────────────────── CLI + MAIN ───────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate mean P@1 (excluding missing queries).")
    p.add_argument("--ground-truth", required=True, help="Path to ground truth JSON.")
    p.add_argument("--results-file", required=True, help="Path to results JSON.")
    return p.parse_args()


def main():
    args = parse_args()
    gt = load_json(args.ground_truth)
    rs = load_json(args.results_file)

    mean_p1, grouped, num_eval, num_missing = mean_precision_at_1_excluding_missing(gt, rs)

    print(f"✅ Mean P@1 (excluding missing queries): {mean_p1:.4f}\n")

    print("📊 P@1 grouped by number of relevant responses per question:")
    for n_rel in sorted(grouped.keys()):
        scores = grouped[n_rel]
        avg = float(np.mean(scores)) if scores else 0.0
        print(f"  - {n_rel:>2} relevant → P@1 = {avg:.4f}  (n={len(scores)})")

    print(f"\n📈 Overall Mean P@1: {mean_p1:.4f}")
    print(f"📊 Evaluated questions (in results): {num_eval}")
    print(f"⚠️  Queries missing from results (excluded): {num_missing}")


if __name__ == "__main__":
    main()


"""
Example usage:
python evaluate_precision_1.py \
  --ground-truth ./data/<dataset>/test-cleaned-with-clusters-question-and-responses-list.json \
  --results-file ./results/<model>/reranked__<dataset>__<modelname>.json

"""