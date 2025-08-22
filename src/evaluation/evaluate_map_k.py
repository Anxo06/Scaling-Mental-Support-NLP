#!/usr/bin/env python3
"""
evaluate_map_k.py
─────────────────
Compute MAP@K for retrieval outputs against ground-truth relevance.
"""

import argparse
import json
from typing import Dict, List, Tuple
import numpy as np
from pathlib import Path


# ───────────────────────────── IO ─────────────────────────────
def load_json(file_path: str) -> dict:
    """Load a JSON file and return its content."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ────────────────────────── METRICS ──────────────────────────
def average_precision_at_k(relevant_responses: set, retrieved_responses: List[str], k: int = 50) -> float:
    """
    AP@k for a single query.
    - relevant_responses: set[str]
    - retrieved_responses: ranked list[str] (top-1 first)
    """
    if not relevant_responses:
        # No relevant items → contribute 0 to MAP (standard IR convention)
        return 0.0

    retrieved_at_k = retrieved_responses[:k]
    num_rel_found = 0
    precision_sum = 0.0

    for i, resp in enumerate(retrieved_at_k, start=1):
        if resp in relevant_responses:
            num_rel_found += 1
            precision_sum += num_rel_found / i

    if num_rel_found == 0:
        return 0.0
    # Normalize by min(k, number of relevant documents)
    return precision_sum / min(len(relevant_responses), k)


def mean_average_precision_at_k(ground_truth: dict, results: dict, k: int = 50) -> Tuple[float, Dict[int, List[float]], int]:
    """
    MAP@k over all queries present in both ground truth and results.
    Returns: (map_k, grouped_by_num_relevant, num_evaluated_queries)
    """
    ap_scores: List[float] = []
    grouped: Dict[int, List[float]] = {}

    for q in ground_truth.get("questions", []):
        qid = q["identifier"]
        rel_set = set(q.get("relevant-responses", []))
        if qid not in results:
            continue

        retrieved_responses = [r["response"] for r in results[qid].get("results", [])]
        ap = average_precision_at_k(rel_set, retrieved_responses, k)
        ap_scores.append(ap)

        n_rel = len(rel_set)
        grouped.setdefault(n_rel, []).append(ap)

    map_k = float(np.mean(ap_scores)) if ap_scores else 0.0
    return map_k, grouped, len(ap_scores)


# ────────────────────────── CLI + MAIN ───────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate MAP@K for retrieval results.")
    p.add_argument("--ground-truth", required=True, help="Path to ground truth JSON.")
    p.add_argument("--results-file", required=True, help="Path to results JSON.")
    p.add_argument("--k", type=int, default=50, help="Cutoff K for MAP@K (default: 50).")
    return p.parse_args()


def main():
    args = parse_args()

    ground_truth = load_json(args.ground_truth)
    results = load_json(args.results_file)

    map_k, grouped_map, n_q = mean_average_precision_at_k(ground_truth, results, k=args.k)

    print(f"✅ Mean Average Precision @ {args.k}: {map_k:.4f}\n")

    print("📊 MAP grouped by number of relevant responses per question:")
    for n_rel in sorted(grouped_map.keys()):
        grp_scores = grouped_map[n_rel]
        avg_ap = float(np.mean(grp_scores)) if grp_scores else 0.0
        print(f"  - {n_rel:>3} relevant → MAP@{args.k} = {avg_ap:.4f}  (n={len(grp_scores)})")

    print(f"\n📈 Overall MAP@{args.k}: {map_k:.4f}")
    print(f"📊 Questions evaluated: {n_q}")


if __name__ == "__main__":
    main()


"""
Example usage:
python evaluate_map_k.py \
  --ground-truth ./data/<dataset>/test-cleaned-with-clusters-question-and-responses-list.json \
  --results-file ./results/<model>/reranked__<dataset>__<modelname>.json \
  --k 100


"""