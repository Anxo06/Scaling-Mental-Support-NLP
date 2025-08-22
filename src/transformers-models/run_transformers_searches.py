#!/usr/bin/env python3
"""
run_transformers_searches.py
───────────────────────────
Second-stage filtering / reranking with BERT/T5 models over
precomputed initial-retrieval JSONs (from dump_initial_retrieval.py).

Strategies
----------
--strategy {bert-filter, bert-confidence, t5-filter, t5-confidence}

Input JSON shape (produced by dump_initial_retrieval.py)
--------------------------------------------------------
{
  "Q123": {
    "question": "How can I ...?",
    "cluster-id": "C7",                # optional
    "results": [
      {
        "response": "...",             # ranked list, top-1 first
        "question-identifier": "Q...", # optional
        "cluster-id": "C...",          # optional
        "score": 12.3                  # optional
      },
      ...
    ]
  },
  ...
}
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Callable

# T5 variants:
from monot5.filter_with_monot5 import filter_with_t5
from monot5.rerank_with_monot5 import confidence_rerank_t5

# BERT variants:
from monobert.filter_with_bert import filter_with_bert
from monobert.filter_with_bert import confidence_rerank_bert


# ───────────────────────── helpers ─────────────────────────
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


# ───────────────────────── CLI ────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Re-rank initial retrieval JSONs with BERT/T5 models."
    )

    # Single file mode
    single = p.add_argument_group("single file")
    single.add_argument("--input-file", type=Path, help="Path to one initial-retrieval JSON.")
    single.add_argument("--model-path", help="Model checkpoint/ID for single-file mode.")

    # Batch directory mode
    batch = p.add_argument_group("batch directory")
    batch.add_argument("--input-dir", type=Path, help="Directory containing initial-retrieval JSONs.")
    batch.add_argument("--glob", default="*.json", help="Glob to select input JSONs inside --input-dir.")
    batch.add_argument("--model-paths", nargs="+", help="List of model checkpoints/IDs for batch mode.")

    # Common
    p.add_argument("--strategy",
                   choices=["bert-filter", "bert-reorder", "t5-filter", "t5-reorder"],
                   default="bert-filter",
                   help="Which re-ranking method to use.")
    p.add_argument("--output-dir", type=Path, default=Path("./results/reranked"),
                   help="Directory to write re-ranked JSONs.")
    p.add_argument("--top-k", type=int, default=150, help="Truncate to top-K per question before re-ranking.")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size for the second-stage model.")
    p.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d-%H%M"),
                   help="Timestamp tag for output filenames.")
    return p.parse_args()


def pick_filter_fn(strategy: str) -> Callable:
    if strategy == "bert-filter":
        return filter_with_bert
    if strategy == "bert-reorder":
        return confidence_rerank_bert
    if strategy == "t5-filter":
        return filter_with_t5
    if strategy == "t5-reorder":
        return confidence_rerank_t5
    raise ValueError(f"Unknown strategy: {strategy}")


# ───────────────────────── main ───────────────────────────
def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    single_mode = args.input_file is not None and args.model_path is not None
    batch_mode = args.input_dir is not None or args.model_paths is not None

    if single_mode and batch_mode:
        raise SystemExit("Specify EITHER single file mode (--input-file --model-path) OR batch mode (--input-dir/--glob --model-paths).")

    if not single_mode and not batch_mode:
        # Defaults for batch mode if nothing specified
        args.input_dir = Path("./results/initial-retrieval")
        args.model_paths = [
            "/mnt/gpu-fastdata/anxo/Scaling-Mental-Support/python-project/experiments/bertranking/models/mentalbert/mentalbert-relevance-all/checkpoint-11727",
        ]
        batch_mode = True

    filter_fn = pick_filter_fn(args.strategy)

    if single_mode:
        in_fp = args.input_file
        model = args.model_path
        tag = Path(model).parent.name if "/" in model else model
        out_fp = args.output_dir / f"{in_fp.stem}__{args.strategy}__Modelname__{tag}__pool{args.top_k}_{args.timestamp}.json"

        print(f"\n🔹 SINGLE FILE")
        print(f"🔹 INPUT    : {in_fp}")
        print(f"🔹 MODEL    : {model}")
        print(f"🔹 STRATEGY : {args.strategy}")
        print(f"🔹 TOP-K    : {args.top_k} | BATCH: {args.batch_size}")
        print(f"🔹 OUTPUT   : {out_fp}")

        all_results_full = load_json(in_fp)
        all_results_trunc = truncate_top_k(all_results_full, args.top_k)

        reranked = filter_fn(all_results_trunc, model, batch_size=args.batch_size)
        save_json(reranked, out_fp)
        print(f"✅ Saved → {out_fp}\n")
        return

    # Batch directory mode
    inputs = sorted((args.input_dir).glob(args.glob))
    if not inputs:
        raise SystemExit(f"No JSONs found in {args.input_dir} matching '{args.glob}'")
    if not args.model_paths:
        raise SystemExit("Batch mode requires --model-paths.")

    print(f"\n🔹 BATCH DIR")
    print(f"🔹 INPUT DIR : {args.input_dir} (glob='{args.glob}', {len(inputs)} files)")
    print(f"🔹 MODELS    : {len(args.model_paths)}")
    print(f"🔹 STRATEGY  : {args.strategy}")
    print(f"🔹 TOP-K     : {args.top_k} | BATCH: {args.batch_size}")
    print(f"🔹 OUTPUT DIR: {args.output_dir}")

    for in_fp in inputs:
        for model in args.model_paths:
            tag = Path(model).parent.name if "/" in model else model
            out_fp = args.output_dir / f"{in_fp.stem}__{args.strategy}__Modelname__{tag}__pool{args.top_k}_{args.timestamp}.json"

            print(f"\n🔹 FILE: {in_fp.name} | MODEL: {tag}")
            all_results_full = load_json(in_fp)
            all_results_trunc = truncate_top_k(all_results_full, args.top_k)

            reranked = filter_fn(all_results_trunc, model, batch_size=args.batch_size)
            save_json(reranked, out_fp)
            print(f"✅ Saved → {out_fp}")

    print()

if __name__ == "__main__":
    main()
