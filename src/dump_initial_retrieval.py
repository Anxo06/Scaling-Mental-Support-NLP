#!/usr/bin/env python3
"""
dump_initial_retrieval.py
─────────────────────────
Query Elasticsearch once to build the initial `all_results` dictionary
(question → top-N retrieved responses) and save it to JSON for later use.

Downstream rerankers (LLM/T5, etc.) can load the JSON without re-querying ES.

Expected questions JSON format
------------------------------
{
  "questions": [
    {
      "identifier": "Q12345",         # (str) unique id of the question
      "question": "How can I ...?",   # (str) the query text to retrieve with
      "cluster-id": "C7"              # (str or null), optional; use "null" if absent
    },
    ...
  ]
}
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

# Your ES semantic search function (BM25 version works too if you prefer)
from retrieval.search_cosine_embeddings import search_semantic


# ───────────────────────── helpers ─────────────────────────
def load_json(fp: str):
    with open(fp, encoding="utf-8") as f:
        return json.load(f)

def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def run_all_queries(questions_file: str, index_name: str, top_n: int = 200):
    """
    Build the `all_results` dict:
        {
          qid: {
            "question": "...",
            "cluster-id": ...,
            "results": [
               {"response": "...", "question-identifier": "...", "cluster-id": "...", "score": ...},
               ...
            ]
          },
          ...
        }
    """
    test_questions = load_json(questions_file)["questions"]
    all_results = {}

    for q in test_questions:
        qid        = q["identifier"]
        query_text = q["question"]
        cluster_id = q.get("cluster-id", "null")

        results = search_semantic(
            query_text,
            question_id=qid,
            cluster_id=cluster_id,
            index_name=index_name,
            top_n=top_n,
        )

        all_results[qid] = {
            "question":   query_text,
            "cluster-id": cluster_id,
            "results":    results,
        }

    return all_results


# ───────────────────────── CLI ────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Dump initial retrieval results (question → top-N responses) to JSON."
    )
    # Mode A: single run
    single = p.add_argument_group("single run")
    single.add_argument("--questions-file", help="Path to questions JSON (single run).")
    single.add_argument("--index-name", help="Elasticsearch index name (single run).")

    # Mode B: batch run over dataset:index pairs
    batch = p.add_argument_group("batch run")
    batch.add_argument(
        "--pairs",
        nargs="+",
        help=("Space-separated list of dataset:index pairs, e.g. "
              "'counselchat:responses_cc_index 7cups:responses_7cups_index'. "
              "If omitted, defaults to the three common pairs."),
    )
    batch.add_argument(
        "--base-dir",
        default="/mnt/datasets/depression/counseling",
        help="Base dir containing dataset subfolders (used in batch mode).",
    )

    # Common
    p.add_argument("--top-n", type=int, default=200, help="Top-N to retrieve.")
    p.add_argument(
        "--result-dir",
        type=Path,
        default=Path("/mnt/gpu-fastdata/anxo/Scaling-Mental-Support/results/initial-retrieval"),
        help="Directory to write JSON outputs.",
    )
    p.add_argument(
        "--timestamp",
        default=datetime.now().strftime("%Y%m%d-%H%M"),
        help="Timestamp tag for output filenames (default: current time).",
    )
    return p.parse_args()


# ───────────────────────── main ───────────────────────────
def main():
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)

    # Decide mode: single vs batch
    single_mode = args.questions_file and args.index_name
    batch_mode = args.pairs is not None

    if single_mode and batch_mode:
        raise SystemExit("Specify EITHER single mode (--questions-file + --index-name) OR batch mode (--pairs).")

    if not single_mode and not batch_mode:
        # default to your three common pairs
        args.pairs = [
            "counselchat:responses_cc_index",
            "7cups:responses_7cups_index",
            "7Cups-CC-mixed:responses_all_index",
        ]
        batch_mode = True

    if single_mode:
        dataset_tag = Path(args.questions_file).stem
        out_file = args.result_dir / f"initial-retrieval__{dataset_tag}__{args.index_name}__top{args.top_n}__{args.timestamp}.json"

        print(f"\n🔹 SINGLE RUN")
        print(f"🔹 QUESTIONS: {args.questions_file}")
        print(f"🔹 INDEX    : {args.index_name}")
        print(f"🔹 TOP_N    : {args.top_n}")
        print(f"🔹 OUTPUT   : {out_file}")

        retrieved = run_all_queries(args.questions_file, args.index_name, top_n=args.top_n)
        ensure_dir(out_file)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(retrieved, f, indent=4, ensure_ascii=False)
        print(f"✅ Saved → {out_file}\n")
        return

    # Batch mode
    print("\n🔹 BATCH RUN")
    for pair in args.pairs:
        try:
            dataset, es_index = pair.split(":", 1)
        except ValueError:
            raise SystemExit(f"Invalid pair '{pair}'. Expected format 'dataset:index'.")

        questions_file = (
            Path(args.base_dir) / dataset / "dataset" /
            "questions_with_answers_only" / "classification_format" / "splits" /
            "test-cleaned-with-clusters-questions-list.json"
        )

        out_file = args.result_dir / f"initial-retrieval__{dataset}__{es_index}__top{args.top_n}__{args.timestamp}.json"

        print(f"\n🔹 DATASET: {dataset} | INDEX: {es_index}")
        print(f"🔹 TOP_N  : {args.top_n}")
        print(f"🔹 QFILE  : {questions_file}")
        print(f"🔹 OUTPUT : {out_file}")

        retrieved = run_all_queries(str(questions_file), es_index, top_n=args.top_n)

        ensure_dir(out_file)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(retrieved, f, indent=4, ensure_ascii=False)
        print(f"✅ Saved → {out_file}")

    print()

if __name__ == "__main__":
    main()


"""Example usage:
Single run:
python dump_initial_retrieval.py \
  --questions-file /path/to/test-cleaned-with-clusters-questions-list.json \
  --index-name responses_all_index \
  --top-n 200 \
  --result-dir ./results/initial-retrieval

Batch run (multiple dataset:index pairs):
python dump_initial_retrieval.py \
  --pairs counselchat:responses_cc_index 7cups:responses_7cups_index \
  --base-dir /mnt/datasets/depression/counseling \
  --top-n 200 \
  --result-dir ./results/initial-retrieval

Batch run (defaults to your 3 datasets if no args given)
python dump_initial_retrieval.py
  """