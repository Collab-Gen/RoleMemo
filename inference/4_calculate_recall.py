"""
Step 4: Recall@10 Statistics

Aggregates the per-sample retrieval scores from step 3 and reports:
  - Recall@10 (Fact):    fraction of ground-truth factual fragments
                         matched above threshold in the top-10 retrieved entries
  - Recall@10 (Insight): fraction of ground-truth insights matched above threshold

These are the two primary retrieval metrics reported in Table 1 of the paper.

Usage:
    python 4_calculate_recall.py --scores data/retrieval_scores.json
"""

import argparse
import json
from pathlib import Path

FACT_THRESHOLD    = 0.6
INSIGHT_THRESHOLD = 0.65


def compute_recall(scores: list[dict]) -> dict:
    fact_hits, fact_total = 0, 0
    insight_hits, insight_total = 0, 0

    for sample in scores:
        for frag_result in sample.get("fragment_results", []):
            fact_total += 1
            if frag_result["similarity"] >= FACT_THRESHOLD:
                fact_hits += 1

        insight_result = sample.get("insight_result", {})
        if insight_result:
            insight_total += 1
            if insight_result["similarity"] >= INSIGHT_THRESHOLD:
                insight_hits += 1

    fact_recall    = fact_hits    / fact_total    if fact_total    else 0.0
    insight_recall = insight_hits / insight_total if insight_total else 0.0

    return {
        "recall@10_fact":    {"score": round(fact_recall, 4),    "hits": fact_hits,    "total": fact_total},
        "recall@10_insight": {"score": round(insight_recall, 4), "hits": insight_hits, "total": insight_total},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True, help="Path to retrieval scores JSON (output of step 3)")
    args = parser.parse_args()

    scores = json.loads(Path(args.scores).read_text(encoding="utf-8"))
    results = compute_recall(scores)

    print(f"\n{'='*40}")
    print(f"  Recall@10 (Fact):    {results['recall@10_fact']['score']:.4f}"
          f"  ({results['recall@10_fact']['hits']}/{results['recall@10_fact']['total']})")
    print(f"  Recall@10 (Insight): {results['recall@10_insight']['score']:.4f}"
          f"  ({results['recall@10_insight']['hits']}/{results['recall@10_insight']['total']})")
    print(f"{'='*40}\n")


if __name__ == "__main__":
    main()
