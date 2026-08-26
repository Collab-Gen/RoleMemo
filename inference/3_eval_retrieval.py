"""
Step 3: Retrieval Quality Evaluation — Recall@10

Measures whether the top-10 retrieved memory entries contain the
ground-truth facts and insights required to answer each query.

Recall@10 is computed separately for:
  - Fact memory:    ground-truth factual fragments (core_fragments)
  - Insight memory: ground-truth persona-conditioned insight (core_insight)

A retrieved entry is counted as a hit if its cosine similarity to the
ground-truth exceeds a threshold (Section 4.1 / Appendix of the paper).
Similarity is computed with Qwen3-Embedding-0.6B (the same embedding model
used for retrieval in step 2).

Usage:
    python 3_eval_retrieval.py \
        --responses  data/responses.json \
        --ground_truth data/ground_truth.json \
        --output     data/retrieval_scores.json
"""

import argparse
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# Similarity thresholds for hit detection
FACT_THRESHOLD    = 0.6
INSIGHT_THRESHOLD = 0.65


def embed_texts(texts: list[str], embedder) -> np.ndarray:
    return embedder.encode(texts, normalize_embeddings=True)


def best_match(query_text: str, candidates: list[str], embedder) -> tuple[str, float]:
    """Return the candidate most similar to query_text and its cosine similarity."""
    if not candidates:
        return "", 0.0
    q_emb = embed_texts([query_text], embedder)[0]
    c_embs = embed_texts(candidates, embedder)
    sims = c_embs @ q_emb
    idx = int(np.argmax(sims))
    return candidates[idx], float(sims[idx])


def parse_retrieved_entries(retrieved_context: str) -> list[str]:
    """
    Extract individual memory entry texts from the retrieved context string
    produced by step 2. Returns up to TOP_K entries.
    """
    entries = []
    for line in retrieved_context.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip prefixes added by step 2
        for prefix in ("[Fact] ", "[Insight] "):
            if line.startswith(prefix):
                line = line[len(prefix):]
                break
        # Remove grounding annotation "(grounded in Fact X): "
        if line.startswith("(grounded in Fact"):
            line = line.split("): ", 1)[-1]
        entries.append(line)
    return entries[:10]  # enforce top-10


def evaluate_sample(response: dict, gt: dict, embedder) -> dict:
    """
    For one sample, compute similarity of each ground-truth entry against
    the top-10 retrieved context.
    """
    retrieved = parse_retrieved_entries(response.get("retrieved_context", ""))

    core_fragments = gt.get("core_fragments", [])
    core_insight   = gt.get("core_insight", "")

    fragment_results = []
    for frag in core_fragments:
        matched, sim = best_match(frag, retrieved, embedder)
        fragment_results.append({"gt": frag, "best_match": matched, "similarity": sim})

    insight_result = {}
    if core_insight:
        matched, sim = best_match(core_insight, retrieved, embedder)
        insight_result = {"gt": core_insight, "best_match": matched, "similarity": sim}

    return {
        "id": response["id"],
        "fragment_results": fragment_results,
        "insight_result": insight_result,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses",    required=True, help="Path to responses JSON (output of step 2)")
    parser.add_argument("--ground_truth", required=True, help="Path to ground truth JSON")
    parser.add_argument("--output",       required=True, help="Path to save retrieval scores JSON")
    args = parser.parse_args()

    responses = json.loads(Path(args.responses).read_text(encoding="utf-8"))
    ground_truth = json.loads(Path(args.ground_truth).read_text(encoding="utf-8"))
    gt_map = {item["id"]: item for item in ground_truth}

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    scores = []
    for response in responses:
        rid = response["id"]
        if rid not in gt_map:
            continue
        result = evaluate_sample(response, gt_map[rid], embedder)
        scores.append(result)

    Path(args.output).write_text(
        json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Done. Scores for {len(scores)} samples saved to {args.output}")
    print("Run 4_calculate_recall.py to compute Recall@10 statistics.")


if __name__ == "__main__":
    main()
