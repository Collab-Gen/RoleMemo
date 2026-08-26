"""
Step 2: Role-Playing Response Generation (f_phi)

Given a query and the retrieved memory context (R(M, q)), the role-playing
agent generates a response in character. This corresponds to:

    y_hat = f_phi(p, q, R(M, q))    [Equation (1), paper Section 3.1]

The retrieval step (R) selects the top-K most relevant memory entries using
Qwen3-Embedding-0.6B. When an insight is retrieved, its referenced facts
are also included to provide grounding evidence (Section 3.1).

Note: The paper uses a dedicated role-playing agent (Doubao) as f_phi.
This script provides a general OpenAI-compatible interface so users can
substitute any instruction-following model.

Usage:
    python 2_generate_response.py \
        --memory   data/memory_bank.json \
        --persona  data/persona.json \
        --queries  data/queries.json \
        --output   data/responses.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).parent.parent))
from config import client, MODEL_ID

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
TOP_K = 10

AGENT_SYSTEM_TEMPLATE = """\
You are roleplaying as the following persona. Stay strictly in character.

{persona}

The following memory entries are relevant to the user's question.
Use them to inform your response, but integrate the information naturally.

{retrieved_context}
"""


def embed_texts(texts: list[str], embedder) -> np.ndarray:
    return embedder.encode(texts, normalize_embeddings=True)


def retrieve(query: str, memory_bank: dict, embedder, top_k: int = TOP_K) -> str:
    """
    Retrieve top-K relevant memory entries for the query using semantic search.
    When an insight is retrieved, its referenced facts are also included.
    Corresponds to R(M, q) in the paper.
    """
    facts = memory_bank.get("facts", [])
    insights = memory_bank.get("insights", [])
    all_entries = facts + insights

    if not all_entries:
        return "(no memory available)"

    texts = [e["text"] for e in all_entries]
    entry_embs = embed_texts(texts, embedder)
    query_emb = embed_texts([query], embedder)[0]

    scores = entry_embs @ query_emb
    top_indices = np.argsort(scores)[::-1][:top_k]
    top_entries = [all_entries[i] for i in top_indices]

    # For retrieved insights, also include their referenced facts
    fact_map = {f["id"]: f for f in facts}
    extra_facts = []
    for entry in top_entries:
        if "refs" in entry:  # this is an insight
            for ref_id in entry["refs"]:
                if ref_id in fact_map and fact_map[ref_id] not in top_entries:
                    extra_facts.append(fact_map[ref_id])

    final_entries = top_entries + extra_facts

    lines = []
    for e in final_entries:
        if "refs" in e:
            refs = ", ".join(str(r) for r in e["refs"])
            lines.append(f"[Insight] (grounded in Fact {refs}): {e['text']}")
        else:
            lines.append(f"[Fact] {e['text']}")
    return "\n".join(lines)


def generate_response(persona: str, query: str, retrieved_context: str) -> str:
    system = AGENT_SYSTEM_TEMPLATE.format(
        persona=persona,
        retrieved_context=retrieved_context,
    )
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": query},
        ],
        temperature=0.8,
    )
    return response.choices[0].message.content.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory",  required=True, help="Path to memory bank JSON (output of step 1)")
    parser.add_argument("--persona", required=True, help="Path to persona JSON")
    parser.add_argument("--queries", required=True, help="Path to queries JSON (list of {id, question})")
    parser.add_argument("--output",  required=True, help="Path to save responses JSON")
    args = parser.parse_args()

    memory_bank = json.loads(Path(args.memory).read_text(encoding="utf-8"))
    persona_data = json.loads(Path(args.persona).read_text(encoding="utf-8"))
    persona_text = persona_data.get("description", json.dumps(persona_data, ensure_ascii=False))
    queries = json.loads(Path(args.queries).read_text(encoding="utf-8"))

    output_path = Path(args.output)
    # Resume from existing output if interrupted
    if output_path.exists():
        results = json.loads(output_path.read_text(encoding="utf-8"))
        done_ids = {r["id"] for r in results}
    else:
        results = []
        done_ids = set()

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    for item in queries:
        qid = item["id"]
        if qid in done_ids:
            print(f"[skip] {qid}")
            continue

        query = item["question"]
        print(f"[generate] {qid}")
        retrieved_context = retrieve(query, memory_bank, embedder)
        answer = generate_response(persona_text, query, retrieved_context)

        results.append({
            "id": qid,
            "question": query,
            "retrieved_context": retrieved_context,
            "answer": answer,
        })
        output_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(f"Done. {len(results)} responses saved to {args.output}")


if __name__ == "__main__":
    main()
