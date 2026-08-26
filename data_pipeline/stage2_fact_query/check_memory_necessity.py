"""
Stage 2b: Memory-Necessity Check (Quality Assurance Step 2)

Verifies that every query genuinely requires retrieving and interpreting
conversation history. A role-playing agent is prompted to answer each query
WITHOUT access to conversation history. If it can answer well from common
sense alone, the sample is discarded.

QA criterion: memory-necessity
  - The agent must fail (or give a clearly inferior answer) when history
    is withheld, confirming that retrieval and interpretation are required.
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import client, MODEL_ID

INPUT_DIR = Path("data/fact_query")
OUTPUT_DIR = Path("data/fact_query_checked")

# System prompt for the role-playing agent (no history provided)
AGENT_SYSTEM_PROMPT = """\
You are roleplaying as the following persona:
{persona_description}

Answer the user's question based only on your persona knowledge.
You do NOT have access to any conversation history.
"""

JUDGE_SYSTEM_PROMPT = """\
You are evaluating whether a role-playing agent's answer to a question \
genuinely requires access to conversation history.

Given:
- The persona description
- The test question
- The agent's answer (produced WITHOUT history)
- The ground-truth insight that history would reveal

Determine: does the agent's answer already capture the key insight, \
or is it clearly generic/inferior compared to what history would enable?

Respond in JSON:
{
  "requires_memory": true/false,
  "reason": "brief explanation"
}

Return requires_memory=true if history is necessary for a good answer.
"""


def agent_answer_without_history(persona_desc: str, question: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": AGENT_SYSTEM_PROMPT.format(persona_description=persona_desc)},
            {"role": "user", "content": question},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def judge_memory_necessity(persona_desc: str, question: str, agent_answer: str, insight: str) -> dict:
    user_msg = json.dumps(
        {
            "persona": persona_desc,
            "question": question,
            "agent_answer_without_history": agent_answer,
            "ground_truth_insight": insight,
        },
        ensure_ascii=False,
    )
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    return json.loads(response.choices[0].message.content)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total, passed = 0, 0
    for input_file in sorted(INPUT_DIR.glob("*.json")):
        output_file = OUTPUT_DIR / input_file.name
        if output_file.exists():
            print(f"[skip] {input_file.name}")
            continue

        data = json.loads(input_file.read_text(encoding="utf-8"))
        persona_desc = data["persona"].get("role_description", "")
        print(f"[check] {input_file.name} — {len(data['samples'])} samples")

        passed_samples = []
        for sample in data["samples"]:
            question = sample["test_question"]["text"]
            insight = sample["insight"]

            agent_answer = agent_answer_without_history(persona_desc, question)
            verdict = judge_memory_necessity(persona_desc, question, agent_answer, insight)

            total += 1
            if verdict.get("requires_memory", False):
                passed += 1
                passed_samples.append({**sample, "memory_necessity_check": verdict})

        output = {**data, "samples": passed_samples}
        output_file.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Done. Pass rate: {passed}/{total} ({passed/total:.1%})")


if __name__ == "__main__":
    main()
