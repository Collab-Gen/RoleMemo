"""
Stage 2a: Fact-Insight Pair and Query Generation

For each persona-conditioned insight that passed Stage 1 quality check,
generate:
  1. Two factual fragments — neutral, mundane user statements that are
     individually ambiguous but jointly support the insight when interpreted
     through the persona lens.
  2. A test question that cannot be answered well from facts alone and
     rewards persona-conditioned reasoning.
  3. A logic connector explaining the fact → insight → question chain.
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import client, MODEL_ID

PROMPT_PATH = Path(__file__).parent / "prompts" / "fact_question_generate.txt"
INPUT_DIR = Path("data/insights_checked")
OUTPUT_DIR = Path("data/fact_query")


def load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def generate_fact_query(persona_data: dict, insight: str, system_prompt: str) -> dict:
    user_msg = json.dumps(
        {"persona": persona_data, "insight": insight}, ensure_ascii=False
    )
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
        temperature=0.8,
    )
    return json.loads(response.choices[0].message.content)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    system_prompt = load_prompt()

    for checked_file in sorted(INPUT_DIR.glob("*.json")):
        output_file = OUTPUT_DIR / checked_file.name
        if output_file.exists():
            print(f"[skip] {checked_file.name}")
            continue

        data = json.loads(checked_file.read_text(encoding="utf-8"))
        insights = [r["original_insight"] for r in data.get("checked_insights", [])]
        print(f"[generate] {checked_file.name} — {len(insights)} insights")

        samples = []
        for insight in insights:
            result = generate_fact_query(data, insight, system_prompt)
            samples.append(result)

        output_file.write_text(
            json.dumps({"persona": data, "samples": samples}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print("Done.")


if __name__ == "__main__":
    main()
