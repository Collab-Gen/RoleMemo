"""
Stage 1a: Persona Insight Generation

For each persona, generate a set of cognitive insights that reflect the
persona's unique reasoning framework. These insights are later used to
construct memory evaluation queries.
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import client, MODEL_ID

PROMPT_PATH = Path(__file__).parent / "prompts" / "insight_generate.txt"
INPUT_DIR = Path("data/personas")       # each file: a persona JSON
OUTPUT_DIR = Path("data/insights")


def load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def generate_insights(persona: dict, system_prompt: str) -> dict:
    user_msg = json.dumps(persona, ensure_ascii=False)
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

    for persona_file in sorted(INPUT_DIR.glob("*.json")):
        output_file = OUTPUT_DIR / persona_file.name
        if output_file.exists():
            print(f"[skip] {persona_file.name}")
            continue

        persona = json.loads(persona_file.read_text(encoding="utf-8"))
        print(f"[generate] {persona_file.name}")
        result = generate_insights(persona, system_prompt)
        output_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Done.")


if __name__ == "__main__":
    main()
