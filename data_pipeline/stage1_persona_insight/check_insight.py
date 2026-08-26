"""
Stage 1b: Insight Specificity Check (Quality Assurance Step 1)

Verifies that each generated insight genuinely requires the persona's unique
perspective rather than generic domain knowledge. Insights that fail this
check are discarded before fact-query generation.

QA criterion: insight specificity
  - The insight must reflect the persona's unique cognitive framework.
  - A generic reader without the persona background should NOT reach the
    same conclusion from the given facts alone.
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import client, MODEL_ID

PROMPT_PATH = Path(__file__).parent / "prompts" / "check_insight.txt"
INPUT_DIR = Path("data/insights")
OUTPUT_DIR = Path("data/insights_checked")

PASS_THRESHOLD = 4  # out of 4 dimensions


def load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def check_insight(insight_data: dict, system_prompt: str) -> list[dict]:
    """Returns a list of checked insights, each with a verdict field."""
    user_msg = json.dumps(insight_data, ensure_ascii=False)
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    return json.loads(response.choices[0].message.content)["results"]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    system_prompt = load_prompt()

    total, passed = 0, 0
    for insight_file in sorted(INPUT_DIR.glob("*.json")):
        output_file = OUTPUT_DIR / insight_file.name
        if output_file.exists():
            print(f"[skip] {insight_file.name}")
            continue

        insight_data = json.loads(insight_file.read_text(encoding="utf-8"))
        print(f"[check] {insight_file.name}")
        results = check_insight(insight_data, system_prompt)

        passed_results = [r for r in results if r.get("score_total", 0) >= PASS_THRESHOLD]
        total += len(results)
        passed += len(passed_results)

        output = {**insight_data, "checked_insights": passed_results}
        output_file.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Done. Pass rate: {passed}/{total} ({passed/total:.1%})")


if __name__ == "__main__":
    main()
