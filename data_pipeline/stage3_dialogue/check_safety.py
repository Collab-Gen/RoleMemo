"""
Stage 3c: Safety Filter (Quality Assurance Step 4)

Filters all generated dialogues for violence and privacy violations.
Any dialogue containing flagged content is discarded entirely.

QA criterion: safety
  - No depictions of physical violence, self-harm, or threats.
  - No real personal information (phone numbers, ID numbers, addresses).
  - No content involving minors in inappropriate contexts.

Two-pass approach:
  1. Rule-based keyword scan (fast, zero API cost).
  2. LLM-based review for borderline cases flagged by the keyword scan.
"""

import json
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import client, MODEL_ID

INPUT_DIR = Path("data/dialogues_checked")
OUTPUT_DIR = Path("data/dialogues_safe")

# --- Pass 1: Rule-based keyword patterns ---
VIOLENCE_PATTERNS = [
    r"\bkill\b", r"\bmurder\b", r"\bsuicide\b", r"\bself.harm\b",
    r"\bthreaten\b", r"\battack\b", r"\bassault\b",
]
PRIVACY_PATTERNS = [
    r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",   # phone numbers
    r"\b\d{9,18}\b",                           # ID / bank card numbers
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b",  # email
]
ALL_PATTERNS = [re.compile(p, re.IGNORECASE) for p in VIOLENCE_PATTERNS + PRIVACY_PATTERNS]

LLM_REVIEW_PROMPT = """\
You are a content safety reviewer. Given a dialogue, determine whether it
contains any of the following:
1. Depictions of violence, self-harm, or threats.
2. Real personal information (phone numbers, ID numbers, addresses, emails).
3. Content involving minors in inappropriate contexts.

Respond in JSON:
{"safe": true/false, "reason": "brief explanation if unsafe"}
"""


def keyword_scan(dialogue: list[dict]) -> bool:
    """Returns True if any unsafe pattern is found."""
    full_text = " ".join(turn["content"] for turn in dialogue)
    return any(p.search(full_text) for p in ALL_PATTERNS)


def llm_safety_review(dialogue: list[dict]) -> dict:
    text = "\n".join(f"{t['role']}: {t['content']}" for t in dialogue)
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": LLM_REVIEW_PROMPT},
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    return json.loads(response.choices[0].message.content)


def is_safe(sample: dict) -> tuple[bool, str]:
    dialogue = sample.get("dialogue", {}).get("dialogue", [])

    # Pass 1: fast keyword scan
    if not keyword_scan(dialogue):
        return True, "clean"

    # Pass 2: LLM review for borderline cases
    verdict = llm_safety_review(dialogue)
    if verdict.get("safe", True):
        return True, "llm:clean"
    return False, verdict.get("reason", "unsafe")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total, passed = 0, 0
    for input_file in sorted(INPUT_DIR.glob("*.json")):
        output_file = OUTPUT_DIR / input_file.name
        if output_file.exists():
            print(f"[skip] {input_file.name}")
            continue

        data = json.loads(input_file.read_text(encoding="utf-8"))
        print(f"[filter] {input_file.name} — {len(data['samples'])} samples")

        safe_samples = []
        for sample in data["samples"]:
            total += 1
            ok, reason = is_safe(sample)
            if ok:
                passed += 1
                safe_samples.append(sample)
            else:
                print(f"  [discard] {reason}")

        output = {**data, "samples": safe_samples}
        output_file.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Done. Pass rate: {passed}/{total} ({passed/total:.1%})")


if __name__ == "__main__":
    main()
