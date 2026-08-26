"""
Stage 3b: Difficulty Control Check (Quality Assurance Step 3)

Verifies that each factual fragment appears in at most one dialogue turn.
Fragments that repeat across multiple turns create exploitable high-frequency
patterns, allowing retrieval models to locate evidence without genuine
reasoning. Such instances are discarded.

QA criterion: difficulty control
  - fragment_1 must appear in exactly one User turn.
  - fragment_2 must appear in exactly one User turn.
  - No fragment content may be echoed or paraphrased by the NPC turn
    immediately following its appearance.
"""

import json
from pathlib import Path

INPUT_DIR = Path("data/dialogues")
OUTPUT_DIR = Path("data/dialogues_checked")


def count_turns_containing(fragment: str, dialogue: list[dict]) -> int:
    """Count how many User turns contain a paraphrase of the fragment."""
    fragment_words = set(fragment.lower().split())
    count = 0
    for turn in dialogue:
        if turn["role"].lower() != "user":
            continue
        turn_words = set(turn["content"].lower().split())
        # Simple overlap heuristic: >50% of fragment words appear in the turn
        overlap = len(fragment_words & turn_words) / max(len(fragment_words), 1)
        if overlap > 0.5:
            count += 1
    return count


def npc_echoes_fragment(fragment: str, dialogue: list[dict]) -> bool:
    """Check whether any NPC turn directly echoes fragment content."""
    fragment_words = set(fragment.lower().split())
    for turn in dialogue:
        if turn["role"].lower() in ("assistant", "npc"):
            turn_words = set(turn["content"].lower().split())
            overlap = len(fragment_words & turn_words) / max(len(fragment_words), 1)
            if overlap > 0.5:
                return True
    return False


def check_sample(sample: dict) -> tuple[bool, str]:
    dialogue = sample.get("dialogue", {}).get("dialogue", [])
    fragments = sample.get("fragments", [])

    for i, fragment in enumerate(fragments):
        count = count_turns_containing(fragment, dialogue)
        if count != 1:
            return False, f"fragment_{i+1} appears in {count} turns (expected 1)"
        if npc_echoes_fragment(fragment, dialogue):
            return False, f"fragment_{i+1} echoed by NPC turn"

    return True, "ok"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total, passed = 0, 0
    for input_file in sorted(INPUT_DIR.glob("*.json")):
        output_file = OUTPUT_DIR / input_file.name
        if output_file.exists():
            print(f"[skip] {input_file.name}")
            continue

        data = json.loads(input_file.read_text(encoding="utf-8"))
        print(f"[check] {input_file.name} — {len(data['samples'])} samples")

        passed_samples = []
        for sample in data["samples"]:
            total += 1
            ok, reason = check_sample(sample)
            if ok:
                passed += 1
                passed_samples.append(sample)
            else:
                print(f"  [discard] {reason}")

        output = {**data, "samples": passed_samples}
        output_file.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Done. Pass rate: {passed}/{total} ({passed/total:.1%})")


if __name__ == "__main__":
    main()
