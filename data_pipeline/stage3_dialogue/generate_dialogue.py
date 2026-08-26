"""
Stage 3a: Dense Conversational Weaving

Embeds factual fragments into natural multi-turn dialogues. The casual
conversation topic serves as the backbone; fragments appear as incidental
mentions rather than focal points, making them difficult to spot without
careful reading.

Each output dialogue covers approximately 24 rounds (48 messages).
"""

import json
import sys
import random
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import client, MODEL_ID

PROMPT_PATH = Path(__file__).parent / "prompts" / "generate_dialogue.txt"
INPUT_DIR = Path("data/fact_query_checked")
OUTPUT_DIR = Path("data/dialogues")

MAX_RETRIES = 3

# Default casual conversation topics used to anchor each dialogue.
# The factual fragments are embedded as incidental mentions within these topics.
# Replace or extend this list to match your domain.
DEFAULT_TOPICS = [
    "weekend plans", "cooking and recipes", "travel destinations",
    "movies and TV shows", "fitness and health", "music", "books",
    "technology and gadgets", "hobbies", "work-life balance",
    "home decoration", "pets", "sports", "online shopping",
    "learning a new skill", "local events", "weather and seasons",
    "social media", "food and restaurants", "childhood memories",
]


def load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def load_topics() -> list[str]:
    topics_path = Path(__file__).parent / "topics.json"
    if topics_path.exists():
        topics_data = json.loads(topics_path.read_text(encoding="utf-8"))
        topics = []
        for items in topics_data.values():
            if isinstance(items, list):
                topics.extend(items)
            elif isinstance(items, dict):
                for sub in items.values():
                    if isinstance(sub, list):
                        topics.extend(sub)
        return topics if topics else DEFAULT_TOPICS
    return DEFAULT_TOPICS


def generate_dialogue(user_persona: str, npc_persona: str, topic: str,
                      fragments: list[str], system_prompt: str) -> dict:
    user_msg = json.dumps(
        {
            "user_persona": user_persona,
            "npc_persona": npc_persona,
            "talk_topics": topic,
            "fragments": fragments,
        },
        ensure_ascii=False,
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
    topics = load_topics()

    for input_file in sorted(INPUT_DIR.glob("*.json")):
        output_file = OUTPUT_DIR / input_file.name
        if output_file.exists():
            print(f"[skip] {input_file.name}")
            continue

        data = json.loads(input_file.read_text(encoding="utf-8"))
        persona = data["persona"]
        npc_persona = persona.get("role_description", "")
        user_persona = persona.get("user_description", "")
        print(f"[generate] {input_file.name} — {len(data['samples'])} samples")

        results = []
        for sample in data["samples"]:
            topic = random.choice(topics)
            fragments = sample["fragments"]

            dialogue_data = None
            for attempt in range(MAX_RETRIES):
                try:
                    dialogue_data = generate_dialogue(
                        user_persona, npc_persona, topic, fragments, system_prompt
                    )
                    break
                except Exception as e:
                    wait = 2 ** attempt
                    print(f"  [retry {attempt+1}] {e} — waiting {wait}s")
                    time.sleep(wait)

            if dialogue_data is None:
                print(f"  [fail] skipping sample after {MAX_RETRIES} retries")
                continue

            results.append({**sample, "topic": topic, "dialogue": dialogue_data})

        output = {**data, "samples": results}
        output_file.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Done.")


if __name__ == "__main__":
    main()
