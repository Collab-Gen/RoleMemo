"""
Step 1: Memory Construction (f_theta)

Processes a conversation history incrementally in fixed-size chunks.
For each chunk, the memory model extracts:
  - Factual cognition (F_i): objective events from the current chunk
  - Insight cognition (I_i): persona-conditioned interpretations derived from F_i

The outputs are stored in a unified memory bank M = union of M_i across all chunks.
Each insight records references to the facts it is grounded in, maintaining
interpretive traceability at inference time (Section 3.1 of the paper).

Usage:
    python 1_generate_memory.py \
        --dialogue data/dialogue.json \
        --persona  data/persona.json \
        --output   data/memory_bank.json \
        --model    /path/to/DualMem-RL
"""

import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

CHUNK_SIZE = 12  # dialogue turns per chunk (12 turns = 6 user + 6 npc)

MEMORY_PROMPT_TEMPLATE = """\
You are a memory construction model operating from the perspective of the following persona:
{persona}

You have processed the conversation so far and accumulated the following memory:
{prior_memory}

Now read the new conversation chunk:
{chunk}

Your task:
1. Extract factual cognition (F): objective events and details mentioned by the user that are relevant to the persona. Format each fact as:
   Fact <id>: <content>

2. Construct insight cognition (I): persona-conditioned interpretations derived from the facts above and prior memory. Each insight must reference the specific fact IDs it is grounded in. Only generate an insight when two or more facts jointly support a clear persona-driven interpretation. If evidence is insufficient, output null. Format each insight as:
   Insight <id> (Fact <ref_ids>): <content>

Output strictly in JSON:
{{
  "facts": [
    {{"id": 1, "text": "..."}},
    ...
  ],
  "insights": [
    {{"id": 1, "refs": [1, 2], "text": "..."}},
    ...
  ]
}}
If no insight can be derived, set "insights" to [].
"""


class MemoryBank:
    """Maintains the unified memory bank M across all chunks."""

    def __init__(self):
        self.facts = []     # list of {"id": int, "text": str}
        self.insights = []  # list of {"id": int, "refs": [int], "text": str}

    def add_chunk(self, chunk_facts: list[dict], chunk_insights: list[dict]):
        """Merge a chunk's cognitions into the global bank, re-indexing IDs."""
        fact_id_offset = len(self.facts)
        for f in chunk_facts:
            self.facts.append({"id": fact_id_offset + f["id"], "text": f["text"]})

        insight_id_offset = len(self.insights)
        for ins in chunk_insights:
            remapped_refs = [fact_id_offset + r for r in ins.get("refs", [])]
            self.insights.append({
                "id": insight_id_offset + ins["id"],
                "refs": remapped_refs,
                "text": ins["text"],
            })

    def format_prior(self) -> str:
        """Format the current memory bank as prompt context (M_{<i})."""
        if not self.facts and not self.insights:
            return "(none)"
        lines = []
        for f in self.facts:
            lines.append(f"Fact {f['id']}: {f['text']}")
        for ins in self.insights:
            refs = ", ".join(str(r) for r in ins["refs"])
            lines.append(f"Insight {ins['id']} (Fact {refs}): {ins['text']}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"facts": self.facts, "insights": self.insights}


def chunk_dialogue(dialogue: list[dict]) -> list[list[dict]]:
    """Split dialogue turns into fixed-size chunks."""
    return [dialogue[i:i + CHUNK_SIZE] for i in range(0, len(dialogue), CHUNK_SIZE)]


def format_chunk(turns: list[dict]) -> str:
    return "\n".join(f"{t['role']}: {t['content']}" for t in turns)


def run_model(model, tokenizer, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            temperature=1.0,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def parse_model_output(raw: str) -> tuple[list[dict], list[dict]]:
    """Parse JSON output from the model; return (facts, insights)."""
    try:
        # Extract JSON from output (model may wrap it in markdown)
        start = raw.find("{")
        end = raw.rfind("}") + 1
        data = json.loads(raw[start:end])
        facts = data.get("facts", [])
        insights = [i for i in data.get("insights", []) if i]  # drop null entries
        return facts, insights
    except Exception:
        return [], []


def build_memory(dialogue: list[dict], persona: str, model, tokenizer) -> MemoryBank:
    """
    Main pipeline: process dialogue chunks sequentially, accumulating M.
    Corresponds to Equation (1) in the paper: M_i = f_theta(C_i, M_{<i}, p)
    """
    bank = MemoryBank()
    chunks = chunk_dialogue(dialogue)

    for i, chunk in enumerate(chunks):
        print(f"  Processing chunk {i+1}/{len(chunks)} ...")
        prompt = MEMORY_PROMPT_TEMPLATE.format(
            persona=persona,
            prior_memory=bank.format_prior(),
            chunk=format_chunk(chunk),
        )
        raw_output = run_model(model, tokenizer, prompt)
        facts, insights = parse_model_output(raw_output)
        bank.add_chunk(facts, insights)

    return bank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dialogue", required=True, help="Path to dialogue JSON")
    parser.add_argument("--persona",  required=True, help="Path to persona JSON")
    parser.add_argument("--output",   required=True, help="Path to save memory bank JSON")
    parser.add_argument("--model",    required=True, help="Path to DualMem model checkpoint")
    args = parser.parse_args()

    dialogue = json.loads(Path(args.dialogue).read_text(encoding="utf-8"))
    persona_data = json.loads(Path(args.persona).read_text(encoding="utf-8"))
    persona_text = persona_data.get("description", json.dumps(persona_data, ensure_ascii=False))

    print(f"Loading model from {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model.eval()

    print("Building memory bank ...")
    bank = build_memory(dialogue, persona_text, model, tokenizer)

    Path(args.output).write_text(
        json.dumps(bank.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Memory bank saved to {args.output}")
    print(f"  Facts:   {len(bank.facts)}")
    print(f"  Insights:{len(bank.insights)}")


if __name__ == "__main__":
    main()
