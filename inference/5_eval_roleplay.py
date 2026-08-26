"""
Step 5: Role-Playing Quality Evaluation (LLM-as-Judge)

Scores each generated response on four dimensions adapted from CharacterEval
(Section 4.1 of the paper), using an LLM judge:

  1. Information Richness       — accurate use of retrieved facts, naturally integrated
  2. Logical Quality            — persona-aligned reasoning, logical consistency
  3. Character Consistency      — persona fidelity, knowledge boundary, speaking style
  4. Conversational Attractiveness — naturalness, empathy, engagement

Each dimension is scored on a 5-point scale. The paper reports the mean
across three independent judge calls on fixed model outputs (Section 4.1).
Run this script three times and average the results for the final score.

Usage:
    python 5_eval_roleplay.py \
        --responses    data/responses.json \
        --ground_truth data/ground_truth.json \
        --output       data/roleplay_scores.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import client, MODEL_ID

JUDGE_PROMPT = """\
You are an expert evaluator of role-playing conversation quality.

Persona:
{persona}

Ground-truth insight the agent should ideally express:
{core_insight}

Key facts the agent should naturally draw upon:
{core_fragments}

User's question:
{question}

Agent's response:
{answer}

Score the response on the following four dimensions. Each dimension is scored 0-5.

1. Information Richness
   5 - Facts are accurately used and seamlessly integrated; response feels uniquely tailored
   4 - Facts used accurately; minor mechanical feel
   3 - Facts present but clearly paraphrased mechanically
   2 - Only surface facts used; key context missing
   1 - Facts absent or irrelevant
   0 - Severely wrong or fabricated facts

2. Logical Quality
   5 - Persona-aligned reasoning, logically coherent, direct stance
   4 - Mostly coherent; slightly hedged
   3 - Logically self-consistent but vague; no clear stance
   2 - Logical gaps or contradicts persona
   1 - Incoherent reasoning
   0 - Completely illogical

3. Character Consistency
   5 - Perfectly in character; values, tone, knowledge boundary all consistent
   4 - Mostly in character; occasionally generic
   3 - Identity vague; language too generic
   2 - Weakly in character; slight OOC or knowledge breach
   1 - Severely out of character
   0 - Completely unrelated to persona

4. Conversational Attractiveness
   5 - Highly engaging; rich expression; empathetic; no AI-like formatting
   4 - Smooth and natural; good communication skills
   3 - Acceptable; lacks emotional depth; slight formulaic feel
   2 - Noticeably robotic; no empathy; structured formatting
   1 - Cold and mechanical; heavy reliance on lists/headers
   0 - Unreadable or nonsensical

Respond in JSON:
{{
  "information_richness": <0-5>,
  "logical_quality": <0-5>,
  "character_consistency": <0-5>,
  "conversational_attractiveness": <0-5>,
  "overall_score": <average of the four>,
  "brief_rationale": "<one sentence>"
}}
"""


def judge_response(sample: dict, gt: dict) -> dict:
    persona = gt.get("persona", "")
    core_insight = gt.get("core_insight", "")
    core_fragments = "\n".join(f"- {f}" for f in gt.get("core_fragments", []))

    prompt = JUDGE_PROMPT.format(
        persona=persona,
        core_insight=core_insight,
        core_fragments=core_fragments,
        question=sample["question"],
        answer=sample["answer"],
    )

    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    return json.loads(response.choices[0].message.content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses",    required=True, help="Path to responses JSON (output of step 2)")
    parser.add_argument("--ground_truth", required=True, help="Path to ground truth JSON")
    parser.add_argument("--output",       required=True, help="Path to save role-playing scores JSON")
    args = parser.parse_args()

    responses = json.loads(Path(args.responses).read_text(encoding="utf-8"))
    ground_truth = json.loads(Path(args.ground_truth).read_text(encoding="utf-8"))
    gt_map = {item["id"]: item for item in ground_truth}

    output_path = Path(args.output)
    if output_path.exists():
        results = json.loads(output_path.read_text(encoding="utf-8"))
        done_ids = {r["id"] for r in results}
    else:
        results = []
        done_ids = set()

    for sample in responses:
        sid = sample["id"]
        if sid in done_ids:
            print(f"[skip] {sid}")
            continue
        if sid not in gt_map:
            continue

        print(f"[judge] {sid}")
        scores = judge_response(sample, gt_map[sid])
        results.append({"id": sid, **scores})
        output_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # Print summary
    dims = ["information_richness", "logical_quality", "character_consistency",
            "conversational_attractiveness", "overall_score"]
    print(f"\n{'='*50}")
    for dim in dims:
        vals = [r[dim] for r in results if dim in r]
        if vals:
            print(f"  {dim:<35} {sum(vals)/len(vals):.3f}")
    print(f"{'='*50}")
    print(f"  Total evaluated: {len(results)}")
    print(f"\nNote: The paper averages scores over 3 independent judge runs.")


if __name__ == "__main__":
    main()
