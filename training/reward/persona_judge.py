"""
Persona-aware LLM judge for RL reward scoring.

Called by VeRL's reward manager to score the model's response
against ground-truth memory and persona (Section 3.3 of the paper).

Scoring dimensions (weighted):
  - Factual utility and accuracy  (30%)
  - Logical consistency + persona alignment (60%)
  - Value / safety                (10%)

Returns a float in [0.0, 1.0].

Configure the judge endpoint via environment variables:
  JUDGE_API_KEY   — API key for the judge LLM
  JUDGE_BASE_URL  — Base URL of an OpenAI-compatible endpoint
  JUDGE_MODEL_ID  — Model identifier to use as judge
"""

import re
import os
import logging
import time
from openai import OpenAI

logger = logging.getLogger(__file__)

API_KEY  = os.environ.get("JUDGE_API_KEY",  "YOUR_JUDGE_API_KEY_HERE")
BASE_URL = os.environ.get("JUDGE_BASE_URL", "YOUR_JUDGE_BASE_URL_HERE")
MODEL_ID = os.environ.get("JUDGE_MODEL_ID", "YOUR_JUDGE_MODEL_ID_HERE")

try:
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
except Exception as e:
    logger.error(f"Failed to initialize judge client: {e}")
    client = None

SYSTEM_PROMPT = """\
# Role
You are an open-domain QA quality auditor. Given a persona and a reference answer, \
score the AI response strictly and objectively.

# Evaluation Principles
1. **Reference positioning**: <standard_reference> provides the facts and angle used \
for this question. If the AI uses different facts or a different angle that is logically \
consistent and factually correct, that is also valid.
2. **Logic first**: For open-ended questions, strict logical coherence is the most \
important criterion.
3. **Zero tolerance for filler**: Vague hedging and content-free pleasantries incur \
heavy deductions.

# Scoring Dimensions (0.0 – 1.0)
Compute the final score as a weighted combination of the three dimensions below.

## 1. Factual Utility and Accuracy (30%)
* **1.0**: Facts fully accurate, covering core facts from the reference or equivalent \
facts with tight reasoning.
* **0.6–0.9**: Facts accurate but incomplete; only partial key points covered.
* **0.1–0.5**: Lacks substance; mostly generic remarks with low information density.
* **0.0**: Contains factual errors, hallucinations, or logical contradictions.

## 2. Logical Consistency and Persona Alignment (60%)
* **1.0**: Tight argumentation, fully consistent with the tone and reasoning style \
required by <persona>. Even novel viewpoints must form a closed logical loop. \
Effectively answers the question.
* **0.5–0.9**: Mostly coherent; persona mostly maintained, but with minor leaps or \
tone drift.
* **0.0–0.4**: Self-contradictory, forced causal attribution, deviates from <persona>, \
or severely off-topic.

## 3. Value and Safety (10%)
* **1.0**: Neutral, objective, and constructive.
* **0.0**: Contains bias, discrimination, malicious content, or policy violations.

# Input Data
- <user_query>: {user_query}
- <standard_reference>: {standard_reference}
- <persona>: {persona}
- <ai_response>: {ai_response}

# Output Requirement
Output exactly one floating-point number as the final score (range 0.0 – 1.0).
No analysis or extra text.
"""


def compute_score(solution_str, ground_truth, prompt, persona, **kwargs) -> float:
    boxed = last_boxed_only_string(solution_str)
    model_answer = remove_boxed(boxed) if boxed is not None else solution_str

    if isinstance(ground_truth, list):
        ground_truth = "; ".join(str(x) for x in ground_truth)

    return llm_judge_score(prompt, ground_truth, model_answer, persona)


def llm_judge_score(query, reference, response, persona, max_retries=5) -> float:
    if client is None:
        logger.error("Judge client not initialized.")
        return 0.0

    prompt = SYSTEM_PROMPT.format(
        user_query=query,
        standard_reference=reference,
        persona=persona,
        ai_response=response,
    )

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=MODEL_ID,
                temperature=0.0,
                max_tokens=10,
            )
            content = completion.choices[0].message.content.strip()
            match = re.search(r"(\d+(\.\d+)?)", content)
            if match:
                score = float(match.group(1))
                if score > 1.0:
                    score /= 10.0
                return min(max(score, 0.0), 1.0)
            logger.warning(f"[Attempt {attempt+1}/{max_retries}] Unexpected format: {content}")
        except Exception as e:
            logger.warning(f"[Attempt {attempt+1}/{max_retries}] API call failed: {e}")

        if attempt < max_retries - 1:
            time.sleep(1)

    logger.error(f"Judge failed after {max_retries} attempts.")
    return 0.0


# ── Utility helpers ────────────────────────────────────────────────

def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        if s[:len(left)] == left:
            return s[len(left):]
    left = "\\boxed{"
    if s[:len(left)] == left and s[-1] == "}":
        return s[len(left):-1]
    return s


def last_boxed_only_string(string):
    if string is None:
        return None
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    idx = string.rfind("\\boxed")
    if idx < 0:
        return None
    i, right_brace_idx, num_open = idx, None, 0
    while i < len(string):
        if string[i] == "{":
            num_open += 1
        if string[i] == "}":
            num_open -= 1
            if num_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        return None
    return string[idx:right_brace_idx + 1]
