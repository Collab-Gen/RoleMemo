# From Facts to Insights: A Persona-Driven Dual Memory Framework and Dataset for Role-Playing Agents


## Overview

Role-playing agents excel in short conversations, but long-term interactions overwhelm context windows. Existing memory frameworks address this with **persona-agnostic summarization** — recording facts as neutral observations — which forces the agent to reinterpret everything at inference time from diluted retrieved context, resulting in generic, out-of-character responses.

We argue that memory should be an **active cognitive process**, not a neutral fact repository. A psychologist agent shouldn't store "late-night gaming"; it should store "behavioral fatigue" — a persona-conditioned interpretation grounded in the fact.

This repository accompanies our paper, which introduces:

1. **RoleMemo** — a large-scale benchmark featuring four persona-conditioned reasoning task types, spanning conversation histories up to 256k tokens, with 2,052 personas and 20,244 queries.
2. **DualMem** — a dual memory framework that decouples memory into *factual cognition* (objective events) and *insight cognition* (persona-driven interpretations grounded in facts), implemented as a trained 4B-parameter model that outperforms zero-shot frameworks driven by 685B-parameter models.

---

## Key Results

Evaluations on RoleMemo show that persona-agnostic frameworks — regardless of driving model scale — systematically fail to retrieve persona-conditioned insights, yielding a structural **insight bottleneck**. DualMem addresses this with a dedicated dual-stream memory model:

- **Recall@10 (Fact & Insight):** DualMem-SFT and DualMem-RL substantially outperform all zero-shot baselines on both factual and insight retrieval.
- **Role-Playing Quality:** Evaluated across four in-character dimensions (information richness, logical quality, character consistency, conversational attractiveness), DualMem-RL achieves the highest scores — with a 4B model surpassing zero-shot frameworks driven by 685B-parameter models.

Full numerical results are reported in the paper (Table 1 and Table 2).

---

## Repository Structure

```
RoleMemo/
├── config.py               # Shared API configuration for inference & evaluation
├── data_pipeline/          # Dataset construction pipeline
│   ├── config.py           # Shared API configuration for data pipeline scripts
│   ├── stage1_persona_insight/   # Stage 1: generate & QA-check insights
│   ├── stage2_fact_query/        # Stage 2: generate fact-query pairs & check memory necessity
│   └── stage3_dialogue/          # Stage 3: weave dialogues & check difficulty/safety
│
├── inference/              # Evaluation pipeline (numbered steps)
│   ├── 1_generate_memory.py      # Run f_theta on long dialogue histories
│   ├── 2_generate_response.py    # Run role-playing agent with DualMem retrieval
│   ├── 3_eval_retrieval.py       # Compute per-sample retrieval similarity scores
│   ├── 4_calculate_recall.py     # Aggregate Recall@10 (Fact) and Recall@10 (Insight)
│   └── 5_eval_roleplay.py        # LLM-as-Judge on 4 role-playing dimensions
│
└── training/               # Training scripts
    ├── train_sft.sh              # SFT via LLaMA-Factory (Qwen3-4B, 1000 steps)
    ├── train_rl.sh               # RL via VeRL/GRPO
    ├── memory_manager.py         # Dual-stream memory bank (used during RL)
    └── reward/
        └── persona_judge.py      # LLM judge reward function for RL
```

---

## Quickstart

### 1. Configure your API

There are two config files — one for data pipeline scripts, one for inference and evaluation:

- **`data_pipeline/config.py`** — used by all scripts under `data_pipeline/`
- **`config.py`** (project root) — used by all scripts under `inference/`

Set credentials via environment variables (recommended):

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="your-base-url"   # any OpenAI-compatible endpoint
export MODEL_ID="your-model-id"
```

Or edit the placeholder strings in the respective `config.py` files directly.

### 2. Data Pipeline

Each stage reads from and writes to fixed directories under `data/`. Place your input persona files under `data/personas/` (one JSON per persona) before starting.

```
data/
├── personas/           ← your input (one JSON file per persona)
├── insights/           ← Stage 1a output
├── insights_checked/   ← Stage 1b output
├── fact_query/         ← Stage 2a output
├── fact_query_checked/ ← Stage 2b output
├── dialogues/          ← Stage 3a output
├── dialogues_checked/  ← Stage 3b output
└── dialogues_safe/     ← Stage 3c output (final dataset)
```

Run all scripts from the repository root:

```bash
# Stage 1a: Generate persona-conditioned insights
python data_pipeline/stage1_persona_insight/generate_insight.py

# Stage 1b: QA check — insight specificity
python data_pipeline/stage1_persona_insight/check_insight.py

# Stage 2a: Generate fact-query pairs
python data_pipeline/stage2_fact_query/generate_fact_query.py

# Stage 2b: QA check — memory necessity
python data_pipeline/stage2_fact_query/check_memory_necessity.py

# Stage 3a: Weave facts into natural dialogues
python data_pipeline/stage3_dialogue/generate_dialogue.py

# Stage 3b: QA check — difficulty control
python data_pipeline/stage3_dialogue/check_difficulty.py

# Stage 3c: QA check — safety filter
python data_pipeline/stage3_dialogue/check_safety.py
```

All stages support resumption: already-processed files are skipped automatically.

### 3. Inference & Evaluation

Run the numbered steps in order:

```bash
# Step 1: Generate memory from long dialogue histories using f_theta
python inference/1_generate_memory.py \
    --model    /path/to/DualMem-RL \
    --dialogue data/dialogue.json \
    --persona  data/persona.json \
    --output   data/memory_bank.json

# Step 2: Generate role-playing responses using retrieved memory
python inference/2_generate_response.py \
    --memory  data/memory_bank.json \
    --persona data/persona.json \
    --queries data/queries.json \
    --output  data/responses.json

# Step 3-4: Retrieval evaluation (Recall@10)
python inference/3_eval_retrieval.py --responses data/responses.json --ground_truth data/ground_truth.json --output data/retrieval_scores.json
python inference/4_calculate_recall.py --scores data/retrieval_scores.json

# Step 5: Role-playing quality evaluation (run 3 times and average)
python inference/5_eval_roleplay.py --responses data/responses.json --ground_truth data/ground_truth.json --output data/roleplay_scores.json
```

### 4. Training

```bash
# SFT (requires LLaMA-Factory)
bash training/train_sft.sh

# RL (requires VeRL)
bash training/train_rl.sh
```

See the comments inside each script for required path configurations.

---

## What's in This Release

This is an initial release focused on reproducibility of the core paper results.

| Component | Status |
|---|---|
| Data pipeline scripts (all 3 stages + 4 QA checks) | ✅ Released |
| Inference & evaluation pipeline (Steps 1–5) | ✅ Released |
| SFT training script (LLaMA-Factory) | ✅ Released |
| RL training script (VeRL/GRPO) | ✅ Released |
| RL reward function (`persona_judge.py`) | ✅ Released |
| Memory manager (`memory_manager.py`) | ✅ Released |
| RoleMemo dataset | 🔜 Coming soon |
| DualMem-SFT model checkpoint | 🔜 Coming soon |
| DualMem-RL model checkpoint | 🔜 Coming soon |
| Full RL agent implementation (recurrent memory module) | 🔜 Coming soon |

> **Note on the RL agent:** The full recurrent memory agent builds on [MemAgent](https://github.com/mem-agent/mem-agent). We are working on a cleaner standalone release. In the meantime, `memory_manager.py` documents the core dual-stream memory bank logic used during training.

---

## Acknowledgements

This work builds on the following open-source projects:

- **[MemAgent](https://github.com/mem-agent/mem-agent)** (Apache 2.0) — Our RL training framework extends MemAgent's recurrent memory infrastructure. We gratefully acknowledge their foundational work on agent-based memory with VeRL/GRPO.
- **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)** (Apache 2.0) — Used for SFT training of the 4B memory construction model.
- **[VeRL](https://github.com/volcengine/verl)** (Apache 2.0) — Used as the underlying RL training engine.
- **[Qwen3](https://huggingface.co/Qwen)** — Base model for f_theta (Qwen3-4B) and retrieval embeddings (Qwen3-Embedding-0.6B).
