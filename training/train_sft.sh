#!/bin/bash
# SFT Training Script for DualMem-SFT
#
# Trains the 4B memory construction model (f_theta) on RoleMemo using
# Supervised Fine-Tuning via LLaMA-Factory.
#
# Paper: Section 3.3 (Training Mechanisms on RoleMemo)
#   - Base model: Qwen3-4B
#   - Learning rate: 1e-5 with cosine scheduling
#   - Warmup ratio: 0.05
#   - Effective batch size: 32 (per_device x gradient_accumulation x num_gpus)
#   - Steps: 1000
#
# Requirements:
#   - LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory
#   - Register the RoleMemo dataset in LLaMA-Factory's dataset_info.json
#     before running this script.

# ── User configuration ────────────────────────────────────────────
LLAMAFACTORY_DIR="/path/to/LLaMA-Factory"   # LLaMA-Factory repo root
MODEL_PATH="/path/to/Qwen3-4B"              # Base model checkpoint
DATASET_DIR="/path/to/rolememo/train"       # Directory containing dataset files
OUTPUT_DIR="./saves/DualMem-SFT"           # Where to save checkpoints
# ─────────────────────────────────────────────────────────────────

DATASET_NAME="rolememo_train"
EVAL_DATASET_NAME="rolememo_eval"

cd $LLAMAFACTORY_DIR

deepspeed --num_gpus 8 src/train.py \
    --deepspeed config/ds_z3_offload.json \
    --stage sft \
    --do_train \
    --do_eval \
    --model_name_or_path $MODEL_PATH \
    --flash_attn fa2 \
    --dataset $DATASET_NAME \
    --dataset_dir $DATASET_DIR \
    --eval_dataset $EVAL_DATASET_NAME \
    --template qwen \
    --finetuning_type full \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --cutoff_len 6144 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --max_steps 1000 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --bf16 True \
    --gradient_checkpointing True \
    --trust_remote_code True \
    --logging_steps 10 \
    --eval_strategy steps \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 3 \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --greater_is_better False \
    --plot_loss True
