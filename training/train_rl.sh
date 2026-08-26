#!/bin/bash
# RL Training Script for DualMem-RL
#
# Trains the 4B memory construction model (f_theta) on RoleMemo using
# GRPO-based Reinforcement Learning via VeRL.
#
# Paper: Section 3.3 (Training Mechanisms on RoleMemo)
#   - Base model: Qwen3-4B (initialized from DualMem-SFT checkpoint)
#   - Algorithm: GRPO (adv_estimator=grpo)
#   - Learning rate: 1e-6
#   - Warmup steps: 20
#   - Train batch size: 16 (rollout n=8 → effective batch 16*8=128 responses)
#   - KL loss coefficient: 0.001
#   - Chunk size: 4000 tokens per memory-processing step
#
# Requirements:
#   - VeRL: https://github.com/volcengine/verl
#   - A VeRL-compatible reward function (see reward/persona_judge.py)
#   - Training data in parquet format under data/persona_task/

# ── User configuration ────────────────────────────────────────────
MODEL_PATH="/path/to/DualMem-SFT-checkpoint"   # SFT checkpoint to start from
TRAIN_PATH="./data/persona_task/train.parquet"
VAL_PATH="./data/persona_task/test.parquet"
OUTPUT_DIR="./saves/DualMem-RL"
# ─────────────────────────────────────────────────────────────────

export WANDB_PROJECT="RoleMemo-RL"
export WANDB_NAME="DualMem-RL-4B"
export RAY_USAGE_STATS_ENABLED=0

NNODES=1
NGPUS_PER_NODE=8

python3 -m verl.trainer.main_ppo \
    recurrent.enable=memory \
    recurrent.memory.config.chunk_size=4000 \
    algorithm.adv_estimator=grpo \
    algorithm.grpo_use_adv=False \
    trainer.save_freq=10 \
    trainer.logger=['console','wandb'] \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=20 \
    actor_rollout_ref.actor.clip_ratio_high=0.20 \
    actor_rollout_ref.actor.entropy_coeff=0.000 \
    data.train_files=$TRAIN_PATH \
    data.val_files=$VAL_PATH \
    data.shuffle=True \
    data.filter_overlong_prompts=True \
    data.train_batch_size=16 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    data.truncation='center' \
    +data.context_key='context' \
    data.max_prompt_length=8192 \
    data.max_response_length=1024 \
    reward_model.reward_manager='thread' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32768 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$WANDB_NAME \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$NGPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.test_freq=5 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.total_epochs=1
