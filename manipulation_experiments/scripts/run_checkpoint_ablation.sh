#!/usr/bin/env bash
#
# run_checkpoint_ablation.sh
#
# Launches all 12 checkpoint ablation runs:
#   Can task, base policy step_6000, 4 models x 3 seeds
#
# This ablation studies the effect of using a later base policy checkpoint
# (step_6000 instead of step_1000) on the Can task.
#
# Usage:
#   bash scripts/run_checkpoint_ablation.sh              # execute all runs
#   DRY_RUN=1 bash scripts/run_checkpoint_ablation.sh    # print commands only
#   NUM_GPUS=4 bash scripts/run_checkpoint_ablation.sh   # use 4 GPUs per run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

NUM_GPUS="${NUM_GPUS:-1}"
DRY_RUN="${DRY_RUN:-0}"

cd "$REPO_DIR"

# ---------------------------------------------------------------------------
# Helper: run or print a command
# ---------------------------------------------------------------------------
run_cmd() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "$@"
    echo ""
  else
    echo "=== Running: $@ ==="
    "$@"
  fi
}

# ---------------------------------------------------------------------------
# Shared parameters for ALL checkpoint ablation runs (Can, step_6000)
# ---------------------------------------------------------------------------
SHARED_ARGS=(
  --distributed True
  --base-policy-wandb-project flow-bc
  --base_policy_wandb_run_id 95j3noe4
  --load-ema True
  --checkpoint_step step_6000
  --wandb_project flow-bc-fpo-finetuning
  --wandb_enable True
  --total_timesteps 5000000
  --task Can
  --eval_env Can
  --gradient_accumulation_steps 1
  --num_minibatches 8
  --log_freq 1
  --save_freq 2
  --rollout_freq 2
  --eval_num_episodes 200
  --data_collection_steps 1600
  --do_chunk_level_ppo True
  --eval_ema False
  --exploration_noise_std None
  --freeze_vision_encoder True
  --gae_lambda 0.99
  --n_action_samples 8
  --n_action_steps 16
  --num_envs 30
  --sampling_steps 10
  --spo_clip_coef 0.01
  --zero_sampling True
  --discount 0.99
)

###########################################################################
# FPO++ (3 runs)
# Run IDs: txf5crib (seed 0), ylxhw9uu (seed 1), 6zp46k32 (seed 2)
###########################################################################
echo "====================================================================="
echo "  Checkpoint Ablation: FPO++ (3 runs)"
echo "====================================================================="

for SEED in 0 1 2; do
  run_cmd torchrun --nproc_per_node="$NUM_GPUS" finetune_online_rl.py \
    "${SHARED_ARGS[@]}" \
    --experiment "finetune-fpo++-can-ckpt-ablation" \
    --sde_sigma 0 \
    --cfm_loss_average_group_size 1 \
    --cfm_loss_use_huber True \
    --cfm_loss_huber_delta 0.5 \
    --clip_coef 0.02 \
    --max_grad_norm 5 \
    --clamp_logratio 5 \
    --clamp_old_cfm_loss 4 \
    --trust_region_mode ppo \
    --seed "$SEED"
done

###########################################################################
# Vanilla FPO (3 runs)
# Run IDs: gzm22fqh (seed 0), opygcs30 (seed 1), 89jg66ll (seed 2)
###########################################################################
echo "====================================================================="
echo "  Checkpoint Ablation: Vanilla FPO (3 runs)"
echo "====================================================================="

for SEED in 0 1 2; do
  run_cmd torchrun --nproc_per_node="$NUM_GPUS" finetune_online_rl.py \
    "${SHARED_ARGS[@]}" \
    --experiment "finetune-vanilla-fpo-can-ckpt-ablation" \
    --sde_sigma 0 \
    --cfm_loss_average_group_size -1 \
    --cfm_loss_use_huber False \
    --cfm_loss_huber_delta 0.5 \
    --clip_coef 0.02 \
    --max_grad_norm 5 \
    --clamp_logratio None \
    --clamp_old_cfm_loss None \
    --trust_region_mode ppo \
    --seed "$SEED"
done

###########################################################################
# DPPO Learned Noise (3 runs)
# Run IDs: 7lz0maky (seed 0), ss13djab (seed 1), nqkx8w0s (seed 2)
###########################################################################
echo "====================================================================="
echo "  Checkpoint Ablation: DPPO Learned Noise (3 runs)"
echo "====================================================================="

for SEED in 0 1 2; do
  run_cmd torchrun --nproc_per_node="$NUM_GPUS" finetune_online_rl.py \
    "${SHARED_ARGS[@]}" \
    --experiment "finetune-dppo-learned-can-ckpt-ablation" \
    --loss_mode dppo \
    --sde_sigma 0.18 \
    --learn_sde_sigma True \
    --noise_injection_min 0.2 \
    --noise_injection_max 0.5 \
    --cfm_loss_average_group_size 1 \
    --cfm_loss_use_huber True \
    --cfm_loss_huber_delta 0.5 \
    --clip_coef 0.01 \
    --max_grad_norm 1 \
    --clamp_logratio 5 \
    --clamp_old_cfm_loss 4 \
    --trust_region_mode ppo \
    --seed "$SEED"
done

###########################################################################
# DPPO Fixed Noise (3 runs)
# Run IDs: 4iz08o2h (seed 0), 0wo72noe (seed 1), 0yibse8t (seed 2)
###########################################################################
echo "====================================================================="
echo "  Checkpoint Ablation: DPPO Fixed Noise (3 runs)"
echo "====================================================================="

for SEED in 0 1 2; do
  run_cmd torchrun --nproc_per_node="$NUM_GPUS" finetune_online_rl.py \
    "${SHARED_ARGS[@]}" \
    --experiment "finetune-dppo-fixed-can-ckpt-ablation" \
    --loss_mode dppo \
    --sde_sigma 0.3 \
    --cfm_loss_average_group_size 1 \
    --cfm_loss_use_huber True \
    --cfm_loss_huber_delta 0.5 \
    --clip_coef 0.01 \
    --max_grad_norm 5 \
    --clamp_logratio 5 \
    --clamp_old_cfm_loss 4 \
    --trust_region_mode ppo \
    --seed "$SEED"
done

echo "====================================================================="
echo "  Checkpoint ablation complete (12 runs total)"
echo "====================================================================="
