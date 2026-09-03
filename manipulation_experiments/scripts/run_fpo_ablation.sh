#!/usr/bin/env bash
#
# run_fpo_ablation.sh
#
# Launches all 18 FPO ablation runs:
#   2 tasks (Square, Threading) x 3 models (FPO++, ASPO, Per-action ratio) x 3 seeds
#
# This ablation compares FPO++ against ASPO and per-action ratio variants
# to understand the contribution of each FPO++ design choice.
#
# Usage:
#   bash scripts/run_fpo_ablation.sh              # execute all runs
#   DRY_RUN=1 bash scripts/run_fpo_ablation.sh    # print commands only
#   NUM_GPUS=4 bash scripts/run_fpo_ablation.sh   # use 4 GPUs per run

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
# Shared parameters for ALL FPO ablation runs
# ---------------------------------------------------------------------------
SHARED_ARGS=(
  --distributed True
  --base-policy-wandb-project flow-bc
  --load-ema True
  --wandb_project flow-bc-fpo-finetuning
  --wandb_enable True
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
  --sde_sigma 0
  --cfm_loss_use_huber True
)

###########################################################################
# SQUARE TASK
# Base: trc7rbt0, step_110000, total_timesteps=8000000, discount=0.995
# FPO++ Square uses: clip_coef=0.01, max_grad_norm=25, cfm_loss_huber_delta=1
###########################################################################

# Square task-specific shared args
SQUARE_ARGS=(
  --base_policy_wandb_run_id trc7rbt0
  --checkpoint_step step_110000
  --total_timesteps 8000000
  --task Square
  --eval_env Square
  --discount 0.995
  --clip_coef 0.01
  --max_grad_norm 25
  --cfm_loss_huber_delta 1
)

# ------- Square: FPO++ (3 runs) -------
# Run IDs: z2u9ryms (seed 0), wzyv707a (seed 1), rsmunbo4 (seed 2)
echo "====================================================================="
echo "  FPO Ablation: Square - FPO++ (3 runs)"
echo "====================================================================="

for SEED in 0 1 2; do
  run_cmd torchrun --nproc_per_node="$NUM_GPUS" finetune_online_rl.py \
    "${SHARED_ARGS[@]}" \
    "${SQUARE_ARGS[@]}" \
    --experiment "finetune-fpo++-square-ablation" \
    --trust_region_mode ppo \
    --cfm_loss_average_group_size 1 \
    --clamp_logratio None \
    --clamp_old_cfm_loss None \
    --seed "$SEED"
done

# ------- Square: ASPO (3 runs) -------
# Run IDs: whhhg3oc (seed 0), safs53cm (seed 1), 069ligh5 (seed 2)
echo "====================================================================="
echo "  FPO Ablation: Square - ASPO (3 runs)"
echo "====================================================================="

for SEED in 0 1 2; do
  run_cmd torchrun --nproc_per_node="$NUM_GPUS" finetune_online_rl.py \
    "${SHARED_ARGS[@]}" \
    "${SQUARE_ARGS[@]}" \
    --experiment "finetune-aspo-square-ablation" \
    --trust_region_mode aspo \
    --cfm_loss_average_group_size 1 \
    --clamp_logratio 5 \
    --clamp_old_cfm_loss 4 \
    --seed "$SEED"
done

# ------- Square: Per-action ratio (3 runs) -------
# Run IDs: eikmrvae (seed 0), kf0ywqzj (seed 1), 4x5cxo0w (seed 2)
echo "====================================================================="
echo "  FPO Ablation: Square - Per-action ratio (3 runs)"
echo "====================================================================="

for SEED in 0 1 2; do
  run_cmd torchrun --nproc_per_node="$NUM_GPUS" finetune_online_rl.py \
    "${SHARED_ARGS[@]}" \
    "${SQUARE_ARGS[@]}" \
    --experiment "finetune-per-action-square-ablation" \
    --trust_region_mode ppo \
    --cfm_loss_average_group_size -1 \
    --clamp_logratio 5 \
    --clamp_old_cfm_loss 4 \
    --seed "$SEED"
done

###########################################################################
# THREADING TASK
# Base: 6vqrn614, step_10000, total_timesteps=8000000, discount=0.999
# FPO++ Threading uses: clip_coef=0.01, max_grad_norm=1, cfm_loss_huber_delta=0.1
###########################################################################

# Threading task-specific shared args
THREADING_ARGS=(
  --base_policy_wandb_run_id 6vqrn614
  --checkpoint_step step_10000
  --total_timesteps 8000000
  --task TwoArmThreading
  --eval_env TwoArmThreading
  --discount 0.999
  --clip_coef 0.01
  --max_grad_norm 1
  --cfm_loss_huber_delta 0.1
)

# ------- Threading: FPO++ (3 runs) -------
# Run IDs: bt3sl4ex (seed 0), fu8wpmbd (seed 1), g2ldgoss (seed 2)
echo "====================================================================="
echo "  FPO Ablation: Threading - FPO++ (3 runs)"
echo "====================================================================="

for SEED in 0 1 2; do
  run_cmd torchrun --nproc_per_node="$NUM_GPUS" finetune_online_rl.py \
    "${SHARED_ARGS[@]}" \
    "${THREADING_ARGS[@]}" \
    --experiment "finetune-fpo++-threading-ablation" \
    --trust_region_mode ppo \
    --cfm_loss_average_group_size 1 \
    --clamp_logratio 5 \
    --clamp_old_cfm_loss 4 \
    --seed "$SEED"
done

# ------- Threading: ASPO (3 runs) -------
# Run IDs: 4pwr3dzu (seed 0), ir471vi8 (seed 1), nx0ktwru (seed 2)
echo "====================================================================="
echo "  FPO Ablation: Threading - ASPO (3 runs)"
echo "====================================================================="

for SEED in 0 1 2; do
  run_cmd torchrun --nproc_per_node="$NUM_GPUS" finetune_online_rl.py \
    "${SHARED_ARGS[@]}" \
    "${THREADING_ARGS[@]}" \
    --experiment "finetune-aspo-threading-ablation" \
    --trust_region_mode aspo \
    --cfm_loss_average_group_size 1 \
    --clamp_logratio 5 \
    --clamp_old_cfm_loss 4 \
    --seed "$SEED"
done

# ------- Threading: Per-action ratio (3 runs) -------
# Run IDs: l5hdijzn (seed 0), pcu57hf1 (seed 1), xmklzaca (seed 2)
echo "====================================================================="
echo "  FPO Ablation: Threading - Per-action ratio (3 runs)"
echo "====================================================================="

for SEED in 0 1 2; do
  run_cmd torchrun --nproc_per_node="$NUM_GPUS" finetune_online_rl.py \
    "${SHARED_ARGS[@]}" \
    "${THREADING_ARGS[@]}" \
    --experiment "finetune-per-action-threading-ablation" \
    --trust_region_mode ppo \
    --cfm_loss_average_group_size -1 \
    --clamp_logratio 5 \
    --clamp_old_cfm_loss 4 \
    --seed "$SEED"
done

echo "====================================================================="
echo "  FPO ablation complete (18 runs total)"
echo "====================================================================="
