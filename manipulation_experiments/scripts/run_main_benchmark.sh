#!/usr/bin/env bash
#
# run_main_benchmark.sh
#
# Launches all 60 main benchmark runs:
#   5 tasks x 4 models (FPO++, DPPO Learned, Vanilla FPO, DPPO Fixed) x 3 seeds
#
# Usage:
#   bash scripts/run_main_benchmark.sh              # execute all runs
#   DRY_RUN=1 bash scripts/run_main_benchmark.sh    # print commands only
#   NUM_GPUS=4 bash scripts/run_main_benchmark.sh   # use 4 GPUs per run
#
# Each run is launched sequentially. To run in parallel, pipe DRY_RUN=1 output
# to a job scheduler or use & to background individual runs.

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
# Shared parameters for ALL 60 main benchmark runs
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
)

# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------
#              task_name       env_name             run_id    ckpt_step      total_ts  discount
TASKS=(
  "can        Can              95j3noe4  step_1000    5000000   0.99"
  "square     Square           trc7rbt0  step_110000  8000000   0.995"
  "box        TwoArmBoxCleanup lainyisy  step_10000   5000000   0.995"
  "tray       TwoArmLiftTray   ri0w9j39  step_20000   8000000   0.999"
  "threading  TwoArmThreading  6vqrn614  step_10000   8000000   0.999"
)

###########################################################################
# 1.1  FPO++
###########################################################################
echo "====================================================================="
echo "  FPO++ (15 runs)"
echo "====================================================================="

# Per-task FPO++ parameters:
#   task_key  clip_coef  max_grad_norm  cfm_huber_delta  clamp_logratio  clamp_old_cfm
FPOP_PARAMS=(
  "can        0.02  5   0.5  5     4"
  "square     0.01  25  1    None  None"
  "box        0.03  5   0.1  5     4"
  "tray       0.03  1   1    5     4"
  "threading  0.01  1   0.1  5     4"
)

for task_line in "${TASKS[@]}"; do
  read -r task_key env_name run_id ckpt_step total_ts disc <<< "$task_line"

  # Find FPO++ params for this task
  for fpop_line in "${FPOP_PARAMS[@]}"; do
    read -r fpop_key clip_coef mgn huber_delta clamp_lr clamp_old <<< "$fpop_line"
    if [[ "$fpop_key" == "$task_key" ]]; then
      for SEED in 0 1 2; do
        run_cmd torchrun --nproc_per_node="$NUM_GPUS" finetune_online_rl.py \
          "${SHARED_ARGS[@]}" \
          --base_policy_wandb_run_id "$run_id" \
          --checkpoint_step "$ckpt_step" \
          --experiment "finetune-fpo++-${task_key}" \
          --total_timesteps "$total_ts" \
          --task "$env_name" \
          --eval_env "$env_name" \
          --discount "$disc" \
          --sde_sigma 0 \
          --cfm_loss_average_group_size 1 \
          --cfm_loss_use_huber True \
          --cfm_loss_huber_delta "$huber_delta" \
          --clip_coef "$clip_coef" \
          --max_grad_norm "$mgn" \
          --clamp_logratio "$clamp_lr" \
          --clamp_old_cfm_loss "$clamp_old" \
          --trust_region_mode ppo \
          --seed "$SEED"
      done
      break
    fi
  done
done

###########################################################################
# 1.2  DPPO Learned Noise
###########################################################################
echo "====================================================================="
echo "  DPPO Learned Noise (15 runs)"
echo "====================================================================="

# Per-task DPPO Learned parameters:
#   task_key  clip_coef  max_grad_norm  cfm_huber_delta  noise_min  noise_max  learn_sde
DPPO_LEARNED_PARAMS=(
  "can        0.01  25  0.5  0.3  0.5  True"
  "square     0.01  5   0.1  0.3  0.5  True"
  "box        0.03  1   0.1  0.2  0.5  True"
  "tray       0.01  25  0.1  0.3  0.5  True"
  "threading  0.01  1   0.1  0.3  0.5  True"
)

for task_line in "${TASKS[@]}"; do
  read -r task_key env_name run_id ckpt_step total_ts disc <<< "$task_line"

  for dp_line in "${DPPO_LEARNED_PARAMS[@]}"; do
    read -r dp_key clip_coef mgn huber_delta noise_min noise_max learn_sde <<< "$dp_line"
    if [[ "$dp_key" == "$task_key" ]]; then
      for SEED in 0 1 2; do
        run_cmd torchrun --nproc_per_node="$NUM_GPUS" finetune_online_rl.py \
          "${SHARED_ARGS[@]}" \
          --base_policy_wandb_run_id "$run_id" \
          --checkpoint_step "$ckpt_step" \
          --experiment "finetune-dppo-learned-${task_key}" \
          --total_timesteps "$total_ts" \
          --task "$env_name" \
          --eval_env "$env_name" \
          --discount "$disc" \
          --loss_mode dppo \
          --sde_sigma 0.18 \
          --learn_sde_sigma "$learn_sde" \
          --noise_injection_min "$noise_min" \
          --noise_injection_max "$noise_max" \
          --cfm_loss_average_group_size 1 \
          --cfm_loss_use_huber True \
          --cfm_loss_huber_delta "$huber_delta" \
          --clip_coef "$clip_coef" \
          --max_grad_norm "$mgn" \
          --clamp_logratio 5 \
          --clamp_old_cfm_loss 4 \
          --trust_region_mode ppo \
          --seed "$SEED"
      done
      break
    fi
  done
done

###########################################################################
# 1.3  Vanilla FPO
###########################################################################
echo "====================================================================="
echo "  Vanilla FPO (15 runs)"
echo "====================================================================="

# Per-task Vanilla FPO parameters:
#   task_key  clip_coef  max_grad_norm  cfm_huber_delta
VANILLA_FPO_PARAMS=(
  "can        0.01  25  0.5"
  "square     0.01  25  1"
  "box        0.03  5   0.5"
  "tray       0.01  25  0.5"
  "threading  0.01  25  0.5"
)

for task_line in "${TASKS[@]}"; do
  read -r task_key env_name run_id ckpt_step total_ts disc <<< "$task_line"

  for vfpo_line in "${VANILLA_FPO_PARAMS[@]}"; do
    read -r vfpo_key clip_coef mgn huber_delta <<< "$vfpo_line"
    if [[ "$vfpo_key" == "$task_key" ]]; then
      for SEED in 0 1 2; do
        run_cmd torchrun --nproc_per_node="$NUM_GPUS" finetune_online_rl.py \
          "${SHARED_ARGS[@]}" \
          --base_policy_wandb_run_id "$run_id" \
          --checkpoint_step "$ckpt_step" \
          --experiment "finetune-vanilla-fpo-${task_key}" \
          --total_timesteps "$total_ts" \
          --task "$env_name" \
          --eval_env "$env_name" \
          --discount "$disc" \
          --sde_sigma 0 \
          --cfm_loss_average_group_size -1 \
          --cfm_loss_use_huber False \
          --cfm_loss_huber_delta "$huber_delta" \
          --clip_coef "$clip_coef" \
          --max_grad_norm "$mgn" \
          --clamp_logratio None \
          --clamp_old_cfm_loss None \
          --trust_region_mode ppo \
          --seed "$SEED"
      done
      break
    fi
  done
done

###########################################################################
# 1.4  DPPO Fixed Noise
###########################################################################
echo "====================================================================="
echo "  DPPO Fixed Noise (15 runs)"
echo "====================================================================="

# Per-task DPPO Fixed parameters:
#   task_key  clip_coef  max_grad_norm  cfm_huber_delta  sde_sigma
DPPO_FIXED_PARAMS=(
  "can        0.01  25  0.5  0.3"
  "square     0.01  25  0.1  0.3"
  "box        0.02  25  0.1  0.24"
  "tray       0.01  5   0.1  0.24"
  "threading  0.02  5   0.1  0.24"
)

for task_line in "${TASKS[@]}"; do
  read -r task_key env_name run_id ckpt_step total_ts disc <<< "$task_line"

  for dpf_line in "${DPPO_FIXED_PARAMS[@]}"; do
    read -r dpf_key clip_coef mgn huber_delta sde_sigma <<< "$dpf_line"
    if [[ "$dpf_key" == "$task_key" ]]; then
      for SEED in 0 1 2; do
        run_cmd torchrun --nproc_per_node="$NUM_GPUS" finetune_online_rl.py \
          "${SHARED_ARGS[@]}" \
          --base_policy_wandb_run_id "$run_id" \
          --checkpoint_step "$ckpt_step" \
          --experiment "finetune-dppo-fixed-${task_key}" \
          --total_timesteps "$total_ts" \
          --task "$env_name" \
          --eval_env "$env_name" \
          --discount "$disc" \
          --loss_mode dppo \
          --sde_sigma "$sde_sigma" \
          --learn_sde_sigma False \
          --cfm_loss_average_group_size 1 \
          --cfm_loss_use_huber True \
          --cfm_loss_huber_delta "$huber_delta" \
          --clip_coef "$clip_coef" \
          --max_grad_norm "$mgn" \
          --clamp_logratio 5 \
          --clamp_old_cfm_loss 4 \
          --trust_region_mode ppo \
          --seed "$SEED"
      done
      break
    fi
  done
done

echo "====================================================================="
echo "  Main benchmark complete (60 runs total)"
echo "====================================================================="
