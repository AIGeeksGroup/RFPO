#!/bin/bash
#
# launch_fpo_ablation.sh (skypilot - remote cluster launcher)
#
# Launches all 18 FPO ablation training jobs on EC2 via SkyPilot:
#   2 tasks (Square, Threading) x 3 models (FPO++, ASPO, Per-action ratio) x 3 seeds
#
# Each run gets its own EC2 cluster with 1x L40S GPU.
#
# Usage:
#   bash scripts/skypilot/launch_fpo_ablation.sh
#
# Requires: SkyPilot configured with FAR-skypilot-wrapper

set -euo pipefail

SLEEP_BETWEEN=20

LAUNCH_COUNT=0

# ---------------------------------------------------------------------------
# Helper: launch a sky job
# ---------------------------------------------------------------------------
launch_job() {
    local cluster_name=$1
    local entrypoint=$2

    echo "[$LAUNCH_COUNT] Launching cluster: $cluster_name"

    SKIP_UNTRACKED=1 sky EC2:manipulation-fpo \
        --cluster "$cluster_name" \
        --num-nodes 1 \
        --gpus=L40S:1 --cpus 16+ --memory 64+ \
        --down \
        --idle-minutes-to-autostop 60 \
        --yes \
        --detach-run \
        --env OMNI_KIT_ACCEPT_EULA=YES \
        --env HF_HUB_DOWNLOAD_TIMEOUT=240 \
        --env HF_HUB_ETAG_TIMEOUT=240 \
        --env HF_TOKEN= \
        --env ENTRYPOINT="$entrypoint" \
        --async

    LAUNCH_COUNT=$((LAUNCH_COUNT + 1))
    sleep $SLEEP_BETWEEN
}

# ---------------------------------------------------------------------------
# Shared base command fragments
# ---------------------------------------------------------------------------
SHARED="torchrun --nproc_per_node=1 finetune_online_rl.py \
  --distributed True \
  --base-policy-wandb-project flow-bc \
  --load-ema True \
  --wandb_project flow-bc-fpo-finetuning \
  --wandb_enable True \
  --gradient_accumulation_steps 1 \
  --num_minibatches 8 \
  --log_freq 1 --save_freq 2 --rollout_freq 2 \
  --eval_num_episodes 200 \
  --data_collection_steps=1600 \
  --do_chunk_level_ppo=True \
  --eval_ema=False \
  --exploration_noise_std=None \
  --freeze_vision_encoder=True \
  --gae_lambda=0.99 \
  --n_action_samples=8 \
  --n_action_steps=16 \
  --num_envs=30 \
  --sampling_steps=10 \
  --spo_clip_coef=0.01 \
  --zero_sampling=True \
  --sde_sigma=0 \
  --cfm_loss_use_huber=True"

###########################################################################
echo "=== Square Ablation (9 runs) ==="
# Base: trc7rbt0, step_110000, total_timesteps=8000000, discount=0.995
# clip_coef=0.01, max_grad_norm=25, cfm_loss_huber_delta=1
###########################################################################

SQUARE_SHARED="$SHARED \
  --base_policy_wandb_run_id trc7rbt0 --checkpoint_step step_110000 \
  --total_timesteps 8000000 --task Square --eval_env Square --discount=0.995 \
  --clip_coef=0.01 --max_grad_norm=25 --cfm_loss_huber_delta=1"

# --- Square: FPO++ ---
# Run IDs: z2u9ryms, wzyv707a, rsmunbo4
echo "  Square - FPO++ (3 runs)"
for SEED in 0 1 2; do
    launch_job "$USER-ablation-fpopp-square-seed${SEED}" \
        "$SQUARE_SHARED \
        --experiment finetune-fpo++-ablation-square-mar24-v1 \
        --trust_region_mode=ppo \
        --cfm_loss_average_group_size=1 \
        --clamp_logratio=None --clamp_old_cfm_loss=None \
        --seed=$SEED"
done

# --- Square: ASPO ---
# Run IDs: whhhg3oc, safs53cm, 069ligh5
echo "  Square - ASPO (3 runs)"
for SEED in 0 1 2; do
    launch_job "$USER-ablation-aspo-square-seed${SEED}" \
        "$SQUARE_SHARED \
        --experiment finetune-fpo++-aspo-ablation-square-mar24-v1 \
        --trust_region_mode=aspo \
        --cfm_loss_average_group_size=1 \
        --clamp_logratio=5 --clamp_old_cfm_loss=4 \
        --seed=$SEED"
done

# --- Square: Per-action ratio ---
# Run IDs: eikmrvae, kf0ywqzj, 4x5cxo0w
echo "  Square - Per-action ratio (3 runs)"
for SEED in 0 1 2; do
    launch_job "$USER-ablation-peraction-square-seed${SEED}" \
        "$SQUARE_SHARED \
        --experiment finetune-fpo++-peraction-ablation-square-mar24-v1 \
        --trust_region_mode=ppo \
        --cfm_loss_average_group_size=-1 \
        --clamp_logratio=5 --clamp_old_cfm_loss=4 \
        --seed=$SEED"
done

###########################################################################
echo "=== Threading Ablation (9 runs) ==="
# Base: 6vqrn614, step_10000, total_timesteps=8000000, discount=0.999
# clip_coef=0.01, max_grad_norm=1, cfm_loss_huber_delta=0.1
###########################################################################

THREADING_SHARED="$SHARED \
  --base_policy_wandb_run_id 6vqrn614 --checkpoint_step step_10000 \
  --total_timesteps 8000000 --task TwoArmThreading --eval_env TwoArmThreading --discount=0.999 \
  --clip_coef=0.01 --max_grad_norm=1 --cfm_loss_huber_delta=0.1"

# --- Threading: FPO++ ---
# Run IDs: bt3sl4ex, fu8wpmbd, g2ldgoss
echo "  Threading - FPO++ (3 runs)"
for SEED in 0 1 2; do
    launch_job "$USER-ablation-fpopp-threading-seed${SEED}" \
        "$THREADING_SHARED \
        --experiment finetune-fpo++-ablation-threading-mar24-v1 \
        --trust_region_mode=ppo \
        --cfm_loss_average_group_size=1 \
        --clamp_logratio=5 --clamp_old_cfm_loss=4 \
        --seed=$SEED"
done

# --- Threading: ASPO ---
# Run IDs: 4pwr3dzu, ir471vi8, nx0ktwru
echo "  Threading - ASPO (3 runs)"
for SEED in 0 1 2; do
    launch_job "$USER-ablation-aspo-threading-seed${SEED}" \
        "$THREADING_SHARED \
        --experiment finetune-fpo++-aspo-ablation-threading-mar24-v1 \
        --trust_region_mode=aspo \
        --cfm_loss_average_group_size=1 \
        --clamp_logratio=5 --clamp_old_cfm_loss=4 \
        --seed=$SEED"
done

# --- Threading: Per-action ratio ---
# Run IDs: l5hdijzn, pcu57hf1, xmklzaca
echo "  Threading - Per-action ratio (3 runs)"
for SEED in 0 1 2; do
    launch_job "$USER-ablation-peraction-threading-seed${SEED}" \
        "$THREADING_SHARED \
        --experiment finetune-fpo++-peraction-ablation-threading-mar24-v1 \
        --trust_region_mode=ppo \
        --cfm_loss_average_group_size=-1 \
        --clamp_logratio=5 --clamp_old_cfm_loss=4 \
        --seed=$SEED"
done

wait
echo "=== All 18 FPO ablation launches complete ==="
