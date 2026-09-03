#!/bin/bash
#
# launch_checkpoint_ablation.sh (skypilot - remote cluster launcher)
#
# Launches all 12 checkpoint ablation training jobs on EC2 via SkyPilot:
#   Can task, base policy step_6000, 4 models x 3 seeds
#
# Each run gets its own EC2 cluster with 1x L40S GPU.
#
# Usage:
#   bash scripts/skypilot/launch_checkpoint_ablation.sh
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
# Shared base command (Can task, step_6000)
# ---------------------------------------------------------------------------
SHARED="torchrun --nproc_per_node=1 finetune_online_rl.py \
  --distributed True \
  --base-policy-wandb-project flow-bc \
  --base_policy_wandb_run_id 95j3noe4 \
  --load-ema True \
  --checkpoint_step step_6000 \
  --wandb_project flow-bc-fpo-finetuning \
  --wandb_enable True \
  --total_timesteps 5000000 \
  --task Can --eval_env Can \
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
  --discount=0.99"

###########################################################################
echo "=== FPO++ (3 runs) ==="
# Run IDs: txf5crib, ylxhw9uu, 6zp46k32
###########################################################################

for SEED in 0 1 2; do
    launch_job "$USER-ckpt-ablation-fpopp-seed${SEED}" \
        "$SHARED \
        --experiment finetune-fpo++-can-mar24-v1 \
        --cfm_loss_average_group_size=1 --cfm_loss_huber_delta=0.5 --cfm_loss_use_huber=True \
        --clamp_logratio=5 --clamp_old_cfm_loss=4 --clip_coef=0.02 \
        --max_grad_norm=5 --sde_sigma=0 --seed=$SEED \
        --trust_region_mode=ppo"
done

###########################################################################
echo "=== Vanilla FPO (3 runs) ==="
# Run IDs: gzm22fqh, opygcs30, 89jg66ll
###########################################################################

for SEED in 0 1 2; do
    launch_job "$USER-ckpt-ablation-vanilla-fpo-seed${SEED}" \
        "$SHARED \
        --experiment finetune-vanilla-fpo-can-mar24-v1 \
        --cfm_loss_average_group_size=-1 --cfm_loss_huber_delta=0.5 --cfm_loss_use_huber=False \
        --clamp_logratio=None --clamp_old_cfm_loss=None --clip_coef=0.02 \
        --max_grad_norm=5 --sde_sigma=0 --seed=$SEED \
        --trust_region_mode=ppo"
done

###########################################################################
echo "=== DPPO Learned Noise (3 runs) ==="
# Run IDs: 7lz0maky, ss13djab, nqkx8w0s
###########################################################################

for SEED in 0 1 2; do
    launch_job "$USER-ckpt-ablation-dppo-learned-seed${SEED}" \
        "$SHARED \
        --experiment finetune-dppo-can-mar24-v1 \
        --cfm_loss_average_group_size=1 --cfm_loss_huber_delta=0.5 --cfm_loss_use_huber=True \
        --clamp_logratio=5 --clamp_old_cfm_loss=4 --clip_coef=0.01 \
        --max_grad_norm=1 --loss_mode=dppo --sde_sigma=0.18 \
        --learn_sde_sigma=True --noise_injection_min=0.2 --noise_injection_max=0.5 \
        --seed=$SEED --trust_region_mode=ppo"
done

###########################################################################
echo "=== DPPO Fixed Noise (3 runs) ==="
# Run IDs: 4iz08o2h, 0wo72noe, 0yibse8t
###########################################################################

for SEED in 0 1 2; do
    launch_job "$USER-ckpt-ablation-dppo-fixed-seed${SEED}" \
        "$SHARED \
        --experiment finetune-dppo-can-mar24-v1 \
        --cfm_loss_average_group_size=1 --cfm_loss_huber_delta=0.5 --cfm_loss_use_huber=True \
        --clamp_logratio=5 --clamp_old_cfm_loss=4 --clip_coef=0.01 \
        --max_grad_norm=5 --loss_mode=dppo --sde_sigma=0.3 \
        --seed=$SEED --trust_region_mode=ppo"
done

wait
echo "=== All 12 checkpoint ablation launches complete ==="
