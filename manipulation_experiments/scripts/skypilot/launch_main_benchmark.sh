#!/bin/bash
#
# launch_main_benchmark.sh (skypilot - remote cluster launcher)
#
# Launches all 60 main benchmark training jobs on EC2 via SkyPilot:
#   5 tasks x 4 models (FPO++, DPPO Learned, Vanilla FPO, DPPO Fixed) x 3 seeds
#
# Each run gets its own EC2 cluster with 1x L40S GPU.
#
# Usage:
#   bash scripts/skypilot/launch_main_benchmark.sh
#
# Requires: SkyPilot configured with FAR-skypilot-wrapper

set -euo pipefail

SLEEP_BETWEEN=20    # seconds between launches
SLEEP_BATCH=120     # extra sleep every BATCH_SIZE launches
BATCH_SIZE=32
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

    if [ $((LAUNCH_COUNT % BATCH_SIZE)) -eq 0 ] && [ $LAUNCH_COUNT -ne 0 ]; then
        echo "Sleeping ${SLEEP_BATCH}s after $LAUNCH_COUNT launches..."
        sleep $SLEEP_BATCH
    fi
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
  --zero_sampling=True"

###########################################################################
echo "=== FPO++ (15 runs) ==="
###########################################################################

for SEED in 0 1 2; do
    # Can
    launch_job "$USER-fpopp-can-seed${SEED}" \
        "$SHARED \
        --base_policy_wandb_run_id 95j3noe4 --checkpoint_step step_1000 \
        --experiment finetune-fpo++-can-mar24-v1 \
        --total_timesteps 5000000 --task Can --eval_env Can --discount=0.99 \
        --cfm_loss_average_group_size=1 --cfm_loss_huber_delta=0.5 --cfm_loss_use_huber=True \
        --clamp_logratio=5 --clamp_old_cfm_loss=4 --clip_coef=0.02 \
        --max_grad_norm=5 --sde_sigma=0 --seed=$SEED \
        --trust_region_mode=ppo"

    # Square
    launch_job "$USER-fpopp-square-seed${SEED}" \
        "$SHARED \
        --base_policy_wandb_run_id trc7rbt0 --checkpoint_step step_110000 \
        --experiment finetune-fpo++-square-mar24-v1 \
        --total_timesteps 8000000 --task Square --eval_env Square --discount=0.995 \
        --cfm_loss_average_group_size=1 --cfm_loss_huber_delta=1 --cfm_loss_use_huber=True \
        --clamp_logratio=None --clamp_old_cfm_loss=None --clip_coef=0.01 \
        --max_grad_norm=25 --sde_sigma=0 --seed=$SEED \
        --trust_region_mode=ppo"

    # Box Clearance
    launch_job "$USER-fpopp-box-seed${SEED}" \
        "$SHARED \
        --base_policy_wandb_run_id lainyisy --checkpoint_step step_10000 \
        --experiment finetune-fpo++-boxcleaning-mar24-v1 \
        --total_timesteps 5000000 --task TwoArmBoxCleanup --eval_env TwoArmBoxCleanup --discount=0.995 \
        --cfm_loss_average_group_size=1 --cfm_loss_huber_delta=0.1 --cfm_loss_use_huber=True \
        --clamp_logratio=5 --clamp_old_cfm_loss=4 --clip_coef=0.03 \
        --max_grad_norm=5 --sde_sigma=0 --seed=$SEED \
        --trust_region_mode=ppo"

    # Tray Lifting
    launch_job "$USER-fpopp-tray-seed${SEED}" \
        "$SHARED \
        --base_policy_wandb_run_id ri0w9j39 --checkpoint_step step_20000 \
        --experiment finetune-fpo++-tray-mar24-v1 \
        --total_timesteps 8000000 --task TwoArmLiftTray --eval_env TwoArmLiftTray --discount=0.999 \
        --cfm_loss_average_group_size=1 --cfm_loss_huber_delta=1 --cfm_loss_use_huber=True \
        --clamp_logratio=5 --clamp_old_cfm_loss=4 --clip_coef=0.03 \
        --max_grad_norm=1 --sde_sigma=0 --seed=$SEED \
        --trust_region_mode=ppo"

    # Threading
    launch_job "$USER-fpopp-threading-seed${SEED}" \
        "$SHARED \
        --base_policy_wandb_run_id 6vqrn614 --checkpoint_step step_10000 \
        --experiment finetune-fpo++-threading-mar24-v1 \
        --total_timesteps 8000000 --task TwoArmThreading --eval_env TwoArmThreading --discount=0.999 \
        --cfm_loss_average_group_size=1 --cfm_loss_huber_delta=0.1 --cfm_loss_use_huber=True \
        --clamp_logratio=5 --clamp_old_cfm_loss=4 --clip_coef=0.01 \
        --max_grad_norm=1 --sde_sigma=0 --seed=$SEED \
        --trust_region_mode=ppo"
done

###########################################################################
echo "=== DPPO Learned Noise (15 runs) ==="
###########################################################################

for SEED in 0 1 2; do
    # Can
    launch_job "$USER-dppo-learned-can-seed${SEED}" \
        "$SHARED \
        --base_policy_wandb_run_id 95j3noe4 --checkpoint_step step_1000 \
        --experiment finetune-dppo-learned-can-mar24-v1 \
        --total_timesteps 5000000 --task Can --eval_env Can --discount=0.99 \
        --cfm_loss_average_group_size=1 --cfm_loss_huber_delta=0.5 --cfm_loss_use_huber=True \
        --clamp_logratio=5 --clamp_old_cfm_loss=4 --clip_coef=0.01 \
        --max_grad_norm=25 --loss_mode=dppo --sde_sigma=0.18 \
        --learn_sde_sigma=True --noise_injection_min=0.3 --noise_injection_max=0.5 \
        --seed=$SEED --trust_region_mode=ppo"

    # Square
    launch_job "$USER-dppo-learned-square-seed${SEED}" \
        "$SHARED \
        --base_policy_wandb_run_id trc7rbt0 --checkpoint_step step_110000 \
        --experiment finetune-dppo-learned-square-mar24-v1 \
        --total_timesteps 8000000 --task Square --eval_env Square --discount=0.995 \
        --cfm_loss_average_group_size=1 --cfm_loss_huber_delta=0.1 --cfm_loss_use_huber=True \
        --clamp_logratio=5 --clamp_old_cfm_loss=4 --clip_coef=0.01 \
        --max_grad_norm=5 --loss_mode=dppo --sde_sigma=0.18 \
        --learn_sde_sigma=True --noise_injection_min=0.3 --noise_injection_max=0.5 \
        --seed=$SEED --trust_region_mode=ppo"

    # Box Clearance
    launch_job "$USER-dppo-learned-box-seed${SEED}" \
        "$SHARED \
        --base_policy_wandb_run_id lainyisy --checkpoint_step step_10000 \
        --experiment finetune-dppo-learned-boxcleaning-mar24-v1 \
        --total_timesteps 5000000 --task TwoArmBoxCleanup --eval_env TwoArmBoxCleanup --discount=0.995 \
        --cfm_loss_average_group_size=1 --cfm_loss_huber_delta=0.1 --cfm_loss_use_huber=True \
        --clamp_logratio=5 --clamp_old_cfm_loss=4 --clip_coef=0.03 \
        --max_grad_norm=1 --loss_mode=dppo --sde_sigma=0.18 \
        --learn_sde_sigma=True --noise_injection_min=0.2 --noise_injection_max=0.5 \
        --seed=$SEED --trust_region_mode=ppo"

    # Tray Lifting
    launch_job "$USER-dppo-learned-tray-seed${SEED}" \
        "$SHARED \
        --base_policy_wandb_run_id ri0w9j39 --checkpoint_step step_20000 \
        --experiment finetune-dppo-learned-tray-mar24-v1 \
        --total_timesteps 8000000 --task TwoArmLiftTray --eval_env TwoArmLiftTray --discount=0.999 \
        --cfm_loss_average_group_size=1 --cfm_loss_huber_delta=0.1 --cfm_loss_use_huber=True \
        --clamp_logratio=5 --clamp_old_cfm_loss=4 --clip_coef=0.01 \
        --max_grad_norm=25 --loss_mode=dppo --sde_sigma=0.18 \
        --learn_sde_sigma=True --noise_injection_min=0.3 --noise_injection_max=0.5 \
        --seed=$SEED --trust_region_mode=ppo"

    # Threading
    launch_job "$USER-dppo-learned-threading-seed${SEED}" \
        "$SHARED \
        --base_policy_wandb_run_id 6vqrn614 --checkpoint_step step_10000 \
        --experiment finetune-dppo-learned-threading-mar24-v1 \
        --total_timesteps 8000000 --task TwoArmThreading --eval_env TwoArmThreading --discount=0.999 \
        --cfm_loss_average_group_size=1 --cfm_loss_huber_delta=0.1 --cfm_loss_use_huber=True \
        --clamp_logratio=5 --clamp_old_cfm_loss=4 --clip_coef=0.01 \
        --max_grad_norm=1 --loss_mode=dppo --sde_sigma=0.18 \
        --learn_sde_sigma=True --noise_injection_min=0.3 --noise_injection_max=0.5 \
        --seed=$SEED --trust_region_mode=ppo"
done

###########################################################################
echo "=== Vanilla FPO (15 runs) ==="
###########################################################################

for SEED in 0 1 2; do
    # Can
    launch_job "$USER-vanilla-fpo-can-seed${SEED}" \
        "$SHARED \
        --base_policy_wandb_run_id 95j3noe4 --checkpoint_step step_1000 \
        --experiment finetune-vanilla-fpo-can-mar24-v1 \
        --total_timesteps 5000000 --task Can --eval_env Can --discount=0.99 \
        --cfm_loss_average_group_size=-1 --cfm_loss_huber_delta=0.5 --cfm_loss_use_huber=False \
        --clamp_logratio=None --clamp_old_cfm_loss=None --clip_coef=0.01 \
        --max_grad_norm=25 --sde_sigma=0 --seed=$SEED \
        --trust_region_mode=ppo"

    # Square
    launch_job "$USER-vanilla-fpo-square-seed${SEED}" \
        "$SHARED \
        --base_policy_wandb_run_id trc7rbt0 --checkpoint_step step_110000 \
        --experiment finetune-vanilla-fpo-square-mar24-v1 \
        --total_timesteps 8000000 --task Square --eval_env Square --discount=0.995 \
        --cfm_loss_average_group_size=-1 --cfm_loss_huber_delta=1 --cfm_loss_use_huber=False \
        --clamp_logratio=None --clamp_old_cfm_loss=None --clip_coef=0.01 \
        --max_grad_norm=25 --sde_sigma=0 --seed=$SEED \
        --trust_region_mode=ppo"

    # Box Clearance
    launch_job "$USER-vanilla-fpo-box-seed${SEED}" \
        "$SHARED \
        --base_policy_wandb_run_id lainyisy --checkpoint_step step_10000 \
        --experiment finetune-vanilla-fpo-boxcleaning-mar24-v1 \
        --total_timesteps 5000000 --task TwoArmBoxCleanup --eval_env TwoArmBoxCleanup --discount=0.995 \
        --cfm_loss_average_group_size=-1 --cfm_loss_huber_delta=0.5 --cfm_loss_use_huber=False \
        --clamp_logratio=None --clamp_old_cfm_loss=None --clip_coef=0.03 \
        --max_grad_norm=5 --sde_sigma=0 --seed=$SEED \
        --trust_region_mode=ppo"

    # Tray Lifting
    launch_job "$USER-vanilla-fpo-tray-seed${SEED}" \
        "$SHARED \
        --base_policy_wandb_run_id ri0w9j39 --checkpoint_step step_20000 \
        --experiment finetune-vanilla-fpo-tray-mar24-v1 \
        --total_timesteps 8000000 --task TwoArmLiftTray --eval_env TwoArmLiftTray --discount=0.999 \
        --cfm_loss_average_group_size=-1 --cfm_loss_huber_delta=0.5 --cfm_loss_use_huber=False \
        --clamp_logratio=None --clamp_old_cfm_loss=None --clip_coef=0.01 \
        --max_grad_norm=25 --sde_sigma=0 --seed=$SEED \
        --trust_region_mode=ppo"

    # Threading
    launch_job "$USER-vanilla-fpo-threading-seed${SEED}" \
        "$SHARED \
        --base_policy_wandb_run_id 6vqrn614 --checkpoint_step step_10000 \
        --experiment finetune-vanilla-fpo-threading-mar24-v1 \
        --total_timesteps 8000000 --task TwoArmThreading --eval_env TwoArmThreading --discount=0.999 \
        --cfm_loss_average_group_size=-1 --cfm_loss_huber_delta=0.5 --cfm_loss_use_huber=False \
        --clamp_logratio=None --clamp_old_cfm_loss=None --clip_coef=0.01 \
        --max_grad_norm=25 --sde_sigma=0 --seed=$SEED \
        --trust_region_mode=ppo"
done

###########################################################################
echo "=== DPPO Fixed Noise (15 runs) ==="
###########################################################################

for SEED in 0 1 2; do
    # Can
    launch_job "$USER-dppo-fixed-can-seed${SEED}" \
        "$SHARED \
        --base_policy_wandb_run_id 95j3noe4 --checkpoint_step step_1000 \
        --experiment finetune-dppo-fixed-can-mar24-v1 \
        --total_timesteps 5000000 --task Can --eval_env Can --discount=0.99 \
        --cfm_loss_average_group_size=1 --cfm_loss_huber_delta=0.5 --cfm_loss_use_huber=True \
        --clamp_logratio=5 --clamp_old_cfm_loss=4 --clip_coef=0.01 \
        --learn_sde_sigma=False --max_grad_norm=25 --loss_mode=dppo --sde_sigma=0.3 \
        --seed=$SEED --trust_region_mode=ppo"

    # Square
    launch_job "$USER-dppo-fixed-square-seed${SEED}" \
        "$SHARED \
        --base_policy_wandb_run_id trc7rbt0 --checkpoint_step step_110000 \
        --experiment finetune-dppo-fixed-square-mar24-v1 \
        --total_timesteps 8000000 --task Square --eval_env Square --discount=0.995 \
        --cfm_loss_average_group_size=1 --cfm_loss_huber_delta=0.1 --cfm_loss_use_huber=True \
        --clamp_logratio=5 --clamp_old_cfm_loss=4 --clip_coef=0.01 \
        --learn_sde_sigma=False --max_grad_norm=25 --loss_mode=dppo --sde_sigma=0.3 \
        --seed=$SEED --trust_region_mode=ppo"

    # Box Clearance
    launch_job "$USER-dppo-fixed-box-seed${SEED}" \
        "$SHARED \
        --base_policy_wandb_run_id lainyisy --checkpoint_step step_10000 \
        --experiment finetune-dppo-fixed-boxcleaning-mar24-v1 \
        --total_timesteps 5000000 --task TwoArmBoxCleanup --eval_env TwoArmBoxCleanup --discount=0.995 \
        --cfm_loss_average_group_size=1 --cfm_loss_huber_delta=0.1 --cfm_loss_use_huber=True \
        --clamp_logratio=5 --clamp_old_cfm_loss=4 --clip_coef=0.02 \
        --learn_sde_sigma=False --max_grad_norm=25 --loss_mode=dppo --sde_sigma=0.24 \
        --seed=$SEED --trust_region_mode=ppo"

    # Tray Lifting
    launch_job "$USER-dppo-fixed-tray-seed${SEED}" \
        "$SHARED \
        --base_policy_wandb_run_id ri0w9j39 --checkpoint_step step_20000 \
        --experiment finetune-dppo-fixed-tray-mar24-v1 \
        --total_timesteps 8000000 --task TwoArmLiftTray --eval_env TwoArmLiftTray --discount=0.999 \
        --cfm_loss_average_group_size=1 --cfm_loss_huber_delta=0.1 --cfm_loss_use_huber=True \
        --clamp_logratio=5 --clamp_old_cfm_loss=4 --clip_coef=0.01 \
        --learn_sde_sigma=False --max_grad_norm=5 --loss_mode=dppo --sde_sigma=0.24 \
        --seed=$SEED --trust_region_mode=ppo"

    # Threading
    launch_job "$USER-dppo-fixed-threading-seed${SEED}" \
        "$SHARED \
        --base_policy_wandb_run_id 6vqrn614 --checkpoint_step step_10000 \
        --experiment finetune-dppo-fixed-threading-mar24-v1 \
        --total_timesteps 8000000 --task TwoArmThreading --eval_env TwoArmThreading --discount=0.999 \
        --cfm_loss_average_group_size=1 --cfm_loss_huber_delta=0.1 --cfm_loss_use_huber=True \
        --clamp_logratio=5 --clamp_old_cfm_loss=4 --clip_coef=0.02 \
        --learn_sde_sigma=False --max_grad_norm=5 --loss_mode=dppo --sde_sigma=0.24 \
        --seed=$SEED --trust_region_mode=ppo"
done

wait
echo "=== All 60 main benchmark launches complete ==="
