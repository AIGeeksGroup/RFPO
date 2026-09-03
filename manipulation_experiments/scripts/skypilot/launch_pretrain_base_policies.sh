#!/bin/bash
#
# launch_pretrain_base_policies.sh (skypilot - remote cluster launcher)
#
# Launches all 5 base policy pretraining jobs on EC2 via SkyPilot.
# Each task gets its own EC2 cluster with 1x L40S GPU.
#
# Base policies: Flow Matching BC with velocity prediction (u/u), L2 loss,
# MLP [1024,1024,1024], ViT backbone, grad_clip_norm=25, seed=3.
#
# Usage:
#   bash scripts/skypilot/launch_pretrain_base_policies.sh
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
# Shared pretraining parameters
# ---------------------------------------------------------------------------
SHARED="python pretrain_flow_bc.py \
  --policy flowmatching \
  --network_architecture mlp \
  --mlp_dims [1024,1024,1024] \
  --vision_backbone vit \
  --flow_network_output_param u \
  --cfm_loss_mode u \
  --cfm_loss_use_huber False \
  --cfm_loss_huber_delta 0.5 \
  --grad_clip_norm 25 \
  --horizon 16 \
  --n_action_steps 8 \
  --sampling_steps 10 \
  --batch_size 512 \
  --learning_rate 1e-4 \
  --lr_backbone 1e-5 \
  --weight_decay 1e-6 \
  --ema_power 0.995 \
  --enable_geometric_augmentations True \
  --seed 3 \
  --num_workers 16 \
  --wandb_enable True \
  --wandb_project flow-bc \
  --eval_num_envs 25 \
  --eval_num_episodes 50 \
  --log_freq 5 \
  --save_freq 1000 \
  --rollout_freq 1000"

###########################################################################
echo "=== Can ==="
# Original run: 95j3noe4
###########################################################################
launch_job "$USER-pretrain-bc-can" \
    "$SHARED \
    --dataset ankile/robomimic-mh-can-image \
    --image_observation_keys robot0_eye_in_hand_image \
    --eval_env Can \
    --steps 500000 \
    --experiment flow_bc_can"

###########################################################################
echo "=== Square ==="
# Original run: trc7rbt0
###########################################################################
launch_job "$USER-pretrain-bc-square" \
    "$SHARED \
    --dataset ankile/robomimic-mh-square-image \
    --image_observation_keys agentview_image \
    --eval_env Square \
    --steps 1000000 \
    --max_num_episodes 100 \
    --experiment flow_bc_square"

###########################################################################
echo "=== Box Clearance (TwoArmBoxCleanup) ==="
# Original run: lainyisy
###########################################################################
launch_job "$USER-pretrain-bc-box" \
    "$SHARED \
    --dataset ankile/dexmg-two-arm-box-cleanup \
    --image_observation_keys 'agentview_image robot0_eye_in_hand_image robot1_eye_in_hand_image' \
    --eval_env TwoArmBoxCleanup \
    --steps 1000000 \
    --experiment flow_bc_boxcleaning"

###########################################################################
echo "=== Tray Lifting (TwoArmLiftTray) ==="
# Original run: ri0w9j39
###########################################################################
launch_job "$USER-pretrain-bc-tray" \
    "$SHARED \
    --dataset ankile/dexmg-two-arm-lift-tray \
    --image_observation_keys 'agentview_image robot0_eye_in_hand_image robot1_eye_in_hand_image' \
    --eval_env TwoArmLiftTray \
    --steps 1000000 \
    --experiment flow_bc_tray"

###########################################################################
echo "=== Threading (TwoArmThreading) ==="
# Original run: 6vqrn614
###########################################################################
launch_job "$USER-pretrain-bc-threading" \
    "$SHARED \
    --dataset ankile/dexmg-two-arm-threading \
    --image_observation_keys 'agentview_image robot0_eye_in_hand_image robot1_eye_in_hand_image' \
    --eval_env TwoArmThreading \
    --steps 1000000 \
    --experiment flow_bc_threading"

wait
echo "=== All 5 pretraining launches complete ==="
