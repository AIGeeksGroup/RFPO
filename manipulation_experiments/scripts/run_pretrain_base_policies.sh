#!/usr/bin/env bash
#
# run_pretrain_base_policies.sh
#
# Pretrains all 5 base policies (Flow Matching BC) used in FPO experiments.
# Each task produces a checkpoint that matches the base policy used in the
# main benchmark (W&B project: flow-bc).
#
# All base policies share these hyperparameters:
#   - network: MLP [1024,1024,1024] with ViT vision backbone
#   - flow_network_output_param=u, cfm_loss_mode=u (velocity prediction)
#   - cfm_loss_use_huber=False (L2 loss)
#   - grad_clip_norm=25
#   - horizon=16, n_action_steps=8, sampling_steps=10
#   - batch_size=512, learning_rate=1e-4, lr_backbone=1e-5
#   - ema_power=0.995, enable_geometric_augmentations=True
#   - seed=3
#
# Usage:
#   bash scripts/run_pretrain_base_policies.sh              # run all 5
#   DRY_RUN=1 bash scripts/run_pretrain_base_policies.sh    # print commands only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

DRY_RUN="${DRY_RUN:-0}"

cd "$REPO_DIR"

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
# Shared parameters across all 5 base policies
# ---------------------------------------------------------------------------
SHARED_ARGS=(
    --policy flowmatching
    --network_architecture mlp
    --mlp_dims "[1024, 1024, 1024]"
    --vision_backbone vit
    --flow_network_output_param u
    --cfm_loss_mode u
    --cfm_loss_use_huber False
    --cfm_loss_huber_delta 0.5
    --grad_clip_norm 25
    --horizon 16
    --n_action_steps 8
    --sampling_steps 10
    --batch_size 512
    --learning_rate 1e-4
    --lr_backbone 1e-5
    --weight_decay 1e-6
    --ema_power 0.995
    --enable_geometric_augmentations True
    --seed 3
    --num_workers 16
    --wandb_enable True
    --wandb_project flow-bc
    --eval_num_envs 25
    --eval_num_episodes 50
    --log_freq 5
    --save_freq 1000
    --rollout_freq 1000
)

###########################################################################
# Can — base policy run: 95j3noe4
###########################################################################
echo "====================================================================="
echo "  Pretraining: Can"
echo "====================================================================="

run_cmd python pretrain_flow_bc.py \
    "${SHARED_ARGS[@]}" \
    --dataset ankile/robomimic-mh-can-image \
    --image_observation_keys "robot0_eye_in_hand_image" \
    --eval_env Can \
    --steps 500000 \
    --experiment flow_bc_can

###########################################################################
# Square — base policy run: trc7rbt0
###########################################################################
echo "====================================================================="
echo "  Pretraining: Square"
echo "====================================================================="

run_cmd python pretrain_flow_bc.py \
    "${SHARED_ARGS[@]}" \
    --dataset ankile/robomimic-mh-square-image \
    --image_observation_keys "agentview_image" \
    --eval_env Square \
    --steps 1000000 \
    --max_num_episodes 100 \
    --experiment flow_bc_square

###########################################################################
# Box Clearance (TwoArmBoxCleanup) — base policy run: lainyisy
###########################################################################
echo "====================================================================="
echo "  Pretraining: Box Clearance"
echo "====================================================================="

run_cmd python pretrain_flow_bc.py \
    "${SHARED_ARGS[@]}" \
    --dataset ankile/dexmg-two-arm-box-cleanup \
    --image_observation_keys "agentview_image robot0_eye_in_hand_image robot1_eye_in_hand_image" \
    --eval_env TwoArmBoxCleanup \
    --steps 1000000 \
    --experiment flow_bc_boxcleaning

###########################################################################
# Tray Lifting (TwoArmLiftTray) — base policy run: ri0w9j39
###########################################################################
echo "====================================================================="
echo "  Pretraining: Tray Lifting"
echo "====================================================================="

run_cmd python pretrain_flow_bc.py \
    "${SHARED_ARGS[@]}" \
    --dataset ankile/dexmg-two-arm-lift-tray \
    --image_observation_keys "agentview_image robot0_eye_in_hand_image robot1_eye_in_hand_image" \
    --eval_env TwoArmLiftTray \
    --steps 1000000 \
    --experiment flow_bc_tray

###########################################################################
# Threading (TwoArmThreading) — base policy run: 6vqrn614
###########################################################################
echo "====================================================================="
echo "  Pretraining: Threading"
echo "====================================================================="

run_cmd python pretrain_flow_bc.py \
    "${SHARED_ARGS[@]}" \
    --dataset ankile/dexmg-two-arm-threading \
    --image_observation_keys "agentview_image robot0_eye_in_hand_image robot1_eye_in_hand_image" \
    --eval_env TwoArmThreading \
    --steps 1000000 \
    --experiment flow_bc_threading

echo "====================================================================="
echo "  All 5 base policy pretraining runs complete"
echo "====================================================================="
