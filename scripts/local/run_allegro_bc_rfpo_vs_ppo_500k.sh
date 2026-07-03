#!/usr/bin/env bash
set -euo pipefail

cd /home/robot/QianSiyuan/RFPO/ref/05_code/vividex_sapien_server_upload_fpo_control_full

export WANDB_MODE=offline
export VIVIDEX_HEADLESS_NO_RENDER=1

PY=.venv-vividex/bin/python
ROOT=.local_runs/allegro_bc_compare_20260629_2303
DATASET="$PWD/$ROOT/allegro_mustard_ppo_teacher_stage2_success20.npz"
PPO_BC="$PWD/$ROOT/ppo_bc_success20/bc_ppo_last.zip"
FPO_BC="$PWD/$ROOT/flow_bc_success20/flow_bc_last.pt"

COMMON_ENV=(
  env.name=ycb-006_mustard_bottle-20200709-subject-01-20200709_143211
  env.robot_name=allegro_hand_ur5
  env.norm_traj=True
  n_envs=8
  n_eval_envs=2
  total_timesteps=500000
  eval_freq=50000
  eval_n_episodes=25
  save_freq=100000
  restore_checkpoint_freq=100000
  vid_freq=null
)

PPO_DIR="$ROOT/bcppo_online_500k"
FPO_DIR="$ROOT/bcrfpo_online_500k"

"$PY" tools/train.py \
  agent=ppo \
  "${COMMON_ENV[@]}" \
  hydra.run.dir="$PPO_DIR" \
  wandb.project=vividex_allegro_bc_compare \
  wandb.group=allegro_bc_compare_success20 \
  wandb.sweep_name_prefix=allegro_bcppo_success20_500k \
  resume_model="$PPO_BC" \
  agent.params.learning_rate=1e-5 \
  agent.params.n_steps=4096 \
  agent.params.batch_size=256 \
  agent.params.n_epochs=5 \
  agent.params.eval_deterministic=True \
  agent.params.bc_anchor_dataset="$DATASET" \
  agent.params.bc_anchor_coef=0.02 \
  agent.params.bc_anchor_min_coef=0.0 \
  agent.params.bc_anchor_decay_steps=250000 \
  agent.params.bc_anchor_batch_size=256

"$PY" tools/train.py \
  agent=fpo \
  "${COMMON_ENV[@]}" \
  hydra.run.dir="$FPO_DIR" \
  wandb.project=vividex_allegro_bc_compare \
  wandb.group=allegro_bc_compare_success20 \
  wandb.sweep_name_prefix=allegro_bcrfpo_success20_500k \
  agent.params.bc_checkpoint="$FPO_BC" \
  agent.params.bc_anchor_dataset="$DATASET" \
  agent.params.actor_hidden_dims='[512,512,512]' \
  agent.params.critic_hidden_dims='[512,512]' \
  agent.params.activation=relu \
  agent.params.sampling_steps=10 \
  agent.params.n_steps=4096 \
  agent.params.batch_size=256 \
  agent.params.n_epochs=4 \
  agent.params.learning_rate=1e-5 \
  agent.params.actor_objective=fpo \
  agent.params.fpo_objective_coef=1.0 \
  agent.params.fpo_objective_min_coef=1.0 \
  agent.params.gaussian_objective_coef=0.0 \
  agent.params.gaussian_objective_min_coef=0.0 \
  agent.params.eval_deterministic=True \
  agent.params.rollout_deterministic=True \
  agent.params.bc_anchor_coef=0.02 \
  agent.params.bc_anchor_min_coef=0.0 \
  agent.params.bc_anchor_decay_steps=250000 \
  agent.params.bc_anchor_batch_size=256 \
  agent.params.action_anchor_coef=0.2 \
  agent.params.action_anchor_min_coef=0.0 \
  agent.params.action_anchor_decay_steps=250000 \
  agent.params.action_anchor_batch_size=256 \
  agent.params.action_anchor_loss=huber \
  agent.params.action_anchor_huber_delta=0.02 \
  agent.params.best_metric=eval/mean_reward \
  agent.params.best_metric_mode=max \
  agent.params.save_best_model=True

"$PY" tools/render_rfpo_rollouts.py \
  --checkpoint "$PPO_DIR/models/model.zip" \
  --out "$ROOT/eval_bcppo_online_500k_stage2_100ep" \
  --episodes 100 \
  --stage 2 \
  --max-steps 120 \
  --no-render \
  --seed 0

"$PY" tools/render_rfpo_rollouts.py \
  --checkpoint "$FPO_DIR/models/best.pt" \
  --out "$ROOT/eval_bcrfpo_online_500k_best_stage2_100ep" \
  --episodes 100 \
  --stage 2 \
  --max-steps 120 \
  --no-render \
  --seed 0

"$PY" tools/render_rfpo_rollouts.py \
  --checkpoint "$FPO_DIR/models/last.pt" \
  --out "$ROOT/eval_bcrfpo_online_500k_last_stage2_100ep" \
  --episodes 100 \
  --stage 2 \
  --max-steps 120 \
  --no-render \
  --seed 0
