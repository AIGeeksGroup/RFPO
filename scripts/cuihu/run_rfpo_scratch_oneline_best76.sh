#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

SEQ="${SEQ:-ycb-006_mustard_bottle-20200709-subject-01-20200709_143211}"
RUN_NAME="${RUN_NAME:-rfpo_scratch_oneline_best76_80m}"

export VIVIDEX_N_ENVS="${VIVIDEX_N_ENVS:-16}"
export VIVIDEX_N_EVAL_ENVS="${VIVIDEX_N_EVAL_ENVS:-4}"
export WANDB_GROUP="${WANDB_GROUP:-rfpo_scratch_oneline}"
export WANDB_PREFIX="${WANDB_PREFIX:-rfpo_scratch_oneline_best76_80m}"

bash scripts/server/train_state_single.sh "$SEQ" "$RUN_NAME" \
  "agent=rfpo_scratch_oneline" \
  "resume_model=null" \
  "n_envs=${VIVIDEX_N_ENVS}" \
  "n_eval_envs=${VIVIDEX_N_EVAL_ENVS}" \
  "total_timesteps=${TOTAL_TIMESTEPS:-80000000}" \
  "eval_freq=${EVAL_FREQ:-200000}" \
  "eval_n_episodes=${EVAL_N_EPISODES:-25}" \
  "save_freq=${SAVE_FREQ:-1000000}" \
  "restore_checkpoint_freq=${RESTORE_FREQ:-1000000}" \
  "env.task_kwargs.reward_kwargs.obj_err_scale=50" \
  "env.task_kwargs.reward_kwargs.object_reward_scale=10.0"
