#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV="${VIVIDEX_VENV:-${REPO}/.venv-vividex}"
PYTHON="${VENV}/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing Python env: ${PYTHON}" >&2
  echo "Run scripts/local/setup_vividex_venv.sh first, or set VIVIDEX_VENV." >&2
  exit 1
fi

cd "${REPO}"
export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export VIVIDEX_HEADLESS_NO_RENDER="${VIVIDEX_HEADLESS_NO_RENDER:-1}"

SEQ="${SEQ:-ycb-006_mustard_bottle-20200709-subject-01-20200709_143211}"
RUN_ROOT="${VIVIDEX_LOCAL_RUN_ROOT:-${REPO}/.local_runs}"
ROOT="${ROOT:-${RUN_ROOT}/allegro_bc_compare_$(date +%Y%m%d_%H%M%S)}"
DATASET="${DATASET:-${ROOT}/allegro_reference_stage2_bc.npz}"
FPO_BC_DIR="${FPO_BC_DIR:-${ROOT}/flow_bc}"
FPO_BC="${FPO_BC:-${FPO_BC_DIR}/flow_bc_last.pt}"
FPO_DIR="${FPO_DIR:-${ROOT}/bcrfpo_conservative_stage2_500k}"
mkdir -p "${ROOT}"

if [[ ! -f "${DATASET}" || "${REGENERATE_BC_DATASET:-0}" == "1" ]]; then
  "${PYTHON}" tools/collect_allegro_reference_bc_dataset.py \
    --output "${DATASET}" \
    --seq-name "${SEQ}" \
    --norm-traj \
    --stage "${BC_STAGE:-2}" \
    --num-trajs "${BC_NUM_TRAJS:-64}" \
    --max-attempts "${BC_MAX_ATTEMPTS:-512}" \
    --success-metric "${BC_SUCCESS_METRIC:-sr10}" \
    --palm-gain "${BC_PALM_GAIN:-0.04}"
fi

if [[ ! -f "${FPO_BC}" || "${REGENERATE_FPO_BC:-0}" == "1" ]]; then
  "${PYTHON}" tools/pretrain_fpo_bc.py \
    --dataset "${DATASET}" \
    --output_dir "${FPO_BC_DIR}" \
    --steps "${FPO_BC_STEPS:-50000}" \
    --batch_size "${FPO_BC_BATCH_SIZE:-1024}" \
    --learning_rate "${FPO_BC_LR:-1e-4}" \
    --save_freq "${FPO_BC_SAVE_FREQ:-25000}" \
    --actor_hidden_dims "${FPO_ACTOR_DIMS:-[512,512,512]}" \
    --critic_hidden_dims "${FPO_CRITIC_DIMS:-[512,512]}" \
    --activation "${FPO_ACTIVATION:-relu}" \
    --sampling_steps "${FPO_SAMPLING_STEPS:-10}" \
    --seed "${SEED:-0}"
fi

N_ENVS="${N_ENVS:-8}"
N_EVAL_ENVS="${N_EVAL_ENVS:-2}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-500000}"
EVAL_FREQ="${EVAL_FREQ:-50000}"
EVAL_N_EPISODES="${EVAL_N_EPISODES:-25}"
SAVE_FREQ="${SAVE_FREQ:-100000}"
RESTORE_FREQ="${RESTORE_FREQ:-100000}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  N_ENVS="${SMOKE_N_ENVS:-1}"
  N_EVAL_ENVS="${SMOKE_N_EVAL_ENVS:-1}"
  TOTAL_TIMESTEPS="${SMOKE_TOTAL_TIMESTEPS:-64}"
  EVAL_FREQ="${SMOKE_EVAL_FREQ:-128}"
  EVAL_N_EPISODES="${SMOKE_EVAL_N_EPISODES:-1}"
  SAVE_FREQ="${SMOKE_SAVE_FREQ:-128}"
  RESTORE_FREQ="${SMOKE_RESTORE_FREQ:-128}"
fi

"${PYTHON}" tools/train.py \
  agent=fpo \
  "env.name=${SEQ}" \
  env.robot_name=allegro_hand_ur5 \
  env.norm_traj=True \
  "n_envs=${N_ENVS}" \
  "n_eval_envs=${N_EVAL_ENVS}" \
  "total_timesteps=${TOTAL_TIMESTEPS}" \
  "eval_freq=${EVAL_FREQ}" \
  "eval_n_episodes=${EVAL_N_EPISODES}" \
  "save_freq=${SAVE_FREQ}" \
  "restore_checkpoint_freq=${RESTORE_FREQ}" \
  vid_freq=null \
  "hydra.run.dir=${FPO_DIR}" \
  wandb.project=vividex_allegro_bc_compare \
  wandb.group=allegro_bc_compare_reference \
  wandb.sweep_name_prefix=allegro_bcrfpo_conservative_stage2_500k \
  "agent.params.bc_checkpoint=${FPO_BC}" \
  "agent.params.bc_anchor_dataset=${DATASET}" \
  agent.params.actor_hidden_dims='[512,512,512]' \
  agent.params.critic_hidden_dims='[512,512]' \
  agent.params.activation=relu \
  agent.params.sampling_steps=10 \
  agent.params.n_steps=4096 \
  agent.params.batch_size=256 \
  agent.params.n_epochs=1 \
  agent.params.learning_rate=1e-6 \
  agent.params.actor_objective=fpo \
  agent.params.fpo_objective_coef=0.25 \
  agent.params.fpo_objective_min_coef=0.25 \
  agent.params.gaussian_objective_coef=0.0 \
  agent.params.gaussian_objective_min_coef=0.0 \
  agent.params.eval_deterministic=True \
  agent.params.rollout_deterministic=True \
  agent.params.initial_curriculum_stage=2 \
  agent.params.curriculum_pregrasp_threshold=2.0 \
  agent.params.bc_anchor_coef=0.10 \
  agent.params.bc_anchor_min_coef=0.10 \
  agent.params.bc_anchor_decay_steps=0 \
  agent.params.bc_anchor_batch_size=256 \
  agent.params.action_anchor_coef=1.0 \
  agent.params.action_anchor_min_coef=1.0 \
  agent.params.action_anchor_decay_steps=0 \
  agent.params.action_anchor_batch_size=256 \
  agent.params.action_anchor_loss=huber \
  agent.params.action_anchor_huber_delta=0.02 \
  agent.params.target_kl=0.02 \
  agent.params.best_metric=eval/mean_reward \
  agent.params.best_metric_mode=max \
  agent.params.save_best_model=True
