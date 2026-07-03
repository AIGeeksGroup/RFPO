#!/usr/bin/env bash
set -euo pipefail

export CUIHU_ROOT="${CUIHU_ROOT:-/share/project/liyanjun/rongshanyu}"
export VIVIDEX_REPO_ROOT="${VIVIDEX_REPO_ROOT:-${CUIHU_ROOT}/vividex_sapien_server_upload_fpo_control_full}"
export VIVIDEX_CONDA_ENV="${VIVIDEX_CONDA_ENV:-${CUIHU_ROOT}/envs/vividex}"
export VIVIDEX_CONDA_SH="${VIVIDEX_CONDA_SH:-${CUIHU_ROOT}/miniconda3/etc/profile.d/conda.sh}"
export VIVIDEX_RUNTIME_ROOT="${VIVIDEX_RUNTIME_ROOT:-${CUIHU_ROOT}/results/vividex_fpo_runtime}"
export VIVIDEX_LOG_ROOT="${VIVIDEX_LOG_ROOT:-${CUIHU_ROOT}/logs}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export VIVIDEX_HEADLESS_NO_RENDER="${VIVIDEX_HEADLESS_NO_RENDER:-1}"

export http_proxy="${http_proxy:-http://10.8.36.23:2080}"
export https_proxy="${https_proxy:-http://10.8.36.23:2080}"

export PPO_RUN="${PPO_RUN:-${CUIHU_ROOT}/results/vividex/results/state_baseline/ppo_mustard_baseline}"
export PPO_10M="${PPO_10M:-${PPO_RUN}/logs/rl_models_10000000_steps.zip}"
export BC_DATASET="${BC_DATASET:-${CUIHU_ROOT}/data/fpo_bc/mustard_ppo10m_stage2.npz}"
export BC_OUT_DIR="${BC_OUT_DIR:-${CUIHU_ROOT}/results/fpo_bc_mustard_ppo10m}"
export BC_CKPT="${BC_CKPT:-${BC_OUT_DIR}/flow_bc_step_200000.pt}"
export SEQ="${SEQ:-ycb-006_mustard_bottle-20200709-subject-01-20200709_143211}"

mkdir -p "${VIVIDEX_LOG_ROOT}" "${CUIHU_ROOT}/data/fpo_bc" "${BC_OUT_DIR}" "${VIVIDEX_RUNTIME_ROOT}"

# shellcheck disable=SC1091
source "${VIVIDEX_REPO_ROOT}/scripts/server/common.sh"
vividex_setup_runtime
vividex_activate_env
cd "${VIVIDEX_REPO_ROOT}"

echo "[cuihu] host=$(hostname)"
echo "[cuihu] repo=${VIVIDEX_REPO_ROOT}"
echo "[cuihu] env=${VIVIDEX_CONDA_ENV}"
echo "[cuihu] runtime=${VIVIDEX_RUNTIME_ROOT}"
echo "[cuihu] log_root=${VIVIDEX_LOG_ROOT}"
