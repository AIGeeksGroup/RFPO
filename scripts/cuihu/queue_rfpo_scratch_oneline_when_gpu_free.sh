#!/usr/bin/env bash
set -euo pipefail

ROOT="${CUIHU_ROOT:-/share/project/liyanjun/rongshanyu}"
REPO="${ROOT}/vividex_sapien_server_upload_fpo_control_full"
LOG_ROOT="${ROOT}/logs"
RUN_NAME="${RUN_NAME:-rfpo_scratch_oneline_best76_80m}"
GPU_USED_LIMIT_MB="${GPU_USED_LIMIT_MB:-20000}"
POLL_SECONDS="${POLL_SECONDS:-120}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"

mkdir -p "${LOG_ROOT}"
cd "${REPO}"

pid_alive() {
  local pid_file="$1"
  [[ -f "${pid_file}" ]] || return 1
  local pid
  pid="$(cat "${pid_file}" 2>/dev/null || true)"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

choose_free_gpu() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F, -v limit="${GPU_USED_LIMIT_MB}" -v allowed="${GPU_LIST}" '
      BEGIN {
        n = split(allowed, ids, ",")
        for (i = 1; i <= n; i++) ok[ids[i] + 0] = 1
      }
      {
        g = $1 + 0
        used = $2 + 0
        if ((g in ok) && used < limit) {
          print g
          exit 0
        }
      }
    '
}

TRAIN_PID_FILE="${LOG_ROOT}/${RUN_NAME}.pid"
QUEUE_LOG="${LOG_ROOT}/${RUN_NAME}_queue.out"
TRAIN_LOG="${LOG_ROOT}/${RUN_NAME}.out"

if pid_alive "${TRAIN_PID_FILE}"; then
  echo "[rfpo-oneline-queue] training already running pid=$(cat "${TRAIN_PID_FILE}")"
  exit 0
fi

{
  echo "[rfpo-oneline-queue] started at $(date '+%F %T')"
  echo "[rfpo-oneline-queue] run_name=${RUN_NAME}"
  echo "[rfpo-oneline-queue] gpu_list=${GPU_LIST}, used_limit_mb=${GPU_USED_LIMIT_MB}"
  while true; do
    if pid_alive "${TRAIN_PID_FILE}"; then
      echo "[rfpo-oneline-queue] training already running pid=$(cat "${TRAIN_PID_FILE}")"
      exit 0
    fi

    gpu="$(choose_free_gpu || true)"
    if [[ -n "${gpu}" ]]; then
      echo "[rfpo-oneline-queue] launching on gpu=${gpu} at $(date '+%F %T')"
      (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        export RUN_NAME="${RUN_NAME}"
        export TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-80000000}"
        export EVAL_FREQ="${EVAL_FREQ:-200000}"
        export EVAL_N_EPISODES="${EVAL_N_EPISODES:-25}"
        export VIVIDEX_N_ENVS="${VIVIDEX_N_ENVS:-16}"
        export VIVIDEX_N_EVAL_ENVS="${VIVIDEX_N_EVAL_ENVS:-4}"
        bash scripts/cuihu/run_rfpo_scratch_oneline_best76.sh
      ) > "${TRAIN_LOG}" 2>&1 &
      echo "$!" > "${TRAIN_PID_FILE}"
      echo "[rfpo-oneline-queue] train_pid=$(cat "${TRAIN_PID_FILE}")"
      echo "[rfpo-oneline-queue] train_log=${TRAIN_LOG}"
      exit 0
    fi

    echo "[rfpo-oneline-queue] $(date '+%F %T') no gpu below ${GPU_USED_LIMIT_MB} MiB; current:"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
    sleep "${POLL_SECONDS}"
  done
} >> "${QUEUE_LOG}" 2>&1 &

echo "$!" > "${LOG_ROOT}/${RUN_NAME}_queue.pid"
echo "[rfpo-oneline-queue] queue_pid=$(cat "${LOG_ROOT}/${RUN_NAME}_queue.pid")"
echo "[rfpo-oneline-queue] queue_log=${QUEUE_LOG}"
