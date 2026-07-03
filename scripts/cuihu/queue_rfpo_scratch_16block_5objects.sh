#!/usr/bin/env bash
set -euo pipefail

ROOT="${CUIHU_ROOT:-/share/project/liyanjun/rongshanyu}"
REPO="${ROOT}/vividex_sapien_server_upload_fpo_control_full"
LOG_ROOT="${ROOT}/logs"
GPU_USED_LIMIT_MB="${GPU_USED_LIMIT_MB:-12000}"
POLL_SECONDS="${POLL_SECONDS:-120}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"

mkdir -p "${LOG_ROOT}"
cd "${REPO}"

OBJECT_TAGS=(clamp mug mustard sugar tomato)
OBJECT_SEQS=(
  ycb-052_extra_large_clamp-20200709-subject-01-20200709_152843
  ycb-025_mug-20200709-subject-01-20200709_150949
  ycb-006_mustard_bottle-20200709-subject-01-20200709_143211
  ycb-004_sugar_box-20200918-subject-06-20200918_113441
  ycb-005_tomato_soup_can-20200709-subject-01-20200709_142853
)

pid_alive() {
  local pid_file="$1"
  [[ -f "${pid_file}" ]] || return 1
  local pid
  pid="$(cat "${pid_file}" 2>/dev/null || true)"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

choose_free_gpu() {
  local locked="$1"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F, -v limit="${GPU_USED_LIMIT_MB}" -v allowed="${GPU_LIST}" -v locked="${locked}" '
      BEGIN {
        n = split(allowed, ids, ",")
        for (i = 1; i <= n; i++) ok[ids[i] + 0] = 1
        m = split(locked, busy, ",")
        for (i = 1; i <= m; i++) if (busy[i] != "") used[busy[i] + 0] = 1
      }
      {
        g = $1 + 0
        mem = $2 + 0
        if ((g in ok) && !(g in used) && mem < limit) {
          print g
          exit 0
        }
      }
    '
}

active_locked_gpus() {
  local result=""
  local pid_file gpu_file pid gpu
  for pid_file in "${LOG_ROOT}"/rfpo16_*.pid; do
    [[ -f "${pid_file}" ]] || continue
    [[ "${pid_file}" == *rfpo16_5objects_queue.pid ]] && continue
    pid="$(cat "${pid_file}" 2>/dev/null || true)"
    [[ -n "${pid}" ]] || continue
    kill -0 "${pid}" 2>/dev/null || continue
    gpu_file="${pid_file%.pid}.gpu"
    [[ -f "${gpu_file}" ]] || continue
    gpu="$(cat "${gpu_file}" 2>/dev/null || true)"
    [[ -n "${gpu}" ]] || continue
    result="${result:+${result},}${gpu}"
  done
  printf '%s\n' "${result}"
}

QUEUE_LOG="${LOG_ROOT}/rfpo16_5objects_queue.out"

if [[ "${RFPO_QUEUE_CHILD:-0}" != "1" ]]; then
  export RFPO_QUEUE_CHILD=1
  nohup setsid bash "$0" "$@" >> "${QUEUE_LOG}" 2>&1 < /dev/null &
  echo "$!" > "${LOG_ROOT}/rfpo16_5objects_queue.pid"
  echo "[rfpo16-queue] queue_pid=$(cat "${LOG_ROOT}/rfpo16_5objects_queue.pid")"
  echo "[rfpo16-queue] queue_log=${QUEUE_LOG}"
  exit 0
fi

{
  echo "[rfpo16-queue] started at $(date '+%F %T')"
  echo "[rfpo16-queue] gpu_list=${GPU_LIST}, used_limit_mb=${GPU_USED_LIMIT_MB}"

  for i in "${!OBJECT_TAGS[@]}"; do
    tag="${OBJECT_TAGS[$i]}"
    seq="${OBJECT_SEQS[$i]}"
    run_prefix="rfpo16_${tag}"
    pid_file="${LOG_ROOT}/${run_prefix}.pid"
    train_log="${LOG_ROOT}/${run_prefix}.out"

    if pid_alive "${pid_file}"; then
      echo "[rfpo16-queue] ${tag} already running pid=$(cat "${pid_file}")"
      continue
    fi

    while true; do
      active_gpus="$(active_locked_gpus)"
      gpu="$(choose_free_gpu "${active_gpus}" || true)"
      if [[ -n "${gpu}" ]]; then
        echo "[rfpo16-queue] launching ${tag} on gpu=${gpu} at $(date '+%F %T')"
        CUDA_VISIBLE_DEVICES="${gpu}" \
        SEQ="${seq}" \
        OBJECT_TAG="${tag}" \
        RUN_PREFIX="${run_prefix}" \
        RFPO_FINAL_TARGET="${RFPO_FINAL_TARGET:-20000000}" \
        EVAL_FREQ="${EVAL_FREQ:-200000}" \
        EVAL_N_EPISODES="${EVAL_N_EPISODES:-25}" \
        VIVIDEX_N_ENVS="${VIVIDEX_N_ENVS:-16}" \
        VIVIDEX_N_EVAL_ENVS="${VIVIDEX_N_EVAL_ENVS:-4}" \
        nohup setsid bash -c '
          set -euo pipefail
          bash scripts/cuihu/run_rfpo_scratch_16block_object.sh
        ' > "${train_log}" 2>&1 < /dev/null &
        echo "$!" > "${pid_file}"
        echo "${gpu}" > "${LOG_ROOT}/${run_prefix}.gpu"
        echo "[rfpo16-queue] ${tag} pid=$(cat "${pid_file}") log=${train_log}"
        break
      fi

      echo "[rfpo16-queue] $(date '+%F %T') no free gpu for ${tag}; active_gpus=${active_gpus}; current:"
      nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
      sleep "${POLL_SECONDS}"
    done
  done

  echo "[rfpo16-queue] all launch attempts submitted"
} >> "${QUEUE_LOG}" 2>&1
