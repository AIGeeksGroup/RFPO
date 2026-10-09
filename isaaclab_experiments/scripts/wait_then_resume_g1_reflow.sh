#!/usr/bin/env bash
# Wait for GPU2 all_ideas_teacher_kd to finish, then resume G1 pure reflow
# from model_1150.pt through iteration 2000. Does not touch GPU0/1/3.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WRAPPER_PID="${WRAPPER_PID:-1411005}"
TRAIN_PID="${TRAIN_PID:-1412163}"
GPU="${GPU:-2}"
CKPT="${CKPT:-${ROOT}/logs/isaaclab_fpo/g1_flat_flow_reflow/2026-08-27_00-04-43_2026-08-27_00-02-39_g1_reflow/model_1150.pt}"
# resume at 1150; tot_iter = 1150 + 850 = 2000
MAX_ITERS="${MAX_ITERS:-850}"
NUM_ENVS="${NUM_ENVS:-8192}"

WORKDIR="/dev/shm/g1_reflow_resume"
LAUNCH_LOG_DIR="${WORKDIR}/launch_logs"
WS_LOG_DIR="${ROOT}/logs/isaaclab_fpo/g1_reflow_resume_1150"
OUT="${LAUNCH_LOG_DIR}/watcher.out"
mkdir -p "$LAUNCH_LOG_DIR" "$WS_LOG_DIR" /dev/shm/_g1_reflow_tmp

log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT"; }

log "watcher start: wait wrapper=${WRAPPER_PID} train=${TRAIN_PID} then resume reflow on GPU${GPU}"
log "ckpt=${CKPT}"
if [[ ! -f "$CKPT" ]]; then
  log "FATAL: checkpoint missing"
  exit 1
fi
cp -f "$CKPT" "${WORKDIR}/model_1150.pt" 2>/dev/null || true
CKPT_LOCAL="${WORKDIR}/model_1150.pt"
if [[ -f "$CKPT_LOCAL" ]]; then
  CKPT="$CKPT_LOCAL"
  log "using shm copy ${CKPT}"
fi

still_alive() {
  local p="$1"
  [[ -n "$p" && -d "/proc/${p}" ]]
}

# Also treat any live teacher-kd train.py as the wait target, in case pids restart.
teacher_kd_running() {
  still_alive "$WRAPPER_PID" && return 0
  still_alive "$TRAIN_PID" && return 0
  pgrep -f 'fpo_variant all_ideas_teacher_kd' >/dev/null 2>&1
}

while teacher_kd_running; do
  iter="$(rg -o 'Learning iteration [0-9]+/2000' /dev/shm/g1_all_ideas_teacher_kd/launch_logs/train.log 2>/dev/null | tail -1 || true)"
  log "waiting for teacher_kd (${iter:-unknown})"
  sleep 120
done
log "teacher_kd processes gone; cooling down GPU${GPU}"
sleep 90

# If GPU2 still looks occupied by dead host contexts, try a device reset.
# Never reset GPU 0/1/3.
mem="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | tr -d ' ' || echo 99999)"
log "GPU${GPU} memory after cooldown: ${mem} MiB"
if [[ "${mem}" =~ ^[0-9]+$ ]] && (( mem > 4096 )); then
  log "attempting nvidia-smi --gpu-reset -i ${GPU}"
  nvidia-smi --gpu-reset -i "$GPU" >>"$OUT" 2>&1 && log "gpu-reset ok" || log "gpu-reset failed (non-fatal)"
  sleep 5
  mem="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | tr -d ' ' || echo unknown)"
  log "GPU${GPU} memory after reset: ${mem} MiB"
fi

source "${ROOT}/source_env.sh"
export OMNI_KIT_ACCEPT_EULA=YES
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export TMPDIR=/dev/shm/_g1_reflow_tmp
export PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR"

STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
run_name="${STAMP}_g1_reflow_resume1150"
TRAIN_LOG="${LAUNCH_LOG_DIR}/train.log"

sync_loop() {
  local src_root="${WORKDIR}/logs/isaaclab_fpo/g1_flat_flow_reflow"
  local dst_root="${WS_LOG_DIR}"
  while true; do
    mkdir -p "$dst_root"
    if [[ -d "$src_root" ]]; then
      find "$src_root" -name 'model_*.pt' -print0 2>/dev/null \
        | while IFS= read -r -d '' ckpt; do
            rel="${ckpt#${src_root}/}"
            mkdir -p "${dst_root}/$(dirname "$rel")"
            cp -f "$ckpt" "${dst_root}/${rel}" 2>/dev/null || true
          done
    fi
    sleep 120
  done
}
sync_loop > "${LAUNCH_LOG_DIR}/sync.log" 2>&1 &
echo "sync_pid=$!" | tee -a "$OUT"

log "launch reflow resume GPU=${GPU} max_iters=${MAX_ITERS} envs=${NUM_ENVS}"
cd "$WORKDIR"
CUDA_VISIBLE_DEVICES="${GPU}" python "${ROOT}/isaaclab_fpo/scripts/train.py" \
  --task Isaac-Velocity-Flat-G1-v0 \
  --headless \
  --disable_fabric \
  --num_envs "$NUM_ENVS" \
  --fpo_variant reflow \
  --max_iterations "$MAX_ITERS" \
  --seed 42 \
  --resume \
  --checkpoint "$CKPT" \
  --run_name "$run_name" \
  agent.device=cuda:0 \
  agent.enable_post_training_eval=false \
  > "$TRAIN_LOG" 2>&1
rc=$?
log "reflow resume exit=${rc} log=${TRAIN_LOG}"
if [[ "$rc" -ne 0 ]]; then
  tail -n 80 "$TRAIN_LOG" | tee -a "$OUT"
fi
cp -f "$TRAIN_LOG" "${WS_LOG_DIR}/train.log" 2>/dev/null || true
exit "$rc"
