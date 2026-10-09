#!/usr/bin/env bash
# Wait for GPU3 G1 all_ideas_fpo resume to finish, then launch Go2 full baseline.
# Does not touch GPU0/1/2.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TRAIN_PID="${TRAIN_PID:-1411017}"
GPU="${GPU:-3}"
WORKDIR="/dev/shm/go2_baseline_full"
LAUNCH_LOG_DIR="${WORKDIR}/launch_logs"
OUT="${LAUNCH_LOG_DIR}/watcher.out"
mkdir -p "$LAUNCH_LOG_DIR"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT"; }

log "watcher start: wait gpu3 all_ideas_fpo pid=${TRAIN_PID}, then Go2 baseline 16384x2500"

still_alive() {
  local p="$1"
  [[ -n "$p" && -d "/proc/${p}" ]]
}

g1_gpu3_running() {
  still_alive "$TRAIN_PID" && return 0
  pgrep -f 'fpo_variant all_ideas_fpo' >/dev/null 2>&1
}

while g1_gpu3_running; do
  iter="$(rg -o 'Learning iteration [0-9]+/2000' /dev/shm/g1_resume/launch_logs/resume1500.out 2>/dev/null | tail -1 || true)"
  log "waiting for all_ideas_fpo (${iter:-unknown})"
  sleep 120
done
log "all_ideas_fpo gone; cooling down GPU${GPU}"
sleep 90

mem="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | tr -d ' ' || echo 99999)"
log "GPU${GPU} memory after cooldown: ${mem} MiB"
if [[ "${mem}" =~ ^[0-9]+$ ]] && (( mem > 4096 )); then
  log "attempting nvidia-smi --gpu-reset -i ${GPU}"
  nvidia-smi --gpu-reset -i "$GPU" >>"$OUT" 2>&1 && log "gpu-reset ok" || log "gpu-reset failed (non-fatal)"
  sleep 5
fi

log "launching Go2 full baseline on GPU${GPU}"
GPU="$GPU" bash "${ROOT}/scripts/run_go2_baseline_full.sh"
rc=$?
log "go2 baseline full exit=${rc}"
exit "$rc"
