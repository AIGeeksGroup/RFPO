#!/usr/bin/env bash
# Wait for Spot theory training to finish, then run sampling-steps sweep on the same GPU.
#
# Default: watch PID from spot_reflow_ideas_1500/pids.txt (theory), GPU=1,
#          then ROBOT=spot VARIANTS=theory APPEND=1 step sweep.
#
# Usage:
#   nohup bash scripts/wait_spot_theory_then_step_sweep.sh &
#   TRAIN_PID=3929332 GPU=1 bash scripts/wait_spot_theory_then_step_sweep.sh
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG_DIR="${ROOT}/logs/isaaclab_fpo/spot_reflow_ideas_1500"
STAMP="$(cat "${LOG_DIR}/stamp.txt" 2>/dev/null || echo "2026-08-22_19-14-14")"
OUT="${LOG_DIR}/wait_theory_then_sweep.out"
CKPT_GLOB="${ROOT}/logs/isaaclab_fpo/spot_reflow_theory/*${STAMP}*_theory/model_1499.pt"
POLL_SEC="${POLL_SEC:-60}"
GPU="${GPU:-1}"

resolve_train_pid() {
  if [[ -n "${TRAIN_PID:-}" ]]; then
    echo "$TRAIN_PID"
    return
  fi
  if [[ -f "${LOG_DIR}/pids.txt" ]]; then
    local pid
    pid="$(awk '$3=="theory"{print $1; exit}' "${LOG_DIR}/pids.txt" || true)"
    if [[ -n "$pid" ]]; then
      echo "$pid"
      return
    fi
  fi
  pgrep -f 'train\.py.*Isaac-Velocity-Flat-Spot-v0.*fpo_variant theory' | head -1 || true
}

final_ckpt_ready() {
  ls -1 $CKPT_GLOB >/dev/null 2>&1
}

echo "[$(date '+%F %T')] wait_spot_theory_then_step_sweep starting (GPU=${GPU})" | tee -a "$OUT"

TRAIN_PID="$(resolve_train_pid || true)"
if [[ -n "$TRAIN_PID" ]]; then
  echo "[$(date '+%F %T')] Watching theory train PID ${TRAIN_PID}" | tee -a "$OUT"
else
  echo "[$(date '+%F %T')] No train PID found; will wait for model_1499.pt only" | tee -a "$OUT"
fi

while true; do
  alive=0
  if [[ -n "$TRAIN_PID" ]] && kill -0 "$TRAIN_PID" 2>/dev/null; then
    alive=1
  fi
  # Re-discover if original pid died but another theory train exists
  if [[ "$alive" -eq 0 ]]; then
    TRAIN_PID="$(resolve_train_pid || true)"
    if [[ -n "$TRAIN_PID" ]] && kill -0 "$TRAIN_PID" 2>/dev/null; then
      alive=1
      echo "[$(date '+%F %T')] Re-attached theory train PID ${TRAIN_PID}" | tee -a "$OUT"
    fi
  fi

  if [[ "$alive" -eq 0 ]] && final_ckpt_ready; then
    CKPT="$(ls -1 $CKPT_GLOB | tail -1)"
    echo "[$(date '+%F %T')] Theory finished. ckpt=${CKPT}" | tee -a "$OUT"
    break
  fi

  if [[ "$alive" -eq 0 ]] && ! final_ckpt_ready; then
    echo "[$(date '+%F %T')] Train not running and model_1499.pt missing; keep waiting ${POLL_SEC}s" | tee -a "$OUT"
  else
    # progress crumb
    last_iter="$(grep -E 'Learning iteration' "${LOG_DIR}/theory.log" 2>/dev/null | tail -1 | tr -d '\033' | sed 's/\[[0-9;]*m//g' | xargs || true)"
    echo "[$(date '+%F %T')] still training... ${last_iter}" | tee -a "$OUT"
  fi
  sleep "$POLL_SEC"
done

# Brief settle so GPU memory is released
sleep "${SETTLE_SEC:-30}"

echo "[$(date '+%F %T')] Launching Spot theory step sweep on GPU ${GPU}" | tee -a "$OUT"
export GPU="${GPU}"
export ROBOT=spot
export VARIANTS=theory
export APPEND=1
# Avoid clobbering concurrent Spot baseline/ideas step_sweep.out on other GPUs
export OUT="${LOG_DIR}/step_sweep_theory.out"
export SUMMARY="${LOG_DIR}/step_sweep_summary.txt"
bash "${ROOT}/scripts/eval_sampling_steps_sweep.sh" | tee -a "$OUT"
echo "[$(date '+%F %T')] Theory step sweep done. See ${SUMMARY} and ${OUT}" | tee -a "$OUT"
