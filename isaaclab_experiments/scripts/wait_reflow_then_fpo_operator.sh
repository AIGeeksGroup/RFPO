#!/usr/bin/env bash
# After Go2 plain reflow (2500) finishes on GPU0, launch fixed fpo_operator (2500).
#
# Usage:
#   nohup bash scripts/wait_reflow_then_fpo_operator.sh >/dev/null 2>&1 &
#   REFLOW_PID=3805885 GPU=0 bash scripts/wait_reflow_then_fpo_operator.sh
#
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source source_env.sh

export OMNI_KIT_ACCEPT_EULA=YES
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export TMPDIR="${TMPDIR:-/nix/plsy/tmp/tmpdir}"
mkdir -p "$TMPDIR"

TASK="Isaac-Velocity-Flat-Unitree-Go2-v0"
GPU="${GPU:-0}"
NUM_ENVS="${NUM_ENVS:-16384}"
MAX_ITERS="${MAX_ITERS:-2500}"
REFLOW_PID="${REFLOW_PID:-}"
REFLOW_LOG="${REFLOW_LOG:-${ROOT}/logs/isaaclab_fpo/reflow_missing_2500/reflow.log}"
LOG_DIR="${ROOT}/logs/isaaclab_fpo/reflow_missing_2500"
mkdir -p "$LOG_DIR"
OUT="${LOG_DIR}/wait_fpo_operator.out"
STAMP="$(date +%Y-%m-%d_%H-%M-%S)"

if [[ -z "$REFLOW_PID" ]]; then
  if [[ -f "${LOG_DIR}/pids.txt" ]]; then
    REFLOW_PID="$(awk '$3=="reflow"{print $1; exit}' "${LOG_DIR}/pids.txt" || true)"
  fi
fi
if [[ -z "$REFLOW_PID" ]]; then
  REFLOW_PID="$(pgrep -f 'train\.py.*Unitree-Go2.*fpo_variant reflow.*2026-08-24_20-16-24_reflow' | head -1 || true)"
fi

echo "[$(date '+%F %T')] wait_reflow_then_fpo_operator starting (GPU=${GPU}, REFLOW_PID=${REFLOW_PID:-none})" | tee -a "$OUT"

if [[ -n "$REFLOW_PID" ]] && kill -0 "$REFLOW_PID" 2>/dev/null; then
  echo "[$(date '+%F %T')] Waiting for Go2 reflow PID ${REFLOW_PID}..." | tee -a "$OUT"
  while kill -0 "$REFLOW_PID" 2>/dev/null; do
    sleep 30
  done
  echo "[$(date '+%F %T')] Go2 reflow PID ${REFLOW_PID} finished." | tee -a "$OUT"
else
  echo "[$(date '+%F %T')] No live reflow PID; checking log completion..." | tee -a "$OUT"
  if [[ -f "$REFLOW_LOG" ]] && grep -q 'Learning iteration 2499/2500' "$REFLOW_LOG"; then
    echo "[$(date '+%F %T')] reflow.log shows 2499/2500; proceeding." | tee -a "$OUT"
  else
    echo "[$(date '+%F %T')] WARN: reflow not clearly finished; launching fpo_operator anyway on GPU ${GPU}." | tee -a "$OUT"
  fi
fi

# Brief pause so GPU memory is released
sleep 20

FPO_LOG="${LOG_DIR}/fpo_operator_${STAMP}.log"
echo "[$(date '+%F %T')] Launching Go2 fpo_operator (fixed EMA endpoint) on GPU ${GPU} -> ${FPO_LOG}" | tee -a "$OUT"

CUDA_VISIBLE_DEVICES="${GPU}" nohup python isaaclab_fpo/scripts/train.py \
  --task "$TASK" \
  --headless \
  --disable_fabric \
  --num_envs "$NUM_ENVS" \
  --fpo_variant fpo_operator \
  --max_iterations "$MAX_ITERS" \
  --run_name "${STAMP}_fpo_operator_fixed" \
  agent.device=cuda:0 \
  agent.enable_post_training_eval=false \
  > "${FPO_LOG}" 2>&1 &

FPO_PID=$!
echo "${FPO_PID} ${GPU} fpo_operator_fixed" >> "${LOG_DIR}/pids.txt"
echo "[$(date '+%F %T')] fpo_operator PID=${FPO_PID} stamp=${STAMP}" | tee -a "$OUT"
echo "$FPO_PID" > "${LOG_DIR}/fpo_operator_fixed.pid"
