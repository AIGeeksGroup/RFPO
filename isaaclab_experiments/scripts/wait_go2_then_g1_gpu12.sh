#!/usr/bin/env bash
# When current Go2 jobs on GPU1/GPU2 finish, start G1 baseline + all_ideas.
#
# Defaults (as of 2026-08-25):
#   GPU1 waits for reflow_coef1p0 (PID 2597893) → G1 baseline
#   GPU2 waits for Go2 all_ideas (PID 3805891) → G1 all_ideas
#
# Usage:
#   nohup bash scripts/wait_go2_then_g1_gpu12.sh >> logs/.../wait_g1.out 2>&1 &
#
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# Use env python explicitly — parent shells may have another venv first on PATH.
source source_env.sh
PY="${ROOT}/thirdparty/miniconda3/envs/isaaclab_fpo/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "ERROR: isaaclab_fpo python missing: $PY" >&2
  exit 1
fi

export OMNI_KIT_ACCEPT_EULA=YES
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export TMPDIR="${TMPDIR:-/nix/plsy/tmp/tmpdir}"
mkdir -p "$TMPDIR"

TASK="Isaac-Velocity-Flat-G1-v0"
NUM_ENVS="${NUM_ENVS:-8192}"
MAX_ITERS="${MAX_ITERS:-2000}"
STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_DIR="${ROOT}/logs/isaaclab_fpo/g1_wait_gpu12_${STAMP}"
mkdir -p "$LOG_DIR"
OUT="${LOG_DIR}/wait.out"

# Go2 jobs currently on GPU1 / GPU2
WAIT_PID_GPU1="${WAIT_PID_GPU1:-2597893}"   # reflow_coef1p0
WAIT_PID_GPU2="${WAIT_PID_GPU2:-3805891}"   # all_ideas
VARIANT_GPU1="${VARIANT_GPU1:-baseline}"
VARIANT_GPU2="${VARIANT_GPU2:-all_ideas}"

echo "[$(date '+%F %T')] wait_go2_then_g1_gpu12 stamp=${STAMP}" | tee -a "$OUT"
echo "  GPU1: wait PID ${WAIT_PID_GPU1} → G1 ${VARIANT_GPU1}" | tee -a "$OUT"
echo "  GPU2: wait PID ${WAIT_PID_GPU2} → G1 ${VARIANT_GPU2}" | tee -a "$OUT"
echo "  num_envs=${NUM_ENVS} max_iters=${MAX_ITERS}" | tee -a "$OUT"

wait_pid() {
  local pid="$1"
  local label="$2"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "[$(date '+%F %T')] Waiting for ${label} PID ${pid}..." | tee -a "$OUT"
    while kill -0 "$pid" 2>/dev/null; do
      sleep 30
    done
    echo "[$(date '+%F %T')] ${label} PID ${pid} finished." | tee -a "$OUT"
  else
    echo "[$(date '+%F %T')] ${label} PID ${pid:-none} not running; proceed." | tee -a "$OUT"
  fi
}

launch_g1() {
  local gpu="$1"
  local variant="$2"
  local log_file="${LOG_DIR}/g1_${variant}_gpu${gpu}.log"
  echo "[$(date '+%F %T')] Launching G1 ${variant} on GPU ${gpu} -> ${log_file}" | tee -a "$OUT"
  CUDA_VISIBLE_DEVICES="${gpu}" nohup "${PY}" isaaclab_fpo/scripts/train.py \
    --task "$TASK" \
    --headless \
    --disable_fabric \
    --num_envs "$NUM_ENVS" \
    --fpo_variant "$variant" \
    --max_iterations "$MAX_ITERS" \
    --run_name "${STAMP}_g1_${variant}" \
    agent.device=cuda:0 \
    agent.enable_post_training_eval=false \
    > "${log_file}" 2>&1 &
  local pid=$!
  echo "${pid} ${gpu} g1_${variant}" >> "${LOG_DIR}/pids.txt"
  echo "[$(date '+%F %T')] G1 ${variant} PID=${pid}" | tee -a "$OUT"
}

# Parallel waiters so whichever GPU frees first starts sooner
(
  wait_pid "$WAIT_PID_GPU1" "Go2-GPU1"
  sleep 25
  launch_g1 1 "$VARIANT_GPU1"
) &
W1=$!

(
  wait_pid "$WAIT_PID_GPU2" "Go2-GPU2"
  sleep 25
  launch_g1 2 "$VARIANT_GPU2"
) &
W2=$!

echo "$W1 $W2" > "${LOG_DIR}/waiter.pids"
echo "[$(date '+%F %T')] waiter PIDs: GPU1-side=${W1} GPU2-side=${W2}" | tee -a "$OUT"
wait "$W1" "$W2"
echo "[$(date '+%F %T')] both G1 jobs launched (or skipped)." | tee -a "$OUT"
