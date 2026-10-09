#!/usr/bin/env bash
# Launch four Spot reflow idea experiments on free GPUs (1500 iterations each).
# Wave 1: reward_aware (GPU0 after baseline) + adaptive_compute (GPU2 after standalone eval)
# Wave 2: fpo_operator (GPU0) + theory (GPU2)
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source source_env.sh

export OMNI_KIT_ACCEPT_EULA=YES
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export TMPDIR="${TMPDIR:-/nix/plsy/tmp/tmpdir}"
mkdir -p "$TMPDIR"

TASK="Isaac-Velocity-Flat-Spot-v0"
STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_DIR="${ROOT}/logs/isaaclab_fpo/spot_reflow_ideas_1500"
mkdir -p "$LOG_DIR"
echo "$STAMP" > "${LOG_DIR}/stamp.txt"
OUT="${LOG_DIR}/launch.out"
LAUNCHER_PID_FILE="${LOG_DIR}/launcher.pid"

NUM_ENVS="${NUM_ENVS:-16384}"
MAX_ITERS="${MAX_ITERS:-1500}"
EVAL_PID_FILE="${EVAL_PID_FILE:-/nix/plsy/tmp/eval_reflow_2500.pid}"

launch() {
  local gpu="$1"
  local variant="$2"
  local log_file="${LOG_DIR}/${variant}.log"
  echo "[$(date '+%F %T')] Launching Spot ${variant} on GPU ${gpu} (num_envs=${NUM_ENVS}) -> ${log_file}" | tee -a "$OUT"
  CUDA_VISIBLE_DEVICES="${gpu}" nohup python isaaclab_fpo/scripts/train.py \
    --task "$TASK" \
    --headless \
    --disable_fabric \
    --num_envs "$NUM_ENVS" \
    --fpo_variant "$variant" \
    --max_iterations "$MAX_ITERS" \
    --run_name "${STAMP}_${variant}" \
    agent.device=cuda:0 \
    agent.enable_post_training_eval=false \
    > "${log_file}" 2>&1 &
  local pid=$!
  echo "${pid} ${gpu} ${variant}" >> "${LOG_DIR}/pids.txt"
  echo "  PID=${pid}" | tee -a "$OUT"
}

poll_pid() {
  local pid="$1"
  local label="$2"
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
  done
  echo "[$(date '+%F %T')] ${label} (PID ${pid}) finished." | tee -a "$OUT"
}

wait_baseline() {
  local pid
  pid="$(pgrep -f 'train\.py.*Isaac-Velocity-Flat-Spot-v0.*11-29-52_spot' | head -1 || true)"
  if [[ -z "$pid" ]]; then
    pid="$(pgrep -f 'train\.py.*Isaac-Velocity-Flat-Spot-v0.*run_name.*_spot' | head -1 || true)"
  fi
  if [[ -n "$pid" ]]; then
    echo "[$(date '+%F %T')] Waiting for Spot baseline (PID ${pid})..." | tee -a "$OUT"
    poll_pid "$pid" "Spot baseline"
  else
    echo "[$(date '+%F %T')] No Spot baseline process found; assuming GPU0 is free." | tee -a "$OUT"
  fi
}

wait_eval() {
  if [[ ! -f "$EVAL_PID_FILE" ]]; then
    return
  fi
  local eval_pid
  eval_pid="$(cat "$EVAL_PID_FILE")"
  # Resolve to eval bash script if pid file points at a wrapper shell.
  local child
  child="$(pgrep -P "$eval_pid" -f 'eval_reflow_2500_final\.sh' | head -1 || true)"
  if [[ -n "$child" ]]; then
    eval_pid="$child"
  fi
  if kill -0 "$eval_pid" 2>/dev/null; then
    echo "[$(date '+%F %T')] Waiting for standalone Go2 eval (PID ${eval_pid})..." | tee -a "$OUT"
    poll_pid "$eval_pid" "standalone Go2 eval"
  fi
}

: > "${LOG_DIR}/pids.txt"
echo $$ > "${LAUNCHER_PID_FILE}"
echo "[$(date '+%F %T')] Spot reflow launcher started (PID $$)." | tee -a "$OUT"

wait_baseline
launch 0 reward_aware
PID_RA=$!

wait_eval
launch 2 adaptive_compute
PID_AC=$!

poll_pid "$PID_RA" "Spot reward_aware"
poll_pid "$PID_AC" "Spot adaptive_compute"

launch 0 fpo_operator
PID_FPO=$!
launch 2 theory
PID_TH=$!

poll_pid "$PID_FPO" "Spot fpo_operator"
poll_pid "$PID_TH" "Spot theory"

echo "[$(date '+%F %T')] All Spot reflow idea jobs finished. logs=${LOG_DIR}" | tee -a "$OUT"
