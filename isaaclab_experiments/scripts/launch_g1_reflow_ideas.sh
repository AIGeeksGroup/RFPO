#!/usr/bin/env bash
# Launch G1 reflow idea experiments (2000 iterations each).
# Portable: set GPUs via env; no waits for other jobs on this box.
#
# Default: four idea variants on GPU0–3. Override freely for a smaller box:
#   GPUS="0,1" VARIANTS="reward_aware,adaptive_compute" bash scripts/launch_g1_reflow_ideas.sh
#   # or sequential waves on two GPUs:
#   GPUS="0,1" VARIANTS="reward_aware,adaptive_compute,fpo_operator,theory" \
#     LAUNCH_MODE=waves bash scripts/launch_g1_reflow_ideas.sh
#
# Also useful single-GPU:
#   GPUS="0" VARIANTS="baseline,reflow,reward_aware,adaptive_compute,fpo_operator,theory,all_ideas" \
#     LAUNCH_MODE=serial bash scripts/launch_g1_reflow_ideas.sh
#
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source source_env.sh

export OMNI_KIT_ACCEPT_EULA=YES
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export TMPDIR="${TMPDIR:-${ROOT}/../tmp/tmpdir}"
mkdir -p "$TMPDIR"

TASK="Isaac-Velocity-Flat-G1-v0"
STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_DIR="${ROOT}/logs/isaaclab_fpo/g1_reflow_ideas_2000"
mkdir -p "$LOG_DIR"
echo "$STAMP" > "${LOG_DIR}/stamp.txt"
OUT="${LOG_DIR}/launch.out"
LAUNCHER_PID_FILE="${LOG_DIR}/launcher.pid"

NUM_ENVS="${NUM_ENVS:-8192}"
MAX_ITERS="${MAX_ITERS:-2000}"
GPUS="${GPUS:-0,1,2,3}"
VARIANTS="${VARIANTS:-reward_aware,adaptive_compute,fpo_operator,theory}"
# parallel | waves (pair-wise on first two GPUs) | serial (one at a time on first GPU)
LAUNCH_MODE="${LAUNCH_MODE:-parallel}"

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
IFS=',' read -r -a VARIANT_ARR <<< "$VARIANTS"

launch() {
  local gpu="$1"
  local variant="$2"
  local log_file="${LOG_DIR}/${variant}.log"
  echo "[$(date '+%F %T')] Launching G1 ${variant} on GPU ${gpu} (num_envs=${NUM_ENVS}, max_iters=${MAX_ITERS}) -> ${log_file}" | tee -a "$OUT"
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
  LAST_PID="$pid"
}

poll_pid() {
  local pid="$1"
  local label="$2"
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
  done
  echo "[$(date '+%F %T')] ${label} (PID ${pid}) finished." | tee -a "$OUT"
}

: > "${LOG_DIR}/pids.txt"
echo $$ > "${LAUNCHER_PID_FILE}"
echo "[$(date '+%F %T')] G1 reflow launcher started (PID $$, mode=${LAUNCH_MODE})." | tee -a "$OUT"
echo "  GPUS=${GPUS}  VARIANTS=${VARIANTS}  NUM_ENVS=${NUM_ENVS}  MAX_ITERS=${MAX_ITERS}" | tee -a "$OUT"

case "$LAUNCH_MODE" in
  parallel)
    if (( ${#VARIANT_ARR[@]} > ${#GPU_ARR[@]} )); then
      echo "ERROR: more variants (${#VARIANT_ARR[@]}) than GPUs (${#GPU_ARR[@]}) for LAUNCH_MODE=parallel." >&2
      echo "Use LAUNCH_MODE=waves or serial, or pass fewer VARIANTS." >&2
      exit 1
    fi
    pids=()
    labels=()
    for i in "${!VARIANT_ARR[@]}"; do
      launch "${GPU_ARR[$i]}" "${VARIANT_ARR[$i]}"
      pids+=("$LAST_PID")
      labels+=("G1 ${VARIANT_ARR[$i]}")
    done
    for i in "${!pids[@]}"; do
      poll_pid "${pids[$i]}" "${labels[$i]}"
    done
    ;;
  waves)
    # Run in pairs on the first two GPUs (Spot-style).
    if (( ${#GPU_ARR[@]} < 2 )); then
      echo "ERROR: LAUNCH_MODE=waves needs at least 2 GPUs in GPUS." >&2
      exit 1
    fi
    i=0
    while (( i < ${#VARIANT_ARR[@]} )); do
      launch "${GPU_ARR[0]}" "${VARIANT_ARR[$i]}"
      pid_a="$LAST_PID"
      label_a="G1 ${VARIANT_ARR[$i]}"
      i=$((i + 1))
      if (( i < ${#VARIANT_ARR[@]} )); then
        launch "${GPU_ARR[1]}" "${VARIANT_ARR[$i]}"
        pid_b="$LAST_PID"
        label_b="G1 ${VARIANT_ARR[$i]}"
        i=$((i + 1))
        poll_pid "$pid_a" "$label_a"
        poll_pid "$pid_b" "$label_b"
      else
        poll_pid "$pid_a" "$label_a"
      fi
    done
    ;;
  serial)
    for variant in "${VARIANT_ARR[@]}"; do
      launch "${GPU_ARR[0]}" "$variant"
      poll_pid "$LAST_PID" "G1 ${variant}"
    done
    ;;
  *)
    echo "Unknown LAUNCH_MODE=${LAUNCH_MODE} (use parallel|waves|serial)" >&2
    exit 1
    ;;
esac

echo "[$(date '+%F %T')] All G1 reflow idea jobs finished. logs=${LOG_DIR}" | tee -a "$OUT"
