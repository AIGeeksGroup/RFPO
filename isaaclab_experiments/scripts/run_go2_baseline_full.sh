#!/usr/bin/env bash
# Go2 full baseline: 16384 envs, 2500 iters (main comparison budget), then 64/1 eval.
# Usage: GPU=3 bash scripts/run_go2_baseline_full.sh
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source source_env.sh

export OMNI_KIT_ACCEPT_EULA=YES
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export TMPDIR="${TMPDIR:-/dev/shm/_go2_baseline_full_tmp}"
export PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR"

TASK="Isaac-Velocity-Flat-Unitree-Go2-v0"
GPU="${GPU:-3}"
NUM_ENVS="${NUM_ENVS:-16384}"
MAX_ITERS="${MAX_ITERS:-2500}"
EVAL_ENVS="${EVAL_ENVS:-2048}"
VARIANT="baseline"
EXP_NAME="unitree_go2_flat_flow"

STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
WORKDIR="/dev/shm/go2_baseline_full"
WS_LOG_DIR="${ROOT}/logs/isaaclab_fpo/go2_baseline_full_${MAX_ITERS}"
LAUNCH_LOG_DIR="${WORKDIR}/launch_logs"
mkdir -p "$WORKDIR" "$LAUNCH_LOG_DIR" "$WS_LOG_DIR"
echo "$STAMP" > "${LAUNCH_LOG_DIR}/stamp.txt"
echo "$STAMP" > "${WS_LOG_DIR}/stamp.txt"
OUT="${LAUNCH_LOG_DIR}/launch.out"
TRAIN_LOG="${LAUNCH_LOG_DIR}/train.log"
EVAL_LOG="${LAUNCH_LOG_DIR}/eval.log"
run_name="${STAMP}_${VARIANT}"

echo "[$(date '+%F %T')] Go2 ${VARIANT} FULL GPU=${GPU} envs=${NUM_ENVS} iters=${MAX_ITERS}" | tee -a "$OUT"

sync_loop() {
  local src_root="${WORKDIR}/logs/isaaclab_fpo/${EXP_NAME}"
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

cd "$WORKDIR"
CUDA_VISIBLE_DEVICES="${GPU}" python "${ROOT}/isaaclab_fpo/scripts/train.py" \
  --task "$TASK" \
  --headless \
  --disable_fabric \
  --num_envs "$NUM_ENVS" \
  --fpo_variant "$VARIANT" \
  --max_iterations "$MAX_ITERS" \
  --seed 42 \
  --run_name "$run_name" \
  agent.device=cuda:0 \
  agent.enable_post_training_eval=false \
  > "${TRAIN_LOG}" 2>&1
train_rc=$?
cd "$ROOT"
echo "[$(date '+%F %T')] train exit=${train_rc} log=${TRAIN_LOG}" | tee -a "$OUT"
if [[ "$train_rc" -ne 0 ]]; then
  tail -n 120 "$TRAIN_LOG" | tee -a "$OUT"
  exit "$train_rc"
fi

ckpt="$(ls -1d "${WORKDIR}/logs/isaaclab_fpo/${EXP_NAME}/"*"${run_name}"/model_$((MAX_ITERS - 1)).pt 2>/dev/null | tail -1 || true)"
echo "checkpoint=${ckpt}" | tee -a "$OUT"
if [[ -n "$ckpt" && -f "$ckpt" ]]; then
  echo "[$(date '+%F %T')] EVAL 64/1 zero+random" | tee -a "$OUT"
  cd "$WORKDIR"
  CUDA_VISIBLE_DEVICES="${GPU}" python -u "${ROOT}/isaaclab_fpo/scripts/eval_sampling_steps.py" \
    --task "$TASK" --headless --disable_fabric --device cuda:0 \
    --num_envs "$EVAL_ENVS" --fpo_variant "$VARIANT" \
    --model "baseline=${ckpt}" \
    --sampling_steps 64 1 --eval_modes zero random --eval_episodes 10 \
    > "${EVAL_LOG}" 2>&1 || true
  cd "$ROOT"
  {
    echo "=== ${VARIANT} ${MAX_ITERS} iters ${NUM_ENVS} envs ==="
    echo "checkpoint: ${ckpt}"
    grep -E 'steps=|SUMMARY|DROP|model ' "${EVAL_LOG}" || true
  } | tee "${WS_LOG_DIR}/mini_step_sweep_summary.txt" | tee -a "$OUT"
  cp -f "$EVAL_LOG" "${WS_LOG_DIR}/eval.log" 2>/dev/null || true
fi
cp -f "$TRAIN_LOG" "${WS_LOG_DIR}/train.log" 2>/dev/null || true
echo "[$(date '+%F %T')] Done. workspace logs=${WS_LOG_DIR}" | tee -a "$OUT"
