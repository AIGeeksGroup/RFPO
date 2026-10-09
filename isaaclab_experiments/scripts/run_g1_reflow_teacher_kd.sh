#!/usr/bin/env bash
# G1: eval frozen full baseline teacher, then train reflow+KD student (500-iter probe).
# Teacher: g1_baseline model_1999.pt (2000-iter, finished).
#
# Usage:
#   GPU=2 bash scripts/run_g1_reflow_teacher_kd.sh
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source source_env.sh

export OMNI_KIT_ACCEPT_EULA=YES
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export TMPDIR="${TMPDIR:-/dev/shm/_g1_teacher_kd_tmp}"
export PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR"

TASK="Isaac-Velocity-Flat-G1-v0"
GPU="${GPU:-2}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERS="${MAX_ITERS:-500}"
EVAL_ENVS="${EVAL_ENVS:-2048}"
VARIANT="reflow_teacher_kd"
EXP_NAME="g1_reflow_teacher_kd"
TEACHER="${TEACHER:-${ROOT}/logs/isaaclab_fpo/g1_flat_flow/2026-08-27_00-04-43_2026-08-27_00-02-39_g1_baseline/model_1999.pt}"

STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
WORKDIR="/dev/shm/g1_reflow_teacher_kd"
WS_LOG_DIR="${ROOT}/logs/isaaclab_fpo/g1_reflow_teacher_kd_${MAX_ITERS}"
LAUNCH_LOG_DIR="${WORKDIR}/launch_logs"
mkdir -p "$WORKDIR" "$LAUNCH_LOG_DIR" "$WS_LOG_DIR"
echo "$STAMP" > "${LAUNCH_LOG_DIR}/stamp.txt"
echo "$STAMP" > "${WS_LOG_DIR}/stamp.txt"
OUT="${LAUNCH_LOG_DIR}/launch.out"
TEACHER_EVAL_LOG="${LAUNCH_LOG_DIR}/teacher_eval.log"
TRAIN_LOG="${LAUNCH_LOG_DIR}/train.log"
EVAL_LOG="${LAUNCH_LOG_DIR}/eval.log"
run_name="${STAMP}_${VARIANT}"

echo "[$(date '+%F %T')] G1 teacher-KD GPU=${GPU} envs=${NUM_ENVS} iters=${MAX_ITERS}" | tee -a "$OUT"
echo "  teacher=${TEACHER}" | tee -a "$OUT"

cd "$WORKDIR"
echo "[$(date '+%F %T')] EVAL frozen G1 baseline teacher 64/1 zero+random" | tee -a "$OUT"
CUDA_VISIBLE_DEVICES="${GPU}" python -u "${ROOT}/isaaclab_fpo/scripts/eval_sampling_steps.py" \
  --task "$TASK" \
  --headless \
  --disable_fabric \
  --device cuda:0 \
  --num_envs "$EVAL_ENVS" \
  --fpo_variant baseline \
  --model "g1_baseline_1999=${TEACHER}" \
  --sampling_steps 64 1 \
  --eval_modes zero random \
  --eval_episodes 10 \
  > "${TEACHER_EVAL_LOG}" 2>&1 || true
{
  echo "=== G1 baseline teacher model_1999 (2000 iters) ==="
  echo "checkpoint: ${TEACHER}"
  grep -E 'steps=|SUMMARY|DROP|model ' "${TEACHER_EVAL_LOG}" || true
} | tee "${WS_LOG_DIR}/teacher_step_sweep_summary.txt" | tee -a "$OUT"
cp -f "$TEACHER_EVAL_LOG" "${WS_LOG_DIR}/teacher_eval.log" 2>/dev/null || true

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

echo "[$(date '+%F %T')] TRAIN student reflow_teacher_kd" | tee -a "$OUT"
CUDA_VISIBLE_DEVICES="${GPU}" python "${ROOT}/isaaclab_fpo/scripts/train.py" \
  --task "$TASK" \
  --headless \
  --disable_fabric \
  --num_envs "$NUM_ENVS" \
  --fpo_variant "$VARIANT" \
  --max_iterations "$MAX_ITERS" \
  --run_name "$run_name" \
  --teacher_checkpoint "$TEACHER" \
  agent.device=cuda:0 \
  agent.enable_post_training_eval=false \
  > "${TRAIN_LOG}" 2>&1
train_rc=$?
echo "[$(date '+%F %T')] train exit=${train_rc}" | tee -a "$OUT"
if [[ "$train_rc" -ne 0 ]]; then
  tail -n 80 "$TRAIN_LOG" | tee -a "$OUT"
  exit "$train_rc"
fi

ckpt="$(ls -1d "${WORKDIR}/logs/isaaclab_fpo/${EXP_NAME}/"*"${run_name}"/model_$((MAX_ITERS - 1)).pt 2>/dev/null | tail -1 || true)"
echo "checkpoint=${ckpt}" | tee -a "$OUT"

if [[ -n "$ckpt" && -f "$ckpt" ]]; then
  echo "[$(date '+%F %T')] EVAL student 64/1 zero+random" | tee -a "$OUT"
  CUDA_VISIBLE_DEVICES="${GPU}" python -u "${ROOT}/isaaclab_fpo/scripts/eval_sampling_steps.py" \
    --task "$TASK" \
    --headless \
    --disable_fabric \
    --device cuda:0 \
    --num_envs "$EVAL_ENVS" \
    --fpo_variant "$VARIANT" \
    --model "g1_teacher_kd=${ckpt}" \
    --sampling_steps 64 1 \
    --eval_modes zero random \
    --eval_episodes 10 \
    > "${EVAL_LOG}" 2>&1 || true
  {
    echo "=== G1 reflow_teacher_kd ${MAX_ITERS} iters ${NUM_ENVS} envs ==="
    echo "checkpoint: ${ckpt}"
    echo "teacher: ${TEACHER}"
    grep -E 'steps=|SUMMARY|DROP|model ' "${EVAL_LOG}" || true
  } | tee "${WS_LOG_DIR}/mini_step_sweep_summary.txt" | tee -a "$OUT"
  cp -f "$EVAL_LOG" "${WS_LOG_DIR}/eval.log" 2>/dev/null || true
fi

cd "$ROOT"
cp -f "$TRAIN_LOG" "${WS_LOG_DIR}/train.log" 2>/dev/null || true
echo "[$(date '+%F %T')] Done. workspace logs=${WS_LOG_DIR}" | tee -a "$OUT"
