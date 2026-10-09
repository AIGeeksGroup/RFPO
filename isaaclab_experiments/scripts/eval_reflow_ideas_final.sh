#!/usr/bin/env bash
# Sequential PostEval-style zero/random eval on four idea final checkpoints.
set -e

ROOT="/workspace/fpo-control/isaaclab_experiments"
cd "$ROOT"
# shellcheck disable=SC1091
source source_env.sh

LOG_DIR="${ROOT}/logs/isaaclab_fpo/reflow_ideas_launch"
mkdir -p "$LOG_DIR"
SUMMARY="${LOG_DIR}/final_ckpt_eval_summary.txt"
: > "$SUMMARY"

TASK="Isaac-Velocity-Flat-Unitree-Go2-v0"
BASE="${ROOT}/logs/isaaclab_fpo"

declare -a JOBS=(
  "reward_aware|reward_aware|${BASE}/unitree_go2_reflow_reward_aware/2026-08-20_00-14-03_2026-08-20_00-14-00_reward_aware/model_1499.pt"
  "adaptive_compute|adaptive_compute|${BASE}/unitree_go2_reflow_adaptive_compute/2026-08-20_00-14-03_2026-08-20_00-14-00_adaptive_compute/model_1499.pt"
  "fpo_operator|fpo_operator|${BASE}/unitree_go2_reflow_fpo_operator/2026-08-20_04-26-28_2026-08-20_00-14-00_fpo_operator/model_1499.pt"
  "theory|theory|${BASE}/unitree_go2_reflow_theory/2026-08-20_04-26-28_2026-08-20_00-14-00_theory/model_1499.pt"
)

for job in "${JOBS[@]}"; do
  IFS='|' read -r variant label ckpt <<< "$job"
  log_file="${LOG_DIR}/eval_${label}.log"
  echo "=== Evaluating ${label} (${variant}) ===" | tee -a "$SUMMARY"
  echo "checkpoint: ${ckpt}" | tee -a "$SUMMARY"

  # Single process on GPU0 to avoid dual-Kit startup crashes.
  set +e
  CUDA_VISIBLE_DEVICES=0 python -u isaaclab_fpo/scripts/eval_sampling_steps.py \
    --task "$TASK" \
    --headless \
    --disable_fabric \
    --device cuda:0 \
    --num_envs 2048 \
    --eval_episodes 10 \
    --sampling_steps 64 \
    --eval_modes zero random \
    --fpo_variant "${variant}" \
    --model "${label}=${ckpt}" \
    > "${log_file}" 2>&1
  rc=$?
  set -e

  if [[ $rc -ne 0 ]]; then
    echo "FAILED rc=${rc} for ${label}. Tail:" | tee -a "$SUMMARY"
    tail -n 40 "${log_file}" | tee -a "$SUMMARY"
    exit "$rc"
  fi

  if ! grep -q "SUMMARY TABLE" "${log_file}"; then
    echo "FAILED: no SUMMARY TABLE in ${log_file}" | tee -a "$SUMMARY"
    tail -n 40 "${log_file}" | tee -a "$SUMMARY"
    exit 1
  fi

  echo "--- results ---" | tee -a "$SUMMARY"
  grep -E "steps= *64|SUMMARY TABLE|DROP FROM|model|zero|random" "${log_file}" | tee -a "$SUMMARY"
  echo | tee -a "$SUMMARY"
done

echo "All four final-checkpoint evals OK." | tee -a "$SUMMARY"
echo "Summary: ${SUMMARY}"
