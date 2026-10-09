#!/usr/bin/env bash
# Standalone PostEval-style zero/random eval on Go2 reflow_ideas_2500 final checkpoints.
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source source_env.sh

export OMNI_KIT_ACCEPT_EULA=YES
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export TMPDIR="${TMPDIR:-/nix/plsy/tmp/tmpdir}"
mkdir -p "$TMPDIR"

GPU="${GPU:-2}"
LOG_DIR="${ROOT}/logs/isaaclab_fpo/reflow_ideas_2500"
mkdir -p "$LOG_DIR"
SUMMARY="${LOG_DIR}/standalone_eval_summary.txt"
OUT="${LOG_DIR}/standalone_eval.out"

TASK="Isaac-Velocity-Flat-Unitree-Go2-v0"
BASE="${ROOT}/logs/isaaclab_fpo"

declare -a JOBS=(
  "baseline|baseline|${BASE}/unitree_go2_flat_flow/2026-08-21_20-43-35_2026-08-21_20-43-28_baseline/model_2499.pt"
  "reward_aware|reward_aware|${BASE}/unitree_go2_reflow_reward_aware/2026-08-21_20-43-35_2026-08-21_20-43-28_reward_aware/model_2499.pt"
  "adaptive_compute|adaptive_compute|${BASE}/unitree_go2_reflow_adaptive_compute/2026-08-21_20-43-35_2026-08-21_20-43-28_adaptive_compute/model_2499.pt"
)

echo "[$(date '+%F %T')] Standalone eval on GPU ${GPU} -> ${SUMMARY}" | tee "$OUT"
: > "$SUMMARY"

for job in "${JOBS[@]}"; do
  IFS='|' read -r variant label ckpt <<< "$job"
  log_file="${LOG_DIR}/eval_${label}_standalone.log"
  echo "=== Evaluating ${label} (${variant}) ===" | tee -a "$SUMMARY" "$OUT"
  echo "checkpoint: ${ckpt}" | tee -a "$SUMMARY" "$OUT"

  if [[ ! -f "$ckpt" ]]; then
    echo "FAILED: missing checkpoint ${ckpt}" | tee -a "$SUMMARY" "$OUT"
    exit 1
  fi

  set +e
  CUDA_VISIBLE_DEVICES="${GPU}" python -u isaaclab_fpo/scripts/eval_sampling_steps.py \
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
    echo "FAILED rc=${rc} for ${label}. Tail:" | tee -a "$SUMMARY" "$OUT"
    tail -n 40 "${log_file}" | tee -a "$SUMMARY" "$OUT"
    exit "$rc"
  fi

  if ! grep -q "SUMMARY TABLE" "${log_file}"; then
    echo "FAILED: no SUMMARY TABLE in ${log_file}" | tee -a "$SUMMARY" "$OUT"
    tail -n 40 "${log_file}" | tee -a "$SUMMARY" "$OUT"
    exit 1
  fi

  echo "--- results ---" | tee -a "$SUMMARY" "$OUT"
  grep -E "steps= *64|SUMMARY TABLE|DROP FROM|model|zero|random" "${log_file}" | tee -a "$SUMMARY" "$OUT"
  echo | tee -a "$SUMMARY" "$OUT"
done

echo "[$(date '+%F %T')] All standalone evals OK." | tee -a "$SUMMARY" "$OUT"
echo "Summary: ${SUMMARY}"
