#!/usr/bin/env bash
# Sweep flow sampling steps [64,32,16,8,4,1] on final ckpts (zero + random).
#
# Usage:
#   # Go2 (default)
#   GPU=0 bash scripts/eval_sampling_steps_sweep.sh
#
#   # Spot (skips missing theory unless REQUIRE_ALL=1)
#   ROBOT=spot GPU=3 bash scripts/eval_sampling_steps_sweep.sh
#
#   VARIANTS="baseline reward_aware" GPU=0 bash scripts/eval_sampling_steps_sweep.sh
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source source_env.sh

export OMNI_KIT_ACCEPT_EULA=YES
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export TMPDIR="${TMPDIR:-/nix/plsy/tmp/tmpdir}"
mkdir -p "$TMPDIR"

ROBOT="${ROBOT:-go2}"
GPU="${GPU:-0}"
REQUIRE_ALL="${REQUIRE_ALL:-0}"
STEPS="${STEPS:-64 32 16 8 4 1}"
NUM_ENVS="${NUM_ENVS:-2048}"

BASE="${ROOT}/logs/isaaclab_fpo"

if [[ "$ROBOT" == "go2" ]]; then
  TASK="Isaac-Velocity-Flat-Unitree-Go2-v0"
  LOG_DIR="${BASE}/reflow_ideas_2500"
  SUMMARY="${SUMMARY:-${LOG_DIR}/step_sweep_summary.txt}"
  OUT="${OUT:-${LOG_DIR}/step_sweep.out}"
  STAMP="$(cat "${LOG_DIR}/stamp.txt" 2>/dev/null || echo "2026-08-21_20-43-28")"
  resolve_ckpt() {
    local variant="$1"
    case "$variant" in
      baseline)
        echo "${BASE}/unitree_go2_flat_flow/2026-08-21_20-43-35_${STAMP}_baseline/model_2499.pt"
        ;;
      reward_aware)
        echo "${BASE}/unitree_go2_reflow_reward_aware/2026-08-21_20-43-35_${STAMP}_reward_aware/model_2499.pt"
        ;;
      adaptive_compute)
        echo "${BASE}/unitree_go2_reflow_adaptive_compute/2026-08-21_20-43-35_${STAMP}_adaptive_compute/model_2499.pt"
        ;;
      theory)
        ls -1d "${BASE}"/unitree_go2_reflow_theory/*"${STAMP}"*_theory/model_2499.pt 2>/dev/null | tail -1
        ;;
      reflow)
        ls -1d "${BASE}"/unitree_go2_flat_flow_reflow/*"${STAMP}"*_reflow/model_2499.pt 2>/dev/null | tail -1
        ;;
      all_ideas)
        ls -1d "${BASE}"/unitree_go2_reflow_all_ideas/*"${STAMP}"*_all_ideas/model_2499.pt 2>/dev/null | tail -1
        ;;
      *)
        return 1
        ;;
    esac
  }
  read -r -a VARIANT_LIST <<< "${VARIANTS:-baseline reward_aware adaptive_compute theory}"
elif [[ "$ROBOT" == "spot" ]]; then
  TASK="Isaac-Velocity-Flat-Spot-v0"
  LOG_DIR="${BASE}/spot_reflow_ideas_1500"
  SUMMARY="${SUMMARY:-${LOG_DIR}/step_sweep_summary.txt}"
  OUT="${OUT:-${LOG_DIR}/step_sweep.out}"
  STAMP="$(cat "${LOG_DIR}/stamp.txt" 2>/dev/null || echo "2026-08-22_19-14-14")"
  resolve_ckpt() {
    local variant="$1"
    case "$variant" in
      baseline)
        echo "${BASE}/spot_flat_flow/2026-08-22_11-29-59_2026-08-22_11-29-52_spot/model_1499.pt"
        ;;
      reward_aware)
        ls -1d "${BASE}"/spot_reflow_reward_aware/*"${STAMP}"*_reward_aware/model_1499.pt 2>/dev/null | tail -1
        ;;
      adaptive_compute)
        ls -1d "${BASE}"/spot_reflow_adaptive_compute/*"${STAMP}"*_adaptive_compute/model_1499.pt 2>/dev/null | tail -1
        ;;
      theory)
        ls -1d "${BASE}"/spot_reflow_theory/*"${STAMP}"*_theory/model_1499.pt 2>/dev/null | tail -1
        ;;
      *)
        return 1
        ;;
    esac
  }
  read -r -a VARIANT_LIST <<< "${VARIANTS:-baseline reward_aware adaptive_compute theory}"
else
  echo "Unknown ROBOT=${ROBOT} (use go2|spot)" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
APPEND="${APPEND:-0}"
echo "[$(date '+%F %T')] ${ROBOT} step sweep on GPU ${GPU} steps=[${STEPS}] -> ${SUMMARY}" | tee -a "$OUT"
if [[ "$APPEND" != "1" ]]; then
  : > "$SUMMARY"
fi
{
  echo "[$(date '+%F %T')] ${ROBOT} step sweep on GPU ${GPU} steps=[${STEPS}] variants=[${VARIANT_LIST[*]}]"
} | tee -a "$SUMMARY"

for variant in "${VARIANT_LIST[@]}"; do
  ckpt="$(resolve_ckpt "$variant" || true)"
  log_file="${LOG_DIR}/eval_${variant}_step_sweep.log"

  echo "=== Evaluating ${variant} ===" | tee -a "$SUMMARY" "$OUT"
  echo "checkpoint: ${ckpt:-MISSING}" | tee -a "$SUMMARY" "$OUT"

  if [[ -z "$ckpt" || ! -f "$ckpt" ]]; then
    msg="MISSING checkpoint for ${variant}"
    if [[ "$REQUIRE_ALL" == "1" ]]; then
      echo "FAILED: ${msg}" | tee -a "$SUMMARY" "$OUT"
      exit 1
    fi
    echo "SKIP: ${msg}" | tee -a "$SUMMARY" "$OUT"
    echo | tee -a "$SUMMARY" "$OUT"
    continue
  fi

  set +e
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="${GPU}" python -u isaaclab_fpo/scripts/eval_sampling_steps.py \
    --task "$TASK" \
    --headless \
    --disable_fabric \
    --device cuda:0 \
    --num_envs "${NUM_ENVS}" \
    --eval_episodes 10 \
    --sampling_steps ${STEPS} \
    --eval_modes zero random \
    --fpo_variant "${variant}" \
    --model "${variant}=${ckpt}" \
    > "${log_file}" 2>&1
  rc=$?
  set -e

  if [[ $rc -ne 0 ]]; then
    echo "FAILED rc=${rc} for ${variant}. Tail:" | tee -a "$SUMMARY" "$OUT"
    tail -n 40 "${log_file}" | tee -a "$SUMMARY" "$OUT"
    exit "$rc"
  fi

  if ! grep -q "SUMMARY TABLE" "${log_file}"; then
    echo "FAILED: no SUMMARY TABLE in ${log_file}" | tee -a "$SUMMARY" "$OUT"
    tail -n 40 "${log_file}" | tee -a "$SUMMARY" "$OUT"
    exit 1
  fi

  echo "--- results ---" | tee -a "$SUMMARY" "$OUT"
  grep -E "steps= *[0-9]+|SUMMARY TABLE|DROP FROM|model|zero|random" "${log_file}" | tee -a "$SUMMARY" "$OUT"
  echo | tee -a "$SUMMARY" "$OUT"
done

echo "[$(date '+%F %T')] ${ROBOT} step sweep finished." | tee -a "$SUMMARY" "$OUT"
echo "Summary: ${SUMMARY}"
