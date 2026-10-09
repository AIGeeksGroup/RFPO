#!/usr/bin/env bash
# Standalone PostEval-style zero/random eval for Spot FPO variants.
#
# Usage:
#   # Eval whatever final ckpts exist (skips missing unless REQUIRE_ALL=1)
#   GPU=3 bash scripts/eval_spot_standalone.sh
#
#   # Only baseline (already finished)
#   VARIANTS=baseline GPU=3 bash scripts/eval_spot_standalone.sh
#
# Protocol (matches Go2 eval_reflow_2500_final.sh):
#   - fresh Isaac process per checkpoint (avoids long-run Vulkan crash)
#   - 2048 envs, 1 episode/env → n=2048
#   - sampling_steps=64, modes=zero/random
#   - --fpo_variant must match how the ckpt was trained
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source source_env.sh

export OMNI_KIT_ACCEPT_EULA=YES
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export TMPDIR="${TMPDIR:-/nix/plsy/tmp/tmpdir}"
mkdir -p "$TMPDIR"

GPU="${GPU:-3}"
REQUIRE_ALL="${REQUIRE_ALL:-0}"
LOG_DIR="${ROOT}/logs/isaaclab_fpo/spot_reflow_ideas_1500"
mkdir -p "$LOG_DIR"
SUMMARY="${LOG_DIR}/standalone_eval_summary.txt"
OUT="${LOG_DIR}/standalone_eval.out"

TASK="Isaac-Velocity-Flat-Spot-v0"
BASE="${ROOT}/logs/isaaclab_fpo"
STAMP="$(cat "${LOG_DIR}/stamp.txt" 2>/dev/null || echo "2026-08-22_19-14-14")"

# Resolve globs to concrete paths (experiment dirs include timestamp prefix).
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

# Default: all four; override with VARIANTS="baseline reward_aware"
read -r -a VARIANT_LIST <<< "${VARIANTS:-baseline reward_aware adaptive_compute theory}"

echo "[$(date '+%F %T')] Spot standalone eval on GPU ${GPU} -> ${SUMMARY}" | tee "$OUT"
: > "$SUMMARY"

for variant in "${VARIANT_LIST[@]}"; do
  ckpt="$(resolve_ckpt "$variant" || true)"
  log_file="${LOG_DIR}/eval_${variant}_standalone.log"

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
  grep -E "steps= *64|SUMMARY TABLE|DROP FROM|model|zero|random" "${log_file}" | tee -a "$SUMMARY" "$OUT"
  echo | tee -a "$SUMMARY" "$OUT"
done

echo "[$(date '+%F %T')] Spot standalone eval finished." | tee -a "$SUMMARY" "$OUT"
echo "Summary: ${SUMMARY}"
