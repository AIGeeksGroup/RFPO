#!/usr/bin/env bash
# Quick reflow_loss_coef ablation on plain Go2 reflow.
# Default: GPU 1, 500 iters, 4096 envs, coef in {0.1, 0.3, 1.0}, then mini step sweep (64,1).
#
# Usage:
#   bash scripts/ablate_reflow_coef.sh
#   GPU=1 MAX_ITERS=500 NUM_ENVS=4096 COEFS="0.1 0.3 1.0" bash scripts/ablate_reflow_coef.sh
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source source_env.sh

export OMNI_KIT_ACCEPT_EULA=YES
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export TMPDIR="${TMPDIR:-/nix/plsy/tmp/tmpdir}"
mkdir -p "$TMPDIR"

TASK="Isaac-Velocity-Flat-Unitree-Go2-v0"
GPU="${GPU:-1}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERS="${MAX_ITERS:-500}"
EVAL_ENVS="${EVAL_ENVS:-2048}"
read -r -a COEF_LIST <<< "${COEFS:-0.1 0.3 1.0}"

STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_DIR="${ROOT}/logs/isaaclab_fpo/reflow_coef_ablation_${MAX_ITERS}"
mkdir -p "$LOG_DIR"
echo "$STAMP" > "${LOG_DIR}/stamp.txt"
OUT="${LOG_DIR}/launch.out"
SUMMARY="${LOG_DIR}/mini_step_sweep_summary.txt"
: > "$SUMMARY"

coef_tag() {
  # 0.1 -> 0p1, 1.0 -> 1p0
  echo "$1" | sed 's/\./p/g'
}

echo "[$(date '+%F %T')] reflow_loss_coef ablation on GPU ${GPU}" | tee -a "$OUT"
echo "  iters=${MAX_ITERS} envs=${NUM_ENVS} coefs=${COEF_LIST[*]} stamp=${STAMP}" | tee -a "$OUT"

for coef in "${COEF_LIST[@]}"; do
  tag="$(coef_tag "$coef")"
  run_name="${STAMP}_reflow_coef${tag}"
  train_log="${LOG_DIR}/train_coef${tag}.log"
  eval_log="${LOG_DIR}/eval_coef${tag}.log"

  echo "[$(date '+%F %T')] TRAIN coef=${coef} -> ${train_log}" | tee -a "$OUT"
  CUDA_VISIBLE_DEVICES="${GPU}" python isaaclab_fpo/scripts/train.py \
    --task "$TASK" \
    --headless \
    --disable_fabric \
    --num_envs "$NUM_ENVS" \
    --fpo_variant reflow \
    --max_iterations "$MAX_ITERS" \
    --run_name "$run_name" \
    agent.device=cuda:0 \
    agent.enable_post_training_eval=false \
    agent.algorithm.reflow_loss_coef="$coef" \
    > "${train_log}" 2>&1

  ckpt="$(ls -1d "${ROOT}/logs/isaaclab_fpo"/unitree_go2_flat_flow_reflow/*"${run_name}"/model_$((MAX_ITERS - 1)).pt 2>/dev/null | tail -1 || true)"
  if [[ -z "$ckpt" || ! -f "$ckpt" ]]; then
    echo "[$(date '+%F %T')] WARN: missing ckpt for coef=${coef} (run=${run_name})" | tee -a "$OUT"
    continue
  fi

  echo "[$(date '+%F %T')] EVAL coef=${coef} ckpt=${ckpt}" | tee -a "$OUT"
  {
    echo "=== Evaluating reflow coef=${coef} ==="
    echo "checkpoint: ${ckpt}"
    CUDA_VISIBLE_DEVICES="${GPU}" python -u isaaclab_fpo/scripts/eval_sampling_steps.py \
      --task "$TASK" \
      --headless \
      --disable_fabric \
      --device cuda:0 \
      --num_envs "$EVAL_ENVS" \
      --fpo_variant reflow \
      --model "coef${tag}=${ckpt}" \
      --sampling_steps 64 1 \
      --eval_modes zero random \
      --eval_episodes 10
  } > "${eval_log}" 2>&1

  {
    echo ""
    echo "=== coef=${coef} ==="
    echo "checkpoint: ${ckpt}"
    grep -E 'steps=|SUMMARY|DROP|model ' "${eval_log}" || true
  } | tee -a "$SUMMARY" >> "$OUT"
done

echo "[$(date '+%F %T')] Ablation finished. summary=${SUMMARY}" | tee -a "$OUT"
