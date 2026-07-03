#!/usr/bin/env bash
set -euo pipefail

ROOT="${CUIHU_ROOT:-/share/project/liyanjun/rongshanyu}"
REPO="${ROOT}/vividex_sapien_server_upload_fpo_control_full"
LOG_ROOT="${ROOT}/logs"
RESULT_ROOT="${ROOT}/results/vividex_fpo_runtime/results/state_baseline"
STAMP="$(date +%Y%m%d_%H%M)"
OUT_ROOT="${ROOT}/render_outputs/rm75_v35_best_6s_${STAMP}"
PY="${ROOT}/envs/vividex/bin/python"

PID_FILES=(
  "${LOG_ROOT}/rm75_rfpo_v35_gpu0.pid"
  "${LOG_ROOT}/rm75_rfpo_v35_gpu1.pid"
  "${LOG_ROOT}/rm75_rfpo_v35_gpu2.pid"
  "${LOG_ROOT}/rm75_rfpo_v35_gpu3.pid"
)
RUNS=(
  "rm75_rfpo_v35a_palmfix_lower_lift_v28a_last_3m_gpu0"
  "rm75_rfpo_v35b_palmfix_wrap_lock_v28b_best_3m_gpu1"
  "rm75_rfpo_v35c_palmfix_reseat_lift_v28a_last_3m_gpu2"
  "rm75_rfpo_v35d_palmfix_low_clamp_v28b_best_3m_gpu3"
)

mkdir -p "${LOG_ROOT}"
echo "[rm75-v35-render] waiting for v35 pid files"
while true; do
  alive=0
  missing=0
  for pid_file in "${PID_FILES[@]}"; do
    if [[ ! -f "${pid_file}" ]]; then
      missing=$((missing + 1))
      alive=$((alive + 1))
      continue
    fi
    pid="$(cat "${pid_file}" || true)"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      alive=$((alive + 1))
    fi
  done
  if [[ "${alive}" -eq 0 ]]; then
    break
  fi
  echo "[rm75-v35-render] waiting: alive_or_missing=${alive}, missing_pid_files=${missing}"
  sleep 120
done

cd "${REPO}"
export VIVIDEX_RM75_URDF_OVERRIDE="${REPO}/assets/robot/rm75_inspire_right/urdf/rm75_inspire_hand_right_driven12.urdf"
unset VIVIDEX_HEADLESS_NO_RENDER
mkdir -p "${OUT_ROOT}"

i=1
for run in "${RUNS[@]}"; do
  ckpt="${RESULT_ROOT}/${run}/models/best.pt"
  if [[ ! -f "${ckpt}" ]]; then
    ckpt="${RESULT_ROOT}/${run}/models/last.pt"
  fi
  if [[ ! -f "${ckpt}" ]]; then
    ckpt="$(find "${RESULT_ROOT}/${run}/models" -maxdepth 1 -type f -name "fpo_step_*.pt" 2>/dev/null | sort -V | tail -1 || true)"
  fi
  if [[ -z "${ckpt}" || ! -f "${ckpt}" ]]; then
    echo "[rm75-v35-render] missing checkpoint for ${run}" >&2
    i=$((i + 1))
    continue
  fi
  out="${OUT_ROOT}/$(printf "%02d" "${i}")_${run}_best6s"
  mkdir -p "${out}"
  echo "[rm75-v35-render] rendering ${run}: ${ckpt} -> ${out}"
  CUDA_VISIBLE_DEVICES=0 "${PY}" tools/render_rfpo_rollouts.py \
    --checkpoint "${ckpt}" \
    --out "${out}" \
    --episodes 1 --fps 20 --max-steps 120 --duration-sec 6 --stage 2
  i=$((i + 1))
done

echo "[rm75-v35-render] done ${OUT_ROOT}"
