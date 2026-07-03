#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV="${VIVIDEX_VENV:-${REPO}/.venv-vividex}"
PYTHON="${VENV}/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing Python env: ${PYTHON}" >&2
  echo "Run scripts/local/setup_vividex_venv.sh first, or set VIVIDEX_VENV." >&2
  exit 1
fi

cd "${REPO}"
export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export VIVIDEX_HEADLESS_NO_RENDER="${VIVIDEX_HEADLESS_NO_RENDER:-1}"

SEQ="${SEQ:-ycb-006_mustard_bottle-20200709-subject-01-20200709_143211}"
RUN_ROOT="${VIVIDEX_LOCAL_RUN_ROOT:-${REPO}/.local_runs}"
ROOT="${ROOT:-${RUN_ROOT}/rm75_bc_compare_$(date +%Y%m%d_%H%M%S)}"
DATASET="${DATASET:-${ROOT}/rm75_reference_bc.npz}"
PPO_BC_DIR="${PPO_BC_DIR:-${ROOT}/ppo_bc}"
FPO_BC_DIR="${FPO_BC_DIR:-${ROOT}/flow_bc}"
PPO_BC="${PPO_BC:-${PPO_BC_DIR}/bc_ppo_last.zip}"
FPO_BC="${FPO_BC:-${FPO_BC_DIR}/flow_bc_last.pt}"
PPO_DIR="${PPO_DIR:-${ROOT}/bcppo_online}"
FPO_DIR="${FPO_DIR:-${ROOT}/bcrfpo_online}"
OBJECT_SCALE="${OBJECT_SCALE:-0.60}"

mkdir -p "${ROOT}"

if [[ ! -f "${DATASET}" || "${REGENERATE_BC_DATASET:-0}" == "1" ]]; then
  "${PYTHON}" tools/collect_rm75_reference_bc_dataset.py \
    --output "${DATASET}" \
    --seq-name "${SEQ}" \
    --robot-name rm75_inspire_right \
    --norm-traj \
    --stage "${BC_STAGE:-0}" \
    --mode approach_target \
    --native-hand-control \
    --native-apply-template-approach-pregrasp \
    --no-done-on-pregrasp-failure \
    --obj-com-done-thresh "${BC_OBJ_COM_DONE_THRESH:-0.35}" \
    --no-contact-grace-steps "${BC_NO_CONTACT_GRACE_STEPS:-260}" \
    --required-non-thumb-contacts "${BC_REQUIRED_NON_THUMB_CONTACTS:-2}" \
    --object-scale "${OBJECT_SCALE}" \
    --force-imitate-steps "${BC_FORCE_IMITATE_STEPS:-260}" \
    --num-trajs "${BC_NUM_TRAJS:-48}" \
    --max-attempts "${BC_MAX_ATTEMPTS:-512}" \
    --filter-success \
    --min-contact-steps "${BC_MIN_CONTACT_STEPS:-45}" \
    --min-stable-steps "${BC_MIN_STABLE_STEPS:-40}" \
    --min-thumb-steps "${BC_MIN_THUMB_STEPS:-45}" \
    --min-non-thumb-steps "${BC_MIN_NON_THUMB_STEPS:-45}" \
    --min-lift "${BC_MIN_LIFT:-0.08}" \
    --close-val "${BC_CLOSE_VAL:-0.30}" \
    --thumb-yaw "${BC_THUMB_YAW:-0.97}" \
    --thumb-pitch "${BC_THUMB_PITCH:-0.80}" \
    --finger-close "${BC_FINGER_CLOSE:-0.72}" \
    --thumb-delay-steps "${BC_THUMB_DELAY_STEPS:-2}" \
    --pinky-action-value "${BC_PINKY_ACTION_VALUE:--0.5}" \
    --pinky-close "${BC_PINKY_CLOSE:-0.0}" \
    --approach-delta "${BC_APPROACH_DX:--0.0204}" "${BC_APPROACH_DY:--0.1595}" "${BC_APPROACH_DZ:-0.1186}" \
    --approach-steps "${BC_APPROACH_STEPS:-14}" \
    --close-steps "${BC_CLOSE_STEPS:-30}" \
    --hold-steps "${BC_HOLD_STEPS:-36}" \
    --lift-steps "${BC_LIFT_STEPS:-140}" \
    --lift-height "${BC_LIFT_HEIGHT:-0.18}" \
    --done-on-norm-success-10 \
    --norm-success-lift-thresh "${BC_NORM_SUCCESS_LIFT_THRESH:-0.08}" \
    --score-lift-target "${BC_SCORE_LIFT_TARGET:-0.08}" \
    --target-track-max-xy-step "${BC_TARGET_TRACK_MAX_XY_STEP:-0.006}" \
    --target-track-max-z-step "${BC_TARGET_TRACK_MAX_Z_STEP:-0.006}" \
    --target-track-gain "${BC_TARGET_TRACK_GAIN:-0.75}" \
    --target-track-z-deadband "${BC_TARGET_TRACK_Z_DEADBAND:-0.002}" \
    --settle-steps "${BC_SETTLE_STEPS:-80}"
fi

if [[ ! -f "${PPO_BC}" || "${REGENERATE_PPO_BC:-0}" == "1" ]]; then
  "${PYTHON}" tools/pretrain_ppo_bc.py \
    --dataset "${DATASET}" \
    --output-dir "${PPO_BC_DIR}" \
    --seq-name "${SEQ}" \
    --robot-name rm75_inspire_right \
    --object-scale "${OBJECT_SCALE}" \
    --steps "${PPO_BC_STEPS:-50000}" \
    --batch-size "${PPO_BC_BATCH_SIZE:-1024}" \
    --learning-rate "${PPO_BC_LR:-1e-4}" \
    --save-freq "${PPO_BC_SAVE_FREQ:-25000}" \
    --seed "${SEED:-0}"
fi

if [[ ! -f "${FPO_BC}" || "${REGENERATE_FPO_BC:-0}" == "1" ]]; then
  "${PYTHON}" tools/pretrain_fpo_bc.py \
    --dataset "${DATASET}" \
    --output_dir "${FPO_BC_DIR}" \
    --steps "${FPO_BC_STEPS:-50000}" \
    --batch_size "${FPO_BC_BATCH_SIZE:-1024}" \
    --learning_rate "${FPO_BC_LR:-1e-4}" \
    --save_freq "${FPO_BC_SAVE_FREQ:-25000}" \
    --actor_hidden_dims "${FPO_ACTOR_DIMS:-[512,512]}" \
    --critic_hidden_dims "${FPO_CRITIC_DIMS:-[512,512]}" \
    --activation "${FPO_ACTIVATION:-elu}" \
    --sampling_steps "${FPO_SAMPLING_STEPS:-8}" \
    --seed "${SEED:-0}"
fi

PPO_ENV=(
  "RUN_DIR=${PPO_DIR}"
  "RUN_NAME=rm75_bcppo_compare"
  "OBJECT_SCALE=${OBJECT_SCALE}"
  "RESUME_MODEL=${PPO_BC}"
  "BC_ANCHOR_DATASET=${DATASET}"
  "BC_ANCHOR_COEF=${PPO_BC_ANCHOR_COEF:-1.0}"
  "BC_ANCHOR_MIN_COEF=${PPO_BC_ANCHOR_MIN_COEF:-0.2}"
)
RFPO_ENV=(
  "RUN_DIR=${FPO_DIR}"
  "RUN_NAME=rm75_bcrfpo_compare"
  "OBJECT_SCALE=${OBJECT_SCALE}"
  "BC_ANCHOR_DATASET=${DATASET}"
  "BC_CHECKPOINT=${FPO_BC}"
)

if [[ "${SMOKE:-0}" == "1" ]]; then
  PPO_ENV+=("SMOKE=1")
  RFPO_ENV+=("SMOKE=1")
fi

env "${PPO_ENV[@]}" scripts/local/rm75_native_ppo_train.sh
env "${RFPO_ENV[@]}" scripts/local/rm75_native_rfpo_train.sh
