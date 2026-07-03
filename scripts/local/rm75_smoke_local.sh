#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV="${VIVIDEX_VENV:-${REPO}/.venv-vividex}"
PYTHON="${VENV}/bin/python"
SEQ="${SEQ:-ycb-006_mustard_bottle-20200709-subject-01-20200709_143211}"
RUN_ROOT="${VIVIDEX_LOCAL_RUN_ROOT:-${REPO}/.local_runs}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
MODE="${1:-all}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing Python env: ${PYTHON}" >&2
  echo "Run scripts/local/setup_vividex_venv.sh first." >&2
  exit 1
fi

cd "${REPO}"
mkdir -p "${RUN_ROOT}"

export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export VIVIDEX_HEADLESS_NO_RENDER="${VIVIDEX_HEADLESS_NO_RENDER:-1}"

run_import_check() {
  "${PYTHON}" - <<'PY'
import torch

mods = ["gym", "sapien", "stable_baselines3", "hydra", "wandb", "cv2", "numpy"]
for name in mods:
    module = __import__(name)
    print(name, getattr(module, "__version__", "ok"))

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device", torch.cuda.get_device_name(0))
PY
}

run_diagnose() {
  "${PYTHON}" tools/diagnose_rm75_reference_alignment.py \
    --seq-name "${SEQ}" \
    --robot-name rm75_inspire_right \
    --norm-traj
}

run_scripted() {
  "${PYTHON}" tools/scripted_rm75_grasp_smoke.py \
    --seq-name "${SEQ}" \
    --robot-name rm75_inspire_right \
    --norm-traj \
    --mode track_reference \
    --close-val "${RM75_CLOSE_VAL:-0.9}" \
    --steps "${RM75_SMOKE_STEPS:-25}"
}

run_ppo() {
  "${PYTHON}" tools/train.py \
    agent=ppo \
    agent.multi_proc=False \
    env.name="${SEQ}" \
    env.norm_traj=True \
    env.robot_name=rm75_inspire_right \
    n_envs=1 \
    n_eval_envs=1 \
    total_timesteps="${SMOKE_TOTAL_TIMESTEPS:-16}" \
    eval_freq=64 \
    eval_n_episodes=1 \
    save_freq=64 \
    restore_checkpoint_freq=64 \
    agent.params.n_steps=8 \
    agent.params.batch_size=8 \
    agent.params.n_epochs=1 \
    hydra.run.dir="${RUN_ROOT}/smoke_ppo_rm75_${RUN_TAG}"
}

run_fpo() {
  "${PYTHON}" tools/train.py \
    agent=fpo \
    agent.multi_proc=False \
    env.name="${SEQ}" \
    env.norm_traj=True \
    env.robot_name=rm75_inspire_right \
    n_envs=1 \
    n_eval_envs=1 \
    total_timesteps="${SMOKE_TOTAL_TIMESTEPS:-16}" \
    eval_freq=64 \
    eval_n_episodes=1 \
    save_freq=64 \
    restore_checkpoint_freq=64 \
    agent.params.n_steps=8 \
    agent.params.batch_size=8 \
    agent.params.n_epochs=1 \
    agent.params.n_samples_per_action=2 \
    agent.params.actor_hidden_dims='[64,64]' \
    agent.params.critic_hidden_dims='[64,64]' \
    agent.params.residual_head_hidden_dims='[32,32]' \
    hydra.run.dir="${RUN_ROOT}/smoke_fpo_rm75_${RUN_TAG}"
}

case "${MODE}" in
  import)
    run_import_check
    ;;
  diagnose)
    run_diagnose
    ;;
  scripted)
    run_scripted
    ;;
  ppo)
    run_ppo
    ;;
  fpo|rfpo)
    run_fpo
    ;;
  all)
    run_import_check
    run_diagnose
    run_scripted
    run_ppo
    run_fpo
    ;;
  *)
    echo "Usage: $0 [import|diagnose|scripted|ppo|fpo|rfpo|all]" >&2
    exit 2
    ;;
esac
