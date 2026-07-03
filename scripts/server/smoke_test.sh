#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

SEQ_NAME="${1:-ycb-006_mustard_bottle-20200709-subject-01-20200709_143211}"
RUN_NAME="${2:-smoke_${SEQ_NAME##*-}}"
ROBOT_NAME="${VIVIDEX_ROBOT_NAME:-allegro_hand_ur5}"
shift $(( $# >= 2 ? 2 : $# ))

vividex_setup_runtime
vividex_activate_env
vividex_print_context

cd "${VIVIDEX_REPO_ROOT}"

echo "[vividex] step 1/2: environment bootstrap"
python scripts/server/check_state_env.py --seq-name "${SEQ_NAME}" --norm-traj --n-envs 1 --robot-name "${ROBOT_NAME}"

RUN_DIR="${VIVIDEX_RESULTS_ROOT}/smoke/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

CMD=(
  python tools/train.py
  "env.name=${SEQ_NAME}"
  "env.robot_name=${ROBOT_NAME}"
  "env.norm_traj=True"
  "agent.multi_proc=False"
  "n_envs=1"
  "n_eval_envs=1"
  "total_timesteps=32"
  "eval_freq=32"
  "eval_n_episodes=1"
  "save_freq=32"
  "restore_checkpoint_freq=16"
  "agent.params.n_steps=32"
  "agent.params.batch_size=32"
  "agent.params.n_epochs=1"
  "hydra.run.dir=${RUN_DIR}"
  "wandb.project=vividex_smoke"
  "wandb.group=smoke"
  "wandb.sweep_name_prefix=smoke"
)

for arg in "$@"; do
  CMD+=("${arg}")
done

echo "[vividex] step 2/2: tiny training smoke test"
printf '[vividex] command: %q ' "${CMD[@]}"
printf '\n'

if [[ "${VIVIDEX_DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

"${CMD[@]}"
