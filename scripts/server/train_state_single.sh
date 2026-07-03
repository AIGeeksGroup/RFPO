#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

SEQ_NAME="${1:-}"
RUN_NAME="${2:-}"
shift 2 || true

[[ -n "${SEQ_NAME}" ]] || vividex_die "usage: bash scripts/server/train_state_single.sh <seq_name> <run_name> [hydra overrides...]"
[[ -n "${RUN_NAME}" ]] || vividex_die "usage: bash scripts/server/train_state_single.sh <seq_name> <run_name> [hydra overrides...]"

vividex_setup_runtime
vividex_activate_env
vividex_print_context

cd "${VIVIDEX_REPO_ROOT}"

RUN_DIR="${VIVIDEX_RESULTS_ROOT}/state_baseline/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

DEFAULT_OVERRIDES=(
  "env.name=${SEQ_NAME}"
  "env.norm_traj=True"
  "n_envs=${VIVIDEX_N_ENVS:-8}"
  "n_eval_envs=${VIVIDEX_N_EVAL_ENVS:-2}"
  "hydra.run.dir=${RUN_DIR}"
  "wandb.project=${WANDB_PROJECT:-vividex_state_baseline}"
  "wandb.group=${WANDB_GROUP:-phase1_state_baseline}"
  "wandb.sweep_name_prefix=${WANDB_PREFIX:-baseline}"
)

CMD=(python tools/train.py)
CMD+=("${DEFAULT_OVERRIDES[@]}")

for arg in "$@"; do
  CMD+=("${arg}")
done

printf '[vividex] command: %q ' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
