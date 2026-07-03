#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

SEQ_NAME="${1:-}"
RUN_NAME="${2:-}"
shift 2 || true

[[ -n "${SEQ_NAME}" ]] || vividex_die "usage: bash scripts/server/submit_state_job.sh <seq_name> <run_name> [hydra overrides...]"
[[ -n "${RUN_NAME}" ]] || vividex_die "usage: bash scripts/server/submit_state_job.sh <seq_name> <run_name> [hydra overrides...]"

vividex_setup_runtime

SLURM_PARTITION="${SLURM_PARTITION:-day}"
SLURM_TIME="${SLURM_TIME:-23:59:00}"
SLURM_GRES="${SLURM_GRES:-gpu:1}"
SLURM_CPUS="${SLURM_CPUS:-8}"
SLURM_MEM="${SLURM_MEM:-32G}"
LOG_PATH="${VIVIDEX_LOG_ROOT}/${RUN_NAME}_%j.out"

CMD=(
  sbatch
  -p "${SLURM_PARTITION}"
  -t "${SLURM_TIME}"
  --gres="${SLURM_GRES}"
  --cpus-per-task="${SLURM_CPUS}"
  --mem="${SLURM_MEM}"
  -J "${RUN_NAME}"
  -o "${LOG_PATH}"
  "${SCRIPT_DIR}/train_state_a100.slurm"
  "${SEQ_NAME}"
  "${RUN_NAME}"
)

for arg in "$@"; do
  CMD+=("${arg}")
done

printf '[vividex] submit: %q ' "${CMD[@]}"
printf '\n'

if [[ "${VIVIDEX_DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

"${CMD[@]}"
