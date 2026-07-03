#!/usr/bin/env bash
set -euo pipefail

SEQ_NAME="${1:-}"
RUN_NAME="${2:-}"
shift 2 || true

if [[ -z "${SEQ_NAME}" || -z "${RUN_NAME}" ]]; then
  echo "usage: sbatch scripts/server/train_state_a100.slurm <seq_name> <run_name> [hydra overrides...]" >&2
  exit 1
fi

if [[ -n "${VIVIDEX_REPO_ROOT:-}" ]]; then
  REPO_ROOT="${VIVIDEX_REPO_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  echo "[vividex][error] cannot determine repo root inside sbatch job" >&2
  exit 1
fi

bash "${REPO_ROOT}/scripts/server/train_state_single.sh" "${SEQ_NAME}" "${RUN_NAME}" "$@"
