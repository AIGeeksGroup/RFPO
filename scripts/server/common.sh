#!/usr/bin/env bash
set -euo pipefail

vividex_die() {
  echo "[vividex][error] $*" >&2
  exit 1
}

vividex_repo_root() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "${script_dir}/../.." && pwd
}

vividex_setup_runtime() {
  export VIVIDEX_REPO_ROOT="${VIVIDEX_REPO_ROOT:-$(vividex_repo_root)}"
  export VIVIDEX_RUNTIME_ROOT="${VIVIDEX_RUNTIME_ROOT:-${VIVIDEX_REPO_ROOT}/.server_runtime}"
  export VIVIDEX_LOG_ROOT="${VIVIDEX_LOG_ROOT:-${VIVIDEX_RUNTIME_ROOT}/logs}"
  export VIVIDEX_RESULTS_ROOT="${VIVIDEX_RESULTS_ROOT:-${VIVIDEX_RUNTIME_ROOT}/results}"
  export VIVIDEX_CACHE_ROOT="${VIVIDEX_CACHE_ROOT:-${VIVIDEX_RUNTIME_ROOT}/cache}"
  export VIVIDEX_TMP_ROOT="${VIVIDEX_TMP_ROOT:-${VIVIDEX_RUNTIME_ROOT}/tmp}"
  export VIVIDEX_WANDB_ROOT="${VIVIDEX_WANDB_ROOT:-${VIVIDEX_RUNTIME_ROOT}/wandb}"
  export HF_HOME="${HF_HOME:-${VIVIDEX_CACHE_ROOT}/huggingface}"
  export TORCH_HOME="${TORCH_HOME:-${VIVIDEX_CACHE_ROOT}/torch}"
  export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${VIVIDEX_CACHE_ROOT}/pip}"
  export MPLCONFIGDIR="${MPLCONFIGDIR:-${VIVIDEX_CACHE_ROOT}/matplotlib}"
  export WANDB_DIR="${WANDB_DIR:-${VIVIDEX_WANDB_ROOT}}"
  export TMPDIR="${TMPDIR:-${VIVIDEX_TMP_ROOT}}"
  export PYTHONPATH="${VIVIDEX_REPO_ROOT}:${PYTHONPATH:-}"
  export WANDB_MODE="${WANDB_MODE:-offline}"
  mkdir -p \
    "${VIVIDEX_LOG_ROOT}" \
    "${VIVIDEX_RESULTS_ROOT}" \
    "${HF_HOME}" \
    "${TORCH_HOME}" \
    "${PIP_CACHE_DIR}" \
    "${MPLCONFIGDIR}" \
    "${WANDB_DIR}" \
    "${TMPDIR}"
}

vividex_source_conda() {
  if [[ -n "${VIVIDEX_CONDA_SH:-}" ]]; then
    [[ -f "${VIVIDEX_CONDA_SH}" ]] || vividex_die "VIVIDEX_CONDA_SH does not exist: ${VIVIDEX_CONDA_SH}"
    # shellcheck disable=SC1090
    source "${VIVIDEX_CONDA_SH}"
    return
  fi

  command -v conda >/dev/null 2>&1 || vividex_die "conda command not found; set VIVIDEX_CONDA_SH first"
  local conda_base
  conda_base="$(conda info --base)"
  [[ -f "${conda_base}/etc/profile.d/conda.sh" ]] || vividex_die "Cannot find conda.sh under ${conda_base}"
  # shellcheck disable=SC1090
  source "${conda_base}/etc/profile.d/conda.sh"
}

vividex_activate_env() {
  [[ -n "${VIVIDEX_CONDA_ENV:-}" ]] || vividex_die "Please export VIVIDEX_CONDA_ENV before running"
  # Some conda MKL activation hooks assume these variables exist when the shell
  # runs with `set -u`. Provide safe defaults before `conda activate`.
  export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-LP64}"
  export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"
  vividex_source_conda
  conda activate "${VIVIDEX_CONDA_ENV}"
}

vividex_print_context() {
  echo "[vividex] repo root      : ${VIVIDEX_REPO_ROOT}"
  echo "[vividex] runtime root   : ${VIVIDEX_RUNTIME_ROOT}"
  echo "[vividex] results root   : ${VIVIDEX_RESULTS_ROOT}"
  echo "[vividex] log root       : ${VIVIDEX_LOG_ROOT}"
  echo "[vividex] conda env      : ${VIVIDEX_CONDA_ENV:-<unset>}"
  echo "[vividex] wandb mode     : ${WANDB_MODE}"
  echo "[vividex] hostname       : $(hostname)"
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "[vividex] slurm job      : ${SLURM_JOB_ID}"
  fi
}
