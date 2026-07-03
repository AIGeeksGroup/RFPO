#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

ENV_TARGET="${1:-}"
[[ -n "${ENV_TARGET}" ]] || vividex_die "usage: bash scripts/server/setup_conda_env.sh /path/to/conda_env"

vividex_setup_runtime
vividex_source_conda

PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_VERSION="${TORCH_VERSION:-2.4.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.19.1}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.4.1}"
PYTORCH_CUDA_VERSION="${PYTORCH_CUDA_VERSION:-12.1}"
MKL_VERSION="${MKL_VERSION:-2023.1.0}"
INTEL_OPENMP_VERSION="${INTEL_OPENMP_VERSION:-2023.1.0}"

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_TARGET}"; then
  echo "[vividex] conda env already exists in env list: ${ENV_TARGET}"
elif [[ -d "${ENV_TARGET}" ]]; then
  echo "[vividex] conda env directory already exists: ${ENV_TARGET}"
else
  conda create -y -p "${ENV_TARGET}" "python=${PYTHON_VERSION}"
fi

export VIVIDEX_CONDA_ENV="${ENV_TARGET}"
vividex_activate_env

python -m pip install --upgrade pip wheel setuptools
echo "[vividex] conda channels before installs"
conda config --show channels || true

conda install --override-channels -y \
  -c pytorch -c nvidia -c defaults \
  "pytorch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}" \
  "pytorch-cuda=${PYTORCH_CUDA_VERSION}"

# Torch 2.4.1 can fail with newer MKL/OpenMP builds via:
# undefined symbol: iJIT_NotifyEvent
# Pinning to the 2023.1.0 runtime matched the locally verified setup.
conda install --override-channels -c defaults -y \
  "mkl=${MKL_VERSION}" \
  "intel-openmp=${INTEL_OPENMP_VERSION}"

TMP_REQUIREMENTS="$(mktemp)"
grep -v '^chumpy$' "${VIVIDEX_REPO_ROOT}/requirements.txt" > "${TMP_REQUIREMENTS}"
python -m pip install -r "${TMP_REQUIREMENTS}"
python -m pip install --no-build-isolation chumpy==0.70
rm -f "${TMP_REQUIREMENTS}"

echo "[vividex] validating fresh environment"
python - <<'PY'
import hydra
import stable_baselines3
import torch
import wandb
print("torch", torch.__version__)
print("hydra", hydra.__version__)
print("stable_baselines3", stable_baselines3.__version__)
print("wandb", wandb.__version__)
PY

echo "[vividex] setup complete for ${ENV_TARGET}"
