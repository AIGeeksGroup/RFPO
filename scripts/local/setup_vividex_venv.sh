#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV="${VIVIDEX_VENV:-${REPO}/.venv-vividex}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install uv first or put it on PATH." >&2
  exit 1
fi

cd "${REPO}"

uv venv "${VENV}" --python "${PYTHON_BIN}"
uv pip install --python "${VENV}/bin/python" pip setuptools wheel
uv pip install --python "${VENV}/bin/python" \
  torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu121
uv pip install --python "${VENV}/bin/python" --no-build-isolation -r requirements.txt
uv pip install --python "${VENV}/bin/python" pytest

"${VENV}/bin/python" - <<'PY'
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
