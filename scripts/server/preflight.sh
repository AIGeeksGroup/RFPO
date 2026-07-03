#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

SEQ_NAME="${1:-ycb-006_mustard_bottle-20200709-subject-01-20200709_143211}"

vividex_setup_runtime
vividex_activate_env
vividex_print_context

cd "${VIVIDEX_REPO_ROOT}"

echo "[vividex] running import checks"
python - <<'PY'
modules = [
    "hydra",
    "stable_baselines3",
    "torch",
    "wandb",
    "gym",
    "sapien",
    "trimesh",
]

loaded = {}
for name in modules:
    try:
        loaded[name] = __import__(name)
    except Exception as exc:
        raise RuntimeError(f"import failed for {name}: {exc}") from exc

print("imports ok")
print("torch", loaded["torch"].__version__)
print("cuda available", loaded["torch"].cuda.is_available())
PY

echo "[vividex] running training entry --help"
python tools/train.py --help >/dev/null

echo "[vividex] verifying sequence file"
test -f "${VIVIDEX_REPO_ROOT}/norm_trajectories/${SEQ_NAME}.npz"

echo "[vividex] preflight passed for ${SEQ_NAME}"
