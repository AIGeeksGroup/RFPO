#!/usr/bin/env bash
# Resume the three G1 jobs that died on checkpoint write.
# GPU1: pure reflow 1150->2000
# GPU2: all_ideas_teacher_kd 200->2000
# GPU3: all_ideas_fpo 1700->2000
# Do not use `set -u` (source_env.sh reads ZSH_VERSION).
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
set +u
source source_env.sh
set -u

export OMNI_KIT_ACCEPT_EULA=YES
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export PYTHONUNBUFFERED=1

TEACHER="${ROOT}/logs/isaaclab_fpo/g1_flat_flow/2026-08-27_00-04-43_2026-08-27_00-02-39_g1_baseline/model_1999.pt"
KD_CKPT="/dev/shm/g1_all_ideas_teacher_kd/logs/isaaclab_fpo/g1_all_ideas_teacher_kd/2026-08-27_18-14-29_2026-08-27_18-13-34_all_ideas_teacher_kd/model_200.pt"
FPO_CKPT="/dev/shm/g1_resume/logs/isaaclab_fpo/g1_reflow_all_ideas_fpo/2026-08-27_18-14-21_2026-08-27_18-12_g1_all_ideas_fpo_resume1500/model_1700.pt"
REFLOW_CKPT="/dev/shm/g1_reflow_resume/model_1150.pt"

launch_one() {
  local gpu="$1"
  local variant="$2"
  local ckpt="$3"
  local max_iters="$4"
  local workdir="$5"
  local extra=()
  shift 5
  extra=("$@")

  mkdir -p "${workdir}/launch_logs" "${workdir}/tmp"
  local stamp run_name log
  stamp="$(date +%Y-%m-%d_%H-%M-%S)"
  run_name="${stamp}_${variant}_resume"
  log="${workdir}/launch_logs/${run_name}.out"

  echo "[$(date '+%F %T')] GPU${gpu} ${variant} resume ckpt=${ckpt} extra_iters=${max_iters}" | tee -a "${workdir}/launch_logs/launch.out"
  (
    cd "$workdir"
    export TMPDIR="${workdir}/tmp"
    export CUDA_VISIBLE_DEVICES="$gpu"
    python "${ROOT}/isaaclab_fpo/scripts/train.py" \
      --task Isaac-Velocity-Flat-G1-v0 \
      --headless --disable_fabric \
      --num_envs 8192 \
      --fpo_variant "$variant" \
      --max_iterations "$max_iters" \
      --seed 42 \
      --resume --checkpoint "$ckpt" \
      --run_name "$run_name" \
      agent.device=cuda:0 \
      agent.enable_post_training_eval=false \
      "${extra[@]}"
  ) >"$log" 2>&1 &
  echo $! > "${workdir}/launch_logs/train.pid"
  echo "  pid=$(cat "${workdir}/launch_logs/train.pid") log=$log"
}

# teacher-kd still needs the frozen baseline teacher
launch_one 2 all_ideas_teacher_kd "$KD_CKPT" 1800 /dev/shm/g1_all_ideas_teacher_kd \
  --teacher_checkpoint "$TEACHER"

launch_one 3 all_ideas_fpo "$FPO_CKPT" 300 /dev/shm/g1_resume

launch_one 1 reflow "$REFLOW_CKPT" 850 /dev/shm/g1_reflow_resume

sleep 2
for d in /dev/shm/g1_all_ideas_teacher_kd /dev/shm/g1_resume /dev/shm/g1_reflow_resume; do
  pid=$(cat "${d}/launch_logs/train.pid" 2>/dev/null || true)
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "ALIVE $pid $d"
  else
    echo "DEAD $d"
    tail -n 20 "${d}/launch_logs/"*.out 2>/dev/null | tail -n 20
  fi
done
