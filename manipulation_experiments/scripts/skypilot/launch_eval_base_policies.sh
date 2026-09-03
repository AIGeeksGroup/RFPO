#!/bin/bash
#
# launch_eval_base_policies.sh (skypilot - remote cluster launcher)
#
# Launches base policy evaluation jobs on EC2 via SkyPilot.
# Evaluates all 5 base policies with both zero-sampling and random-sampling.
# Total: 10 evaluation runs on a single cluster.
#
# Usage:
#   bash scripts/skypilot/launch_eval_base_policies.sh
#
# Requires: SkyPilot configured with FAR-skypilot-wrapper

set -euo pipefail

# All 10 evaluation commands concatenated with &&
EVAL_CMD="python eval_checkpoint.py --wandb_run_id 95j3noe4 --wandb_project flow-bc --checkpoint_step step_1000 --eval_env Can --eval_num_episodes 200 --eval-num-envs 30 --load-ema True && \
python eval_checkpoint.py --wandb_run_id trc7rbt0 --wandb_project flow-bc --checkpoint_step step_110000 --eval_env Square --eval_num_episodes 200 --eval-num-envs 30 --load-ema True && \
python eval_checkpoint.py --wandb_run_id lainyisy --wandb_project flow-bc --checkpoint_step step_10000 --eval_env TwoArmBoxCleanup --eval_num_episodes 200 --eval-num-envs 30 --load-ema True && \
python eval_checkpoint.py --wandb_run_id ri0w9j39 --wandb_project flow-bc --checkpoint_step step_20000 --eval_env TwoArmLiftTray --eval_num_episodes 200 --eval-num-envs 30 --load-ema True && \
python eval_checkpoint.py --wandb_run_id 6vqrn614 --wandb_project flow-bc --checkpoint_step step_10000 --eval_env TwoArmThreading --eval_num_episodes 200 --eval-num-envs 30 --load-ema True && \
python eval_checkpoint.py --zero-sampling False --wandb_run_id 95j3noe4 --wandb_project flow-bc --checkpoint_step step_1000 --eval_env Can --eval_num_episodes 200 --eval-num-envs 30 --load-ema True && \
python eval_checkpoint.py --zero-sampling False --wandb_run_id trc7rbt0 --wandb_project flow-bc --checkpoint_step step_110000 --eval_env Square --eval_num_episodes 200 --eval-num-envs 30 --load-ema True && \
python eval_checkpoint.py --zero-sampling False --wandb_run_id lainyisy --wandb_project flow-bc --checkpoint_step step_10000 --eval_env TwoArmBoxCleanup --eval_num_episodes 200 --eval-num-envs 30 --load-ema True && \
python eval_checkpoint.py --zero-sampling False --wandb_run_id ri0w9j39 --wandb_project flow-bc --checkpoint_step step_20000 --eval_env TwoArmLiftTray --eval_num_episodes 200 --eval-num-envs 30 --load-ema True && \
python eval_checkpoint.py --zero-sampling False --wandb_run_id 6vqrn614 --wandb_project flow-bc --checkpoint_step step_10000 --eval_env TwoArmThreading --eval_num_episodes 200 --eval-num-envs 30 --load-ema True"

echo "Launching base policy evaluation cluster..."

SKIP_UNTRACKED=1 sky EC2:manipulation-fpo \
    --cluster "$USER-fpo-eval-base-policies" \
    --num-nodes 1 \
    --gpus=L40S:1 --cpus 16+ --memory 64+ \
    --down \
    --idle-minutes-to-autostop 60 \
    --yes \
    --detach-run \
    --env OMNI_KIT_ACCEPT_EULA=YES \
    --env HF_HUB_DOWNLOAD_TIMEOUT=240 \
    --env HF_HUB_ETAG_TIMEOUT=240 \
    --env HF_TOKEN= \
    --env ENTRYPOINT="$EVAL_CMD"

echo "=== Base policy evaluation launch complete ==="
