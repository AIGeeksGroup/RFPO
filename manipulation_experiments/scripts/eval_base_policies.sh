#!/usr/bin/env bash
#
# eval_base_policies.sh
#
# Evaluates all 5 pretrained BC base policies with both zero-sampling
# (deterministic) and random-sampling (stochastic) inference.
#
# Total: 5 tasks x 2 sampling modes = 10 evaluation runs
# Each evaluation runs 200 episodes across 30 parallel environments.
#
# Usage:
#   bash scripts/eval_base_policies.sh              # execute all evaluations
#   DRY_RUN=1 bash scripts/eval_base_policies.sh    # print commands only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

DRY_RUN="${DRY_RUN:-0}"

cd "$REPO_DIR"

# ---------------------------------------------------------------------------
# Helper: run or print a command
# ---------------------------------------------------------------------------
run_cmd() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "$@"
    echo ""
  else
    echo "=== Running: $@ ==="
    "$@"
  fi
}

# ---------------------------------------------------------------------------
# Base policy definitions:
#   run_id  wandb_project  checkpoint_step  eval_env
# ---------------------------------------------------------------------------
BASE_POLICIES=(
  "95j3noe4  flow-bc  step_1000    Can"
  "trc7rbt0  flow-bc  step_110000  Square"
  "lainyisy  flow-bc  step_10000   TwoArmBoxCleanup"
  "ri0w9j39  flow-bc  step_20000   TwoArmLiftTray"
  "6vqrn614  flow-bc  step_10000   TwoArmThreading"
)

# Shared evaluation parameters
EVAL_ARGS=(
  --eval_num_episodes 200
  --eval-num-envs 30
  --load-ema True
)

###########################################################################
# Zero-sampling evaluation (deterministic inference, default)
###########################################################################
echo "====================================================================="
echo "  Base Policy Evaluation: Zero Sampling (5 runs)"
echo "====================================================================="

for policy_line in "${BASE_POLICIES[@]}"; do
  read -r run_id wandb_proj ckpt_step eval_env <<< "$policy_line"

  run_cmd python eval_checkpoint.py \
    --wandb_run_id "$run_id" \
    --wandb_project "$wandb_proj" \
    --checkpoint_step "$ckpt_step" \
    --eval_env "$eval_env" \
    "${EVAL_ARGS[@]}"
done

###########################################################################
# Random-sampling evaluation (stochastic inference)
###########################################################################
echo "====================================================================="
echo "  Base Policy Evaluation: Random Sampling (5 runs)"
echo "====================================================================="

for policy_line in "${BASE_POLICIES[@]}"; do
  read -r run_id wandb_proj ckpt_step eval_env <<< "$policy_line"

  run_cmd python eval_checkpoint.py \
    --zero-sampling False \
    --wandb_run_id "$run_id" \
    --wandb_project "$wandb_proj" \
    --checkpoint_step "$ckpt_step" \
    --eval_env "$eval_env" \
    "${EVAL_ARGS[@]}"
done

echo "====================================================================="
echo "  Base policy evaluation complete (10 runs total)"
echo "====================================================================="
