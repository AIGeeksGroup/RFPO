#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

RESULT_ROOT="$RUNTIME_ROOT/results/refpo_fixed_batch_audit"
CONTROL_JSON="$RESULT_ROOT/control.json"
CANDIDATE_JSON="$RESULT_ROOT/candidate.json"
ANALYSIS_JSON="$RESULT_ROOT/analysis.json"

for path in "$CONTROL_JSON" "$CANDIDATE_JSON" "$ANALYSIS_JSON"; do
  if [[ -e "$path" ]]; then
    echo "Refusing to overwrite existing H61 artifact: $path" >&2
    exit 2
  fi
done

mkdir -p "$RESULT_ROOT" "$RUNTIME_ROOT/logs"

CONDITION=control MASTER_PORT=29661 \
  bash "$SCRIPT_DIR/run_can_refpo_audit.sh" \
  2>&1 | tee "$RUNTIME_ROOT/logs/can_h61_refpo_control.log"

CONDITION=candidate MASTER_PORT=29662 \
  bash "$SCRIPT_DIR/run_can_refpo_audit.sh" \
  2>&1 | tee "$RUNTIME_ROOT/logs/can_h61_refpo_candidate.log"

cd "$PROJECT_ROOT"
python research_scripts/analyze_refpo_audit.py \
  --control "$CONTROL_JSON" \
  --candidate "$CANDIDATE_JSON" \
  --output "$ANALYSIS_JSON" \
  2>&1 | tee "$RUNTIME_ROOT/logs/can_h61_refpo_analysis.log"
