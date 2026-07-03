#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_FILE="${1:-${SCRIPT_DIR}/phase1_tasks.tsv}"
shift $(( $# >= 1 ? 1 : $# ))

[[ -f "${TASK_FILE}" ]] || { echo "[vividex][error] task file not found: ${TASK_FILE}" >&2; exit 1; }

while IFS=$'\t' read -r RUN_NAME SEQ_NAME; do
  [[ -z "${RUN_NAME}" ]] && continue
  [[ "${RUN_NAME}" == \#* ]] && continue
  bash "${SCRIPT_DIR}/submit_state_job.sh" "${SEQ_NAME}" "${RUN_NAME}" "$@"
done < "${TASK_FILE}"
