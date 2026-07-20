#!/usr/bin/env bash
set -euo pipefail
# Run with: bash scripts/nohup_train_nz100.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/train_nz100_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

nohup bash scripts/train_nz100.sh > "${LOG_FILE}" 2>&1 &
PID="$!"

echo "Started NZ100 training with nohup."
echo "PID: ${PID}"
echo "Log: ${LOG_FILE}"
echo "Watch log: tail -f ${LOG_FILE}"
