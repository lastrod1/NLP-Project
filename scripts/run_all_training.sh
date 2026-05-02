#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -d "${ROOT_DIR}/venv" ]]; then
  source "${ROOT_DIR}/venv/bin/activate"
fi

echo "Running vanilla baseline training..."
"${ROOT_DIR}/scripts/run_baseline.sh"

echo
echo "Running HedgeBERT training..."
"${ROOT_DIR}/scripts/run_hedgebert.sh"

echo
echo "Running Ablation training..."
"${ROOT_DIR}/scripts/run_ablation.sh"

echo
echo "Training runs finished."
