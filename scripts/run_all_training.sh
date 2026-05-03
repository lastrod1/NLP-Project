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

echo "Running BART training..."
"${ROOT_DIR}/scripts/run_bart.sh"

echo "Running Electra training..."
"${ROOT_DIR}/scripts/run_electra.sh"

echo "Running small bert training..."
"${ROOT_DIR}/scripts/run_small_bert.sh"

echo
echo "Training runs finished."
