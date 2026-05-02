#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -d "${ROOT_DIR}/venv" ]]; then
  source "${ROOT_DIR}/venv/bin/activate"
fi

echo "Running vanilla BERT baseline..."
"${ROOT_DIR}/scripts/run_baseline.sh"

echo
echo "Running ELECTRA..."
"${ROOT_DIR}/scripts/run_electra.sh"

echo
echo "Running small BERT..."
"${ROOT_DIR}/scripts/run_small_bert.sh"

echo
echo "Running Llama 2..."
"${ROOT_DIR}/scripts/run_llama2.sh"

echo
echo "Model comparison runs finished."
