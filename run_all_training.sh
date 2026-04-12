#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -d "${ROOT_DIR}/venv" ]]; then
  source "${ROOT_DIR}/venv/bin/activate"
fi

echo "Running vanilla baseline training..."
"${ROOT_DIR}/run_baseline.sh"

echo
echo "Running HedgeBERT training..."
"${ROOT_DIR}/run_hedgebert.sh"

echo
echo "Training runs finished."
