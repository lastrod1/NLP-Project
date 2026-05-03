#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -d "${ROOT_DIR}/venv" ]]; then
  # Use the project virtual environment when it is available.
  source "${ROOT_DIR}/venv/bin/activate"
fi

python3 "${ROOT_DIR}/src/train_baseline.py" \
  --train_path "${ROOT_DIR}/data/processed/training_combined.tsv" \
  --bench_path "${ROOT_DIR}/data/processed/benchmark.tsv" \
  --output_dir "${ROOT_DIR}/outputs/baseline_output" \
  --epochs 5 \
  --batch_size 16 \
  --seed 67 \
  "$@"
