#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -d "${ROOT_DIR}/venv" ]]; then
  source "${ROOT_DIR}/venv/bin/activate"
fi

python3 "${ROOT_DIR}/train_hedgebert.py" \
  --train_path "${ROOT_DIR}/training_combined.tsv" \
  --bench_path "${ROOT_DIR}/benchmark.tsv" \
  --output_dir "${ROOT_DIR}/ablation_output" \
  --baseline_results "${ROOT_DIR}/baseline_output/results.json" \
  --epochs 3 \
  --batch_size 16 \
  --seed 42 \
  --random_scalar \
  "$@"
