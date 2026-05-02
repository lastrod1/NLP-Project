#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -d "${ROOT_DIR}/venv" ]]; then
  source "${ROOT_DIR}/venv/bin/activate"
fi

python3 "${ROOT_DIR}/src/train_openai_sentiment.py" \
  --train_path "${ROOT_DIR}/data/processed/training_combined.tsv" \
  --bench_path "${ROOT_DIR}/data/processed/benchmark.tsv" \
  --output_dir "${ROOT_DIR}/outputs/openai_gpt_output" \
  --base_model "gpt-4.1-mini-2025-04-14" \
  "$@"
