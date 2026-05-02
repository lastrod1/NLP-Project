#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -d "${ROOT_DIR}/venv" ]]; then
  source "${ROOT_DIR}/venv/bin/activate"
fi

python3 "${ROOT_DIR}/src/train_hf_sequence_classifier.py" \
  --model_name "google/electra-small-discriminator" \
  --model_label "ELECTRA Small" \
  --train_path "${ROOT_DIR}/data/processed/training_combined.tsv" \
  --bench_path "${ROOT_DIR}/data/processed/benchmark.tsv" \
  --output_dir "${ROOT_DIR}/outputs/electra_output" \
  --epochs 3 \
  --batch_size 16 \
  --seed 42 \
  "$@"
