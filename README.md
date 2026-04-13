# NLP Project

## Structure

- `src/`: Python source files for data prep, scoring, and model training
- `scripts/`: shell helpers for training runs
- `data/raw/`: source corpora used by the project
- `data/processed/`: generated TSV files and learned weights
- `outputs/`: model checkpoints and evaluation summaries
- `Project_Proposal_Paper/`: proposal materials kept as-is for context
- `Sources/`: reference papers and background reading

## Common Runs

From the project root:

```bash
./scripts/run_baseline.sh
./scripts/run_hedgebert.sh
./scripts/run_ablation.sh
./scripts/run_all_training.sh
```

## Notes

- The training scripts now prefer `cuda`, then Apple `mps`, then `cpu`.
- The run scripts automatically activate `venv/` if it exists.
- Processed artifacts are expected under `data/processed/`.
