# HedgeBERT: Sentiment Classification Under Hedged Language

This project studies whether hedged language makes sentiment classification harder, and whether an explicit hedge-awareness signal can improve robustness. The core idea is to compare standard transformer sentiment classifiers against `HedgeBERT`, a BERT-based model that receives both the sentence text and a learned hedge-intensity scalar.

## What the project does

- Builds a large sentence-level sentiment dataset from Amazon Reviews 2023 with balanced sampling across all discovered categories
- Parses the SFU Review Corpus into sentence-level examples with hedged/direct labels
- Constructs a balanced evaluation benchmark for comparing performance on hedged vs. direct language
- Learns hedge cue weights with logistic regression and injects the resulting hedge score into `HedgeBERT`
- Trains and evaluates multiple baselines, including vanilla BERT, small BERT, ELECTRA, and BART

## Current dataset sizes

- `training_data.tsv`: `13,600` Amazon review sentences
- `training_combined.tsv`: `14,600` sentences total
  - `13,600` Amazon direct sentiment sentences
  - `1,000` SFU hedged sentiment sentences
- `benchmark.tsv`: `256` balanced evaluation sentences

## Repository structure

- `src/`: data preparation, hedge scoring, prompting, and training code
- `scripts/`: runnable shell helpers for dataset building and model training
- `data/raw/`: raw SFU review corpus files
- `data/processed/`: generated TSV files, learned weights, and processed artifacts
- `outputs/`: model checkpoints and evaluation summaries
- `Project_Proposal_Paper/`: proposal materials
- `Sources/`: reference papers

## Experimental pipeline

1. Parse the SFU corpus into sentence-level records
2. Build a balanced benchmark from SFU
3. Build balanced Amazon sentiment training data
4. Combine Amazon and SFU training examples without benchmark leakage
5. Learn hedge cue weights with logistic regression
6. Train and compare baseline and hedge-aware models

## Main scripts

### Data preparation

```bash
./venv/bin/python src/parse_sfu.py
./venv/bin/python src/build_benchmark.py
./scripts/build_training_data.sh
./venv/bin/python src/build_combined_training.py
./venv/bin/python src/learn_weights.py
```

### Core models

```bash
./scripts/run_baseline.sh
./scripts/run_hedgebert.sh
./scripts/run_ablation.sh
```

### Additional baselines

```bash
./scripts/run_electra.sh
./scripts/run_small_bert_baseline.sh
./scripts/run_small_bert.sh
./scripts/run_bart.sh
```

`run_small_bert.sh` trains the small HedgeBERT variant, while `run_small_bert_baseline.sh` trains the plain small-BERT baseline so both can be compared side by side.

## Model overview

- `train_baseline.py`: fine-tunes vanilla BERT for binary sentiment classification
- `train_hedgebert.py`: fine-tunes HedgeBERT by concatenating a hedge-intensity scalar to the `[CLS]` representation
- `train_hf_sequence_classifier.py`: generic trainer for additional Hugging Face sequence classifiers
- `learn_weights.py`: fits a logistic regression model over hedge cue categories to produce `weights.json`
- `hedge_scorer.py`: converts detected cue matches into a hedge score in `[0,1]`

## Evaluation setup

The main evaluation dataset is `benchmark.tsv`, a balanced benchmark derived from the SFU Review Corpus. It is balanced across:

- positive vs. negative
- hedged vs. direct
- 8 review domains

The key analysis is not only overall accuracy, but also the gap between:

- accuracy on hedged sentences
- accuracy on direct sentences

## Environment notes

- The training scripts prefer `cuda`, then Apple `mps`, then `cpu`
- Shell scripts automatically activate `venv/` if it exists
- Lighter models are recommended for local Mac training
