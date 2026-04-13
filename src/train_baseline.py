"""
train_baseline.py
-----------------
Fine-tunes BERT-base-uncased on the combined training set and evaluates
separately on hedged vs. direct subsets of the benchmark.

This is the VANILLA BERT BASELINE for HedgeBERT — no CLS injection,
no hedge scalar. Its job is to confirm that a standard BERT model
exhibits a measurable accuracy gap between hedged and direct sentiment.

Usage:
    python train_baseline.py \
        --train_path  training_combined.tsv \
        --bench_path  benchmark.tsv \
        --output_dir  ./baseline_output \
        --epochs      3 \
        --batch_size  16 \
        --seed        42

Output:
    baseline_output/
        best_model/          # saved HuggingFace model + tokenizer
        results.json         # all metrics in one place
        results_summary.txt  # human-readable summary for the presentation

Requirements:
    pip install transformers datasets scikit-learn torch pandas
"""

import argparse
import json
import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, classification_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device():
    """Pick the best available accelerator for the current machine."""
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    return torch.device("cpu"), "cpu"


def validate_split(val_split, n_rows):
    if not 0 < val_split < 1:
        raise ValueError("--val_split must be between 0 and 1.")
    val_size = int(n_rows * val_split)
    if val_size <= 0 or val_size >= n_rows:
        raise ValueError(
            "--val_split produces an empty train or validation set. "
            "Use a value that leaves at least one row in each split."
        )
    return val_size


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SentimentDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(
            sentences,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }
        if "token_type_ids" in self.encodings:
            item["token_type_ids"] = self.encodings["token_type_ids"][idx]
        return item


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        labels         = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, loader, device):
    """Return loss, accuracy, F1, and raw predictions."""
    model.eval()
    all_preds  = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels         = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )
            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro")
    avg_loss = total_loss / len(loader)
    return avg_loss, acc, f1, all_preds, all_labels


def evaluate_subsets(model, tokenizer, bench_df, device, batch_size, max_len=128):
    """
    Evaluate separately on hedged and direct subsets.
    This is the core diagnostic metric for HedgeBERT.
    """
    results = {}
    for subset_name, mask in [("hedged", bench_df["is_hedged"] == True),
                               ("direct", bench_df["is_hedged"] == False)]:
        sub_df = bench_df[mask].reset_index(drop=True)
        if len(sub_df) == 0:
            print(f"  [WARN] No {subset_name} examples in benchmark — skipping.")
            continue

        dataset = SentimentDataset(
            sub_df["sentence"].tolist(),
            sub_df["label"].tolist(),
            tokenizer,
            max_len=max_len,
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        _, acc, f1, preds, labels = evaluate(model, loader, device)

        # Per-class breakdown
        report = classification_report(
            labels, preds,
            target_names=["negative", "positive"],
            output_dict=True,
        )

        results[subset_name] = {
            "n":        len(sub_df),
            "accuracy": round(acc, 4),
            "macro_f1": round(f1, 4),
            "report":   report,
        }
        print(f"  {subset_name:<8}  n={len(sub_df):>3}  "
              f"acc={acc:.4f}  macro_f1={f1:.4f}")

    # Accuracy gap — the key diagnostic
    if "hedged" in results and "direct" in results:
        gap = results["direct"]["accuracy"] - results["hedged"]["accuracy"]
        results["accuracy_gap_direct_minus_hedged"] = round(gap, 4)
        print(f"\n  Accuracy gap (direct - hedged): {gap:+.4f}")
        if gap > 0:
            print("  → BERT performs better on direct than hedged sentiment ✓")
        else:
            print("  → No gap detected (unexpected for vanilla BERT)")

    return results


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------

def write_summary(results, output_dir):
    lines = []
    lines.append("=" * 60)
    lines.append("  HedgeBERT — Vanilla BERT Baseline Results")
    lines.append("=" * 60)

    lines.append("\n[Training]")
    for epoch_r in results.get("training_history", []):
        lines.append(
            f"  Epoch {epoch_r['epoch']}  "
            f"train_loss={epoch_r['train_loss']:.4f}  "
            f"val_acc={epoch_r.get('val_acc', 'N/A')}"
        )

    lines.append("\n[Benchmark — Overall]")
    overall = results.get("benchmark_overall", {})
    lines.append(f"  Accuracy : {overall.get('accuracy', 'N/A')}")
    lines.append(f"  Macro F1 : {overall.get('macro_f1', 'N/A')}")

    lines.append("\n[Benchmark — Hedged vs Direct Subsets]")
    for subset in ["hedged", "direct"]:
        s = results.get("benchmark_subsets", {}).get(subset, {})
        if s:
            lines.append(
                f"  {subset:<8}  n={s['n']:>3}  "
                f"acc={s['accuracy']:.4f}  macro_f1={s['macro_f1']:.4f}"
            )

    gap = results.get("benchmark_subsets", {}).get(
        "accuracy_gap_direct_minus_hedged", None
    )
    if gap is not None:
        lines.append(f"\n  Accuracy gap (direct - hedged) : {gap:+.4f}")
        lines.append(
            f"  Interpretation : BERT scores "
            f"{abs(gap)*100:.1f} percentage points "
            f"{'higher on direct' if gap > 0 else 'higher on hedged'} sentiment."
        )

    lines.append("\n" + "=" * 60)

    summary_path = os.path.join(output_dir, "results_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSummary saved to: {summary_path}")
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune vanilla BERT baseline for HedgeBERT."
    )
    parser.add_argument("--train_path",  default=str(PROCESSED_DIR / "training_combined.tsv"),
                        help="Combined training TSV (from build_combined_training.py)")
    parser.add_argument("--bench_path",  default=str(PROCESSED_DIR / "benchmark.tsv"),
                        help="256-sentence benchmark TSV (from build_benchmark.py)")
    parser.add_argument("--output_dir",  default=str(OUTPUTS_DIR / "baseline_output"),
                        help="Directory to save model and results")
    parser.add_argument("--model_name",  default="bert-base-uncased",
                        help="HuggingFace model name (default: bert-base-uncased)")
    parser.add_argument("--epochs",      type=int,   default=3)
    parser.add_argument("--batch_size",  type=int,   default=16)
    parser.add_argument("--max_len",     type=int,   default=128)
    parser.add_argument("--lr",          type=float, default=2e-5)
    parser.add_argument("--val_split",   type=float, default=0.1,
                        help="Fraction of training data held out for validation")
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device, device_name = resolve_device()
    print(f"\nDevice: {device}")
    if device_name == "cpu" and hasattr(torch.backends, "mps") and torch.backends.mps.is_built():
        print("MPS support is built into PyTorch, but it is not available in this runtime.")

    # ── Load data ──────────────────────────────────────────────────────────

    print(f"\nLoading training data: {args.train_path}")
    train_df = pd.read_csv(args.train_path, sep="\t")
    print(f"  {len(train_df)} training sentences")
    print(f"  Positive: {(train_df['label']==1).sum()}  "
          f"Negative: {(train_df['label']==0).sum()}")

    print(f"\nLoading benchmark: {args.bench_path}")
    bench_df = pd.read_csv(args.bench_path, sep="\t")

    # Normalize is_hedged column (TSV may store as string "True"/"False")
    bench_df["is_hedged"] = bench_df["is_hedged"].map(
        lambda x: x if isinstance(x, bool) else str(x).strip().lower() == "true"
    )
    print(f"  {len(bench_df)} benchmark sentences")
    print(f"  Hedged: {bench_df['is_hedged'].sum()}  "
          f"Direct: {(~bench_df['is_hedged']).sum()}")

    # ── Train / val split ──────────────────────────────────────────────────

    train_df = train_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    val_size  = validate_split(args.val_split, len(train_df))
    val_df    = train_df.iloc[:val_size].reset_index(drop=True)
    train_df  = train_df.iloc[val_size:].reset_index(drop=True)
    print(f"\nTrain: {len(train_df)}  Val: {len(val_df)}")

    # ── Tokenizer & datasets ───────────────────────────────────────────────

    print(f"\nLoading tokenizer: {args.model_name}")
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)

    train_dataset = SentimentDataset(
        train_df["sentence"].tolist(), train_df["label"].tolist(),
        tokenizer, args.max_len,
    )
    val_dataset = SentimentDataset(
        val_df["sentence"].tolist(), val_df["label"].tolist(),
        tokenizer, args.max_len,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)

    # ── Model ──────────────────────────────────────────────────────────────

    print(f"Loading model: {args.model_name}")
    model = BertForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    )
    model.to(device)

    # ── Optimizer & scheduler ──────────────────────────────────────────────

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── Training ───────────────────────────────────────────────────────────

    print(f"\n{'='*50}")
    print(f"  Training for {args.epochs} epochs")
    print(f"{'='*50}")

    training_history = []
    best_val_acc = -1.0
    best_model_dir = os.path.join(args.output_dir, "best_model")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        _, val_acc, val_f1, _, _ = evaluate(model, val_loader, device)

        print(f"  Epoch {epoch}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  "
              f"val_acc={val_acc:.4f}  val_f1={val_f1:.4f}")

        training_history.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 4),
            "val_acc":    round(val_acc, 4),
            "val_f1":     round(val_f1, 4),
        })

        # Save best model by validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
            print(f"    ✓ New best model saved (val_acc={val_acc:.4f})")

    # ── Benchmark evaluation ───────────────────────────────────────────────

    print(f"\n{'='*50}")
    print("  Benchmark Evaluation (best model)")
    print(f"{'='*50}")

    # Load best model for final eval
    best_model = BertForSequenceClassification.from_pretrained(best_model_dir)
    best_model.to(device)

    # Overall benchmark
    bench_dataset = SentimentDataset(
        bench_df["sentence"].tolist(), bench_df["label"].tolist(),
        tokenizer, args.max_len,
    )
    bench_loader = DataLoader(bench_dataset, batch_size=args.batch_size, shuffle=False)
    _, bench_acc, bench_f1, _, _ = evaluate(best_model, bench_loader, device)
    print(f"\n  Overall benchmark  acc={bench_acc:.4f}  macro_f1={bench_f1:.4f}")

    # Hedged vs direct subsets — the key diagnostic
    print("\n  Subset breakdown:")
    subset_results = evaluate_subsets(
        best_model, tokenizer, bench_df, device, args.batch_size, args.max_len
    )

    # ── Save results ───────────────────────────────────────────────────────

    all_results = {
        "model":       args.model_name,
        "epochs":      args.epochs,
        "batch_size":  args.batch_size,
        "lr":          args.lr,
        "seed":        args.seed,
        "best_val_acc": best_val_acc,
        "training_history": training_history,
        "benchmark_overall": {
            "accuracy":  round(bench_acc, 4),
            "macro_f1":  round(bench_f1, 4),
        },
        "benchmark_subsets": subset_results,
    }

    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to: {results_path}")

    write_summary(all_results, args.output_dir)


if __name__ == "__main__":
    main()
