"""
train_hedgebert.py
------------------
Fine-tunes HedgeBERT: BERT-base-uncased with a hedge intensity scalar
injected at the CLS token level before the classification head.

Architecture:
    [CLS token (768-dim)] + [hedge scalar (1-dim)] → 769-dim → Linear → 2 classes

The hedge scalar is computed by hedge_scorer.py and concatenated to the
CLS representation after BERT encoding. The classification head is
re-initialized (not loaded from checkpoint) to accept the 769-dim input.

Usage:
    python train_hedgebert.py \
        --train_path  training_combined.tsv \
        --bench_path  benchmark.tsv \
        --output_dir  ./hedgebert_output \
        --epochs      3 \
        --batch_size  16 \
        --seed        42

    # Ablation: replace hedge scalar with random noise to test scorer contribution
    python train_hedgebert.py ... --random_scalar

Output:
    hedgebert_output/
        best_model/          # saved model weights + tokenizer
        results.json         # all metrics
        results_summary.txt  # human-readable, presentation-ready

Requirements:
    pip install transformers scikit-learn torch pandas
    (hedge_scorer.py must be in the same directory)
"""

import argparse
import json
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertModel,
    BertTokenizerFast,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, classification_report

from hedge_scorer import score_hedge


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# HedgeBERT Model
# ---------------------------------------------------------------------------

class HedgeBERT(nn.Module):
    """
    BERT-base with a hedge scalar injected at the CLS level.

    Forward pass:
        1. Encode input with BERT → take CLS vector (768-dim)
        2. Concatenate hedge scalar → 769-dim vector
        3. Pass through dropout + linear head → logits (2-dim)

    The classification head is randomly initialized so it learns
    to use both the BERT representation and the hedge signal jointly.
    """

    def __init__(self, bert_model_name: str, dropout: float = 0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        hidden_size = self.bert.config.hidden_size  # 768 for bert-base

        # +1 for the hedge scalar → 769-dim input to classifier
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size + 1, 2)

        # Initialize classifier weights (Xavier uniform is standard)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask, token_type_ids, hedge_scalars):
        """
        Args:
            input_ids, attention_mask, token_type_ids : standard BERT inputs
            hedge_scalars : float tensor of shape (batch_size,)
                            values in [0.0, 1.0] from hedge_scorer

        Returns:
            logits : tensor of shape (batch_size, 2)
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # CLS token representation — shape: (batch_size, 768)
        cls_vector = outputs.last_hidden_state[:, 0, :]

        # Reshape scalar to (batch_size, 1) and concatenate → (batch_size, 769)
        hedge_scalars = hedge_scalars.unsqueeze(1).float()
        combined = torch.cat([cls_vector, hedge_scalars], dim=1)

        logits = self.classifier(self.dropout(combined))
        return logits


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HedgeDataset(Dataset):
    def __init__(self, sentences, labels, hedge_scores, tokenizer,
                 max_len=128, random_scalar=False):
        self.encodings = tokenizer(
            sentences,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels       = torch.tensor(labels, dtype=torch.long)
        if random_scalar:
            # Ablation: replace real scores with uniform random noise
            self.hedge_scores = torch.tensor(
                [random.uniform(0.0, 1.0) for _ in sentences],
                dtype=torch.float,
            )
        else:
            self.hedge_scores = torch.tensor(hedge_scores, dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "token_type_ids": self.encodings["token_type_ids"][idx],
            "hedge_scalar":   self.hedge_scores[idx],
            "labels":         self.labels[idx],
        }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, scheduler, device, criterion):
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        logits = model(
            input_ids      = batch["input_ids"].to(device),
            attention_mask = batch["attention_mask"].to(device),
            token_type_ids = batch["token_type_ids"].to(device),
            hedge_scalars  = batch["hedge_scalar"].to(device),
        )
        loss = criterion(logits, batch["labels"].to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, loader, device, criterion):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            logits = model(
                input_ids      = batch["input_ids"].to(device),
                attention_mask = batch["attention_mask"].to(device),
                token_type_ids = batch["token_type_ids"].to(device),
                hedge_scalars  = batch["hedge_scalar"].to(device),
            )
            loss = criterion(logits, batch["labels"].to(device))
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro")
    return total_loss / len(loader), acc, f1, all_preds, all_labels


def evaluate_subsets(model, tokenizer, bench_df, device, batch_size,
                     criterion, max_len=128, random_scalar=False):
    """Evaluate separately on hedged and direct subsets."""
    results = {}
    for subset_name, mask in [("hedged", bench_df["is_hedged"] == True),
                               ("direct", bench_df["is_hedged"] == False)]:
        sub_df = bench_df[mask].reset_index(drop=True)
        if len(sub_df) == 0:
            continue

        hedge_scores = [score_hedge(s) for s in sub_df["sentence"].tolist()]
        dataset = HedgeDataset(
            sub_df["sentence"].tolist(),
            sub_df["label"].tolist(),
            hedge_scores,
            tokenizer,
            max_len=max_len,
            random_scalar=random_scalar,
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        _, acc, f1, preds, labels = evaluate(model, loader, device, criterion)

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

    if "hedged" in results and "direct" in results:
        gap = results["direct"]["accuracy"] - results["hedged"]["accuracy"]
        results["accuracy_gap_direct_minus_hedged"] = round(gap, 4)
        print(f"\n  Accuracy gap (direct - hedged): {gap:+.4f}")
        if gap < 0:
            print("  → HedgeBERT performs better on hedged than direct ✓")
        elif gap == 0:
            print("  → No gap between hedged and direct")
        else:
            print("  → Direct still outperforms hedged")

    return results


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------

def write_summary(results, output_dir, baseline_results_path=None):
    lines = []
    lines.append("=" * 60)
    mode = "ABLATION (random scalar)" if results.get("random_scalar") else "HedgeBERT (hedge scalar)"
    lines.append(f"  {mode}")
    lines.append("=" * 60)

    lines.append("\n[Training]")
    for epoch_r in results.get("training_history", []):
        lines.append(
            f"  Epoch {epoch_r['epoch']}  "
            f"train_loss={epoch_r['train_loss']:.4f}  "
            f"val_acc={epoch_r['val_acc']:.4f}"
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

    # Compare against baseline if provided
    if baseline_results_path and os.path.exists(baseline_results_path):
        with open(baseline_results_path) as f:
            baseline = json.load(f)
        b_hedged = baseline.get("benchmark_subsets", {}).get("hedged", {})
        b_direct = baseline.get("benchmark_subsets", {}).get("direct", {})
        h_hedged = results.get("benchmark_subsets", {}).get("hedged", {})
        h_direct = results.get("benchmark_subsets", {}).get("direct", {})

        lines.append("\n[Comparison vs Vanilla BERT Baseline]")
        if b_hedged and h_hedged:
            delta_h = h_hedged["accuracy"] - b_hedged["accuracy"]
            lines.append(f"  Hedged acc delta : {delta_h:+.4f}  "
                         f"({'improvement' if delta_h > 0 else 'regression'})")
        if b_direct and h_direct:
            delta_d = h_direct["accuracy"] - b_direct["accuracy"]
            lines.append(f"  Direct acc delta : {delta_d:+.4f}  "
                         f"({'improvement' if delta_d > 0 else 'regression'})")

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
        description="Fine-tune HedgeBERT (BERT + CLS hedge scalar injection)."
    )
    parser.add_argument("--train_path",   default="training_combined.tsv")
    parser.add_argument("--bench_path",   default="benchmark.tsv")
    parser.add_argument("--output_dir",   default="./hedgebert_output")
    parser.add_argument("--baseline_results",
                        default="./baseline_output/results.json",
                        help="Path to baseline results.json for comparison")
    parser.add_argument("--model_name",   default="bert-base-uncased")
    parser.add_argument("--epochs",       type=int,   default=3)
    parser.add_argument("--batch_size",   type=int,   default=16)
    parser.add_argument("--max_len",      type=int,   default=128)
    parser.add_argument("--lr",           type=float, default=2e-5)
    parser.add_argument("--val_split",    type=float, default=0.1)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--random_scalar", action="store_true",
                        help="Ablation: replace hedge scalar with random noise")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {device}")
    mode = "ABLATION MODE (random scalar)" if args.random_scalar else "HedgeBERT (real hedge scalar)"
    print(f"Mode   : {mode}")

    # ── Load data ──────────────────────────────────────────────────────────

    print(f"\nLoading training data : {args.train_path}")
    train_df = pd.read_csv(args.train_path, sep="\t")
    print(f"  {len(train_df)} sentences")

    print(f"Loading benchmark     : {args.bench_path}")
    bench_df = pd.read_csv(args.bench_path, sep="\t")
    bench_df["is_hedged"] = bench_df["is_hedged"].map(
        lambda x: x if isinstance(x, bool) else str(x).strip().lower() == "true"
    )
    print(f"  {len(bench_df)} sentences  "
          f"(hedged={bench_df['is_hedged'].sum()}  "
          f"direct={(~bench_df['is_hedged']).sum()})")

    # ── Pre-compute hedge scores ───────────────────────────────────────────

    print("\nScoring hedge intensity for training sentences...")
    train_scores = [score_hedge(s) for s in train_df["sentence"].tolist()]
    hedged_count = sum(1 for s in train_scores if s > 0.2)
    print(f"  Sentences with hedge score > 0.2 : {hedged_count}/{len(train_scores)}")

    # ── Train / val split ──────────────────────────────────────────────────

    train_df = train_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    train_scores_shuffled = [train_scores[i] for i in train_df.index] \
        if hasattr(train_df, 'index') else train_scores

    # Re-score after shuffle to keep alignment
    sentences_list = train_df["sentence"].tolist()
    train_scores   = [score_hedge(s) for s in sentences_list]

    val_size     = int(len(train_df) * args.val_split)
    val_df       = train_df.iloc[:val_size].reset_index(drop=True)
    train_df     = train_df.iloc[val_size:].reset_index(drop=True)
    val_scores   = train_scores[:val_size]
    train_scores = train_scores[val_size:]
    print(f"\nTrain : {len(train_df)}  Val : {len(val_df)}")

    # ── Tokenizer & datasets ───────────────────────────────────────────────

    print(f"\nLoading tokenizer : {args.model_name}")
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)

    train_dataset = HedgeDataset(
        train_df["sentence"].tolist(), train_df["label"].tolist(),
        train_scores, tokenizer, args.max_len, args.random_scalar,
    )
    val_dataset = HedgeDataset(
        val_df["sentence"].tolist(), val_df["label"].tolist(),
        val_scores, tokenizer, args.max_len, args.random_scalar,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)

    # ── Model ──────────────────────────────────────────────────────────────

    print(f"Loading HedgeBERT ({args.model_name} + 1-dim hedge scalar)")
    model = HedgeBERT(args.model_name)
    model.to(device)
    print(f"  Classifier input dim : {model.bert.config.hidden_size + 1} (768 + 1)")

    # ── Optimizer & scheduler ──────────────────────────────────────────────

    criterion   = nn.CrossEntropyLoss()
    optimizer   = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # ── Training ───────────────────────────────────────────────────────────

    print(f"\n{'='*50}")
    print(f"  Training for {args.epochs} epochs")
    print(f"{'='*50}")

    training_history = []
    best_val_acc     = 0.0
    best_model_dir   = os.path.join(args.output_dir, "best_model")
    os.makedirs(best_model_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, criterion
        )
        _, val_acc, val_f1, _, _ = evaluate(model, val_loader, device, criterion)

        print(f"  Epoch {epoch}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  "
              f"val_acc={val_acc:.4f}  val_f1={val_f1:.4f}")

        training_history.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 4),
            "val_acc":    round(val_acc, 4),
            "val_f1":     round(val_f1, 4),
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save BERT weights + tokenizer
            model.bert.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
            # Save classifier head separately
            torch.save(model.classifier.state_dict(),
                       os.path.join(best_model_dir, "classifier_head.pt"))
            print(f"    ✓ New best model saved (val_acc={val_acc:.4f})")

    # ── Benchmark evaluation ───────────────────────────────────────────────

    print(f"\n{'='*50}")
    print("  Benchmark Evaluation (best model)")
    print(f"{'='*50}")

    # Reload best model
    best_model = HedgeBERT(best_model_dir)
    best_model.classifier.load_state_dict(
        torch.load(os.path.join(best_model_dir, "classifier_head.pt"),
                   map_location=device)
    )
    best_model.to(device)

    # Overall benchmark
    bench_scores  = [score_hedge(s) for s in bench_df["sentence"].tolist()]
    bench_dataset = HedgeDataset(
        bench_df["sentence"].tolist(), bench_df["label"].tolist(),
        bench_scores, tokenizer, args.max_len, args.random_scalar,
    )
    bench_loader = DataLoader(bench_dataset, batch_size=args.batch_size, shuffle=False)
    _, bench_acc, bench_f1, _, _ = evaluate(best_model, bench_loader, device, criterion)
    print(f"\n  Overall  acc={bench_acc:.4f}  macro_f1={bench_f1:.4f}")

    # Hedged vs direct subsets
    print("\n  Subset breakdown:")
    subset_results = evaluate_subsets(
        best_model, tokenizer, bench_df, device,
        args.batch_size, criterion, args.max_len, args.random_scalar,
    )

    # ── Save results ───────────────────────────────────────────────────────

    all_results = {
        "model":           args.model_name,
        "mode":            "ablation_random_scalar" if args.random_scalar else "hedgebert",
        "random_scalar":   args.random_scalar,
        "epochs":          args.epochs,
        "batch_size":      args.batch_size,
        "lr":              args.lr,
        "seed":            args.seed,
        "best_val_acc":    best_val_acc,
        "training_history": training_history,
        "benchmark_overall": {
            "accuracy": round(bench_acc, 4),
            "macro_f1": round(bench_f1, 4),
        },
        "benchmark_subsets": subset_results,
    }

    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to : {results_path}")

    write_summary(all_results, args.output_dir, args.baseline_results)


if __name__ == "__main__":
    main()