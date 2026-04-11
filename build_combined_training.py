"""
build_combined_training.py

Combines two training sources for HedgeBERT fine-tuning:

    Source 1 — Amazon Electronics (training_data.tsv)
               Direct sentiment sentences, no hedge annotations
               1500 sentences (750 pos / 750 neg)

    Source 2 — SFU hedged sentences (sfu_benchmark.tsv)
               Real human-annotated hedged sentences
               500 sentences (250 pos / 250 neg)
               Benchmark sentences excluded to prevent leakage

Final training set: ~2000 sentences (75% direct, 25% hedged)

Output TSV columns:
    sentence   : sentence text
    label      : 0 = negative, 1 = positive
    source     : 'amazon' or 'sfu_hedged'
    is_hedged  : False for Amazon, True for SFU hedged

Usage:
    python build_combined_training.py
        --amazon_path training_data.tsv
        --sfu_path sfu_benchmark.tsv
        --benchmark_path benchmark.tsv
        --output_path training_combined.tsv
        --sfu_sample 500
        --seed 42

Requirements:
    pip install pandas
"""

import argparse
import pandas as pd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build combined HedgeBERT training data from Amazon + SFU."
    )
    parser.add_argument(
        "--amazon_path",
        default="training_data.tsv",
        help="Amazon training TSV from build_training_data.py"
    )
    parser.add_argument(
        "--sfu_path",
        default="sfu_benchmark.tsv",
        help="Full SFU parsed TSV from parse_sfu.py"
    )
    parser.add_argument(
        "--benchmark_path",
        default="benchmark.tsv",
        help="256-sentence benchmark TSV to exclude from SFU pool"
    )
    parser.add_argument(
        "--output_path",
        default="training_combined.tsv",
        help="Output combined training TSV"
    )
    parser.add_argument(
        "--sfu_sample",
        type=int,
        default=500,
        help="Number of SFU hedged sentences to include (default: 500)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    args = parser.parse_args()

    if args.sfu_sample % 2 != 0:
        print("[ERROR] --sfu_sample must be even for balanced pos/neg sampling")
        return

    # ------------------------------------------------------------------
    # Load Amazon sentences
    # ------------------------------------------------------------------
    print(f"\nLoading Amazon training data: {args.amazon_path}")
    df_amazon = pd.read_csv(args.amazon_path, sep="\t")
    print(f"  Loaded {len(df_amazon)} sentences")
    print(f"  Positive: {(df_amazon['label'] == 1).sum()}")
    print(f"  Negative: {(df_amazon['label'] == 0).sum()}")

    # Standardise columns
    df_amazon_out = pd.DataFrame({
        "sentence" : df_amazon["sentence"],
        "label"    : df_amazon["label"],
        "source"   : "amazon",
        "is_hedged": False,
    })

    # ------------------------------------------------------------------
    # Load SFU hedged sentences, excluding benchmark
    # ------------------------------------------------------------------
    print(f"\nLoading SFU corpus: {args.sfu_path}")
    df_sfu = pd.read_csv(args.sfu_path, sep="\t")
    print(f"  Loaded {len(df_sfu)} total sentences")

    print(f"Loading benchmark exclusion list: {args.benchmark_path}")
    df_bench = pd.read_csv(args.benchmark_path, sep="\t")
    bench_sentences = set(df_bench["sentence"].tolist())
    print(f"  Excluding {len(bench_sentences)} benchmark sentences")

    # Filter to hedged only, non-benchmark
    df_sfu_hedged = df_sfu[
        (df_sfu["is_hedged"] == True) &
        (~df_sfu["sentence"].isin(bench_sentences))
    ].copy()

    print(f"  Available hedged non-benchmark sentences: {len(df_sfu_hedged)}")

    # Balanced sampling from SFU hedged pool
    per_class = args.sfu_sample // 2
    sfu_pos = df_sfu_hedged[df_sfu_hedged["label"] == 1]
    sfu_neg = df_sfu_hedged[df_sfu_hedged["label"] == 0]

    if len(sfu_pos) < per_class:
        print(f"[ERROR] Not enough positive SFU hedged sentences: "
              f"need {per_class}, have {len(sfu_pos)}")
        return
    if len(sfu_neg) < per_class:
        print(f"[ERROR] Not enough negative SFU hedged sentences: "
              f"need {per_class}, have {len(sfu_neg)}")
        return

    sfu_pos_sample = sfu_pos.sample(n=per_class, random_state=args.seed)
    sfu_neg_sample = sfu_neg.sample(n=per_class, random_state=args.seed)
    df_sfu_sample  = pd.concat([sfu_pos_sample, sfu_neg_sample],
                                ignore_index=True)

    print(f"  Sampled {len(df_sfu_sample)} SFU hedged sentences "
          f"({per_class} pos / {per_class} neg)")

    df_sfu_out = pd.DataFrame({
        "sentence" : df_sfu_sample["sentence"],
        "label"    : df_sfu_sample["label"],
        "source"   : "sfu_hedged",
        "is_hedged": True,
    })

    # ------------------------------------------------------------------
    # Combine and shuffle
    # ------------------------------------------------------------------
    df_combined = pd.concat([df_amazon_out, df_sfu_out], ignore_index=True)
    df_combined = df_combined.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    total    = len(df_combined)
    pos      = (df_combined["label"] == 1).sum()
    neg      = (df_combined["label"] == 0).sum()
    hedged   = (df_combined["is_hedged"] == True).sum()
    direct   = (df_combined["is_hedged"] == False).sum()
    amazon_n = (df_combined["source"] == "amazon").sum()
    sfu_n    = (df_combined["source"] == "sfu_hedged").sum()

    print(f"\n  {'='*46}")
    print(f"  Final Combined Training Set")
    print(f"  {'='*46}")
    print(f"  Total sentences   : {total}")
    print(f"  Amazon (direct)   : {amazon_n} "
          f"({amazon_n/total*100:.1f}%)")
    print(f"  SFU (hedged)      : {sfu_n} "
          f"({sfu_n/total*100:.1f}%)")
    print(f"  Positive (label=1): {pos}")
    print(f"  Negative (label=0): {neg}")
    print(f"  Hedged sentences  : {hedged}")
    print(f"  Direct sentences  : {direct}")
    print(f"  {'='*46}\n")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    df_combined.to_csv(args.output_path, sep="\t", index=False)
    print(f"Saved {total} sentences to: {args.output_path}")


if __name__ == "__main__":
    main()
