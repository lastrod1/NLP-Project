import argparse
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

def main():
    def normalize_bool(series):
        return series.map(
            lambda value: value if isinstance(value, bool)
            else str(value).strip().lower() == "true"
        )

    parser = argparse.ArgumentParser(
        description="Build combined HedgeBERT training data from Amazon + SFU."
    )
    parser.add_argument(
        "--amazon_path",
        default=str(PROCESSED_DIR / "training_data.tsv"),
        help="Amazon training TSV from build_training_data.py"
    )
    parser.add_argument(
        "--sfu_path",
        default=str(PROCESSED_DIR / "sfu_benchmark.tsv"),
        help="Full SFU parsed TSV from parse_sfu.py"
    )
    parser.add_argument(
        "--benchmark_path",
        default=str(PROCESSED_DIR / "benchmark.tsv"),
        help="256-sentence benchmark TSV to exclude from SFU pool"
    )
    parser.add_argument(
        "--output_path",
        default=str(PROCESSED_DIR / "training_combined.tsv"),
        help="Output combined training TSV"
    )
    parser.add_argument(
        "--sfu_sample",
        type=int,
        default=1000,
        help="Number of SFU hedged sentences to include (default: 1000)"
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

    # Amazon sentences
    df_amazon = pd.read_csv(args.amazon_path, sep="\t")
    print(f"  Loaded {len(df_amazon)} sentences")
    print(f"  Positive: {(df_amazon['label'] == 1).sum()}")
    print(f"  Negative: {(df_amazon['label'] == 0).sum()}")

    df_amazon_out = pd.DataFrame({
        "sentence" : df_amazon["sentence"],
        "label"    : df_amazon["label"],
        "source"   : "amazon",
        "is_hedged": False,
    })

    # SFU hedged sentences
    df_sfu = pd.read_csv(args.sfu_path, sep="\t")
    if "is_hedged" in df_sfu.columns:
        df_sfu["is_hedged"] = normalize_bool(df_sfu["is_hedged"])
    print(f"  Loaded {len(df_sfu)} total sentences")

    df_bench = pd.read_csv(args.benchmark_path, sep="\t")
    bench_sentences = set(df_bench["sentence"].tolist())
    print(f"  Excluding {len(bench_sentences)} benchmark sentences")

    # Filter to hedged only, non-benchmark
    df_sfu_hedged = df_sfu[
        (df_sfu["is_hedged"] == True) &
        (~df_sfu["sentence"].isin(bench_sentences))
    ].copy()

    print(f"  Available hedged non-benchmark sentences: {len(df_sfu_hedged)}")

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

    df_combined = pd.concat([df_amazon_out, df_sfu_out], ignore_index=True)
    df_combined = df_combined.sample(frac=1, random_state=args.seed).reset_index(drop=True)

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

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_combined.to_csv(output_path, sep="\t", index=False)
    print(f"Saved {total} sentences to: {output_path}")


if __name__ == "__main__":
    main()
