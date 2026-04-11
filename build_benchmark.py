"""
build_benchmark.py

Samples a balanced benchmark from the full SFU parsed TSV produced by
parse_sfu.py. The benchmark is strictly balanced across:

    - Subset:    hedged vs. direct (equal halves)
    - Sentiment: positive vs. negative (equal within each subset)
    - Domain:    exactly equal samples from all 8 domains

Final size: 8 sentences x 8 domains x 4 cells = 256 sentences total

    Cell breakdown:
        hedged   + positive : 64  (8 per domain)
        hedged   + negative : 64  (8 per domain)
        direct   + positive : 64  (8 per domain)
        direct   + negative : 64  (8 per domain)

Usage:
    python build_benchmark.py --input_path sfu_benchmark.tsv
                              --output_path benchmark.tsv
                              --seed 42
"""

import argparse
import pandas as pd


# ---------------------------------------------------------------------------
# Quality filter
# ---------------------------------------------------------------------------

MIN_WORDS = 6
MAX_WORDS = 60

def quality_filter(df):
    """Remove sentences that are too short, too long, or low quality."""
    word_counts = df["sentence"].str.split().str.len()
    mask = (word_counts >= MIN_WORDS) & (word_counts <= MAX_WORDS)
    before = len(df)
    df = df[mask].copy()
    after = len(df)
    print(f"  Quality filter: removed {before - after} sentences "
          f"(outside {MIN_WORDS}-{MAX_WORDS} word range)")
    return df


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

SAMPLES_PER_CELL = 8  # per domain per cell
DOMAINS = ["BOOKS", "CARS", "COMPUTERS", "COOKWARE",
           "HOTELS", "MOVIES", "MUSIC", "PHONES"]

CELLS = [
    {"is_hedged": True,  "label": 1, "name": "hedged+positive"},
    {"is_hedged": True,  "label": 0, "name": "hedged+negative"},
    {"is_hedged": False, "label": 1, "name": "direct+positive"},
    {"is_hedged": False, "label": 0, "name": "direct+negative"},
]


def sample_benchmark(df, seed):
    """Strictly sample SAMPLES_PER_CELL sentences per domain per cell."""
    sampled_parts = []

    print(f"\n  Sampling {SAMPLES_PER_CELL} per domain per cell "
          f"(seed={seed}):\n")

    for cell in CELLS:
        cell_df = df[(df["is_hedged"] == cell["is_hedged"]) &
                     (df["label"]     == cell["label"])].copy()

        cell_parts = []
        for domain in DOMAINS:
            domain_df = cell_df[cell_df["domain"] == domain]

            if len(domain_df) < SAMPLES_PER_CELL:
                raise ValueError(
                    f"Not enough sentences for cell='{cell['name']}' "
                    f"domain='{domain}': need {SAMPLES_PER_CELL}, "
                    f"have {len(domain_df)}"
                )

            sampled = domain_df.sample(n=SAMPLES_PER_CELL,
                                       random_state=seed)
            cell_parts.append(sampled)

        cell_sample = pd.concat(cell_parts, ignore_index=True)
        sampled_parts.append(cell_sample)

        print(f"    {cell['name']:<22} : {len(cell_sample)} sentences "
              f"({SAMPLES_PER_CELL} per domain)")

    benchmark = pd.concat(sampled_parts, ignore_index=True)
    return benchmark


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def print_stats(df, label="Benchmark"):
    """Print a summary of the benchmark."""
    total   = len(df)
    hedged  = df[df["is_hedged"] == True]
    direct  = df[df["is_hedged"] == False]

    print(f"\n  {'='*46}")
    print(f"  {label}")
    print(f"  {'='*46}")
    print(f"  Total sentences   : {total}")
    print(f"  Hedged            : {len(hedged)}")
    print(f"  Direct            : {len(direct)}")
    print(f"  Positive (label=1): {len(df[df['label']==1])}")
    print(f"  Negative (label=0): {len(df[df['label']==0])}")

    print(f"\n  Domain distribution:")
    for domain in DOMAINS:
        d = df[df["domain"] == domain]
        dh = d[d["is_hedged"] == True]
        dd = d[d["is_hedged"] == False]
        print(f"    {domain:<12} total={len(d):>3}  "
              f"hedged={len(dh):>2}  direct={len(dd):>2}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build balanced HedgeBERT benchmark from parsed SFU TSV."
    )
    parser.add_argument(
        "--input_path",
        default="sfu_benchmark.tsv",
        help="Path to full parsed TSV from parse_sfu.py"
    )
    parser.add_argument(
        "--output_path",
        default="benchmark.tsv",
        help="Output path for the sampled benchmark TSV"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    args = parser.parse_args()

    # Load
    print(f"\nLoading: {args.input_path}")
    df = pd.read_csv(args.input_path, sep="\t")
    print(f"  Loaded {len(df)} sentences")

    # Quality filter
    df = quality_filter(df)

    # Sample
    print(f"\nSampling benchmark...")
    benchmark = sample_benchmark(df, seed=args.seed)

    # Stats
    print_stats(benchmark, label="Final Benchmark")

    # Save
    benchmark.to_csv(args.output_path, sep="\t", index=False)
    print(f"  Saved {len(benchmark)} sentences to: {args.output_path}")


if __name__ == "__main__":
    main()
