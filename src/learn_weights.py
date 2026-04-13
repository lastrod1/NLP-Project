import argparse
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report

CUE_INVENTORY = {
    "epistemic": [
        "i think", "i believe", "i feel", "i guess", "i suppose",
        "i assume", "i suspect", "it seems", "it appears", "it looks like",
        "in my opinion", "in my view", "to me", "i would say",
        "i'm not sure", "i am not sure", "not sure", "hard to say",
        "i wonder", "i doubt", "seems like", "seems to",
    ],
    "modal": [
        "might", "could", "would", "may", "should",
        "can", "ought to", "need to",
    ],
    "approximator": [
        "kind of", "sort of", "somewhat", "rather", "fairly",
        "pretty much", "quite", "a bit", "a little", "slightly",
        "more or less", "roughly", "approximately", "around",
        "almost", "nearly", "in a way", "to some extent",
    ],
    "diminisher": [
        "decent", "acceptable", "okay", "fine", "not bad",
        "alright", "all right", "tolerable", "adequate", "average",
        "mediocre", "passable", "satisfactory", "sufficient",
    ],
    "conditional": [
        "if", "unless", "provided that", "as long as",
        "assuming that", "in case", "supposing",
    ],
}

CONJUNCTIONS = {
    "but", "although", "though", "however", "yet",
    "while", "whereas", "nevertheless", "despite",
}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def compile_cue_patterns():
    """Compile whole-word regexes so short cues like 'can' stay well-behaved."""
    patterns = {}
    for category, cues in CUE_INVENTORY.items():
        patterns[category] = []
        for cue in cues:
            pattern = re.compile(rf"\b{re.escape(cue)}\b", re.IGNORECASE)
            patterns[category].append((cue, pattern))
    return patterns


CUE_PATTERNS = compile_cue_patterns()

def split_at_conjunctions(sentence):
    """
    Split sentence into clauses at conjunction boundaries.
    Returns list of clause strings.
    """
    pattern = r'\b(' + '|'.join(re.escape(c) for c in CONJUNCTIONS) + r')\b'
    clauses = re.split(pattern, sentence.lower(), flags=re.IGNORECASE)
    # Filter out the conjunction tokens themselves and empty strings
    clauses = [c.strip() for c in clauses
               if c.strip() and c.strip().lower() not in CONJUNCTIONS]
    return clauses if clauses else [sentence.lower()]


def get_clause_with_cue(sentence, cue):
    """Return the clause that contains the cue, or full sentence if not split."""
    clauses = split_at_conjunctions(sentence)
    for clause in clauses:
        if cue in clause:
            return clause
    return sentence.lower()

def count_cues_per_category(sentence):
    """
    Count how many cues from each category appear in the sentence.
    Returns a dict: {category: count}
    Applies scope simplification — checks cue presence per clause.
    """
    sent_lower = sentence.lower()
    counts = {cat: 0 for cat in CUE_INVENTORY}

    for category, cue_patterns in CUE_PATTERNS.items():
        for cue, pattern in cue_patterns:
            if pattern.search(sent_lower):
                # Check which clause the cue lives in
                clause = get_clause_with_cue(sent_lower, cue)
                if pattern.search(clause):
                    counts[category] += 1

    return counts


def extract_features(sentences):
    """Extract cue count features for a list of sentences.
    Returns numpy array of shape (n_sentences, n_categories).
    """
    categories = list(CUE_INVENTORY.keys())
    rows = []
    for sent in sentences:
        counts = count_cues_per_category(sent)
        rows.append([counts[cat] for cat in categories])
    return np.array(rows), categories

def main():
    parser = argparse.ArgumentParser(
        description="Learn hedge cue weights from SFU corpus via logistic regression."
    )
    parser.add_argument(
        "--full_sfu",
        default=str(PROCESSED_DIR / "sfu_benchmark.tsv"),
        help="Full parsed SFU TSV from parse_sfu.py (default: sfu_benchmark.tsv)"
    )
    parser.add_argument(
        "--benchmark",
        default=str(PROCESSED_DIR / "benchmark.tsv"),
        help="256-sentence benchmark TSV to exclude from training (default: benchmark.tsv)"
    )
    parser.add_argument(
        "--output_weights",
        default=str(PROCESSED_DIR / "weights.json"),
        help="Output path for learned weights JSON (default: weights.json)"
    )
    args = parser.parse_args()

    # Load full SFU
    print(f"\nLoading full SFU parse: {args.full_sfu}")
    df_full = pd.read_csv(args.full_sfu, sep="\t")
    print(f"  Loaded {len(df_full)} sentences")

    # Load benchmark and exclude those sentences
    print(f"Loading benchmark to exclude: {args.benchmark}")
    df_bench = pd.read_csv(args.benchmark, sep="\t")
    bench_sentences = set(df_bench["sentence"].tolist())
    df_train = df_full[~df_full["sentence"].isin(bench_sentences)].copy()
    print(f"  Excluded {len(df_full) - len(df_train)} benchmark sentences")
    print(f"  Training pool: {len(df_train)} sentences")

    # Extract features
    print(f"\nExtracting cue count features...")
    X, categories = extract_features(df_train["sentence"].tolist())
    y = df_train["is_hedged"].astype(int).values

    print(f"  Feature matrix shape : {X.shape}")
    print(f"  Positive (hedged)    : {y.sum()}")
    print(f"  Negative (direct)    : {(y == 0).sum()}")

    # Train logistic regression
    print(f"\nTraining logistic regression...")
    clf = LogisticRegression(
        class_weight="balanced",  # handle class imbalance
        max_iter=1000,
        random_state=42,
    )
    clf.fit(X, y)

    # Evaluate
    y_pred = clf.predict(X)
    print(f"\nTraining set performance:")
    print(classification_report(y, y_pred,
                                 target_names=["direct", "hedged"]))

    # Extract and normalize coefficients
    raw_coeffs = clf.coef_[0]
    print(f"\nRaw logistic regression coefficients:")
    for cat, coef in zip(categories, raw_coeffs):
        print(f"  {cat:<15} : {coef:.4f}")

    # Clip negatives to 0 (negative coefficients mean the cue predicts direct,
    # which shouldn't happen for hedge cues — likely noise)
    clipped = np.clip(raw_coeffs, 0, None)

    # Normalize to [0.1, 0.9] range
    # Floor at 0.1 so every detected cue contributes something
    # Cap at 0.9 so no single cue dominates completely
    if clipped.max() > 0:
        scaler = MinMaxScaler(feature_range=(0.1, 0.9))
        normalized = scaler.fit_transform(clipped.reshape(-1, 1)).flatten()
    else:
        normalized = np.full(len(clipped), 0.1)

    # Build weights dict
    weights = {cat: round(float(w), 4) for cat, w in zip(categories, normalized)}

    print(f"\nNormalized weights [0.1, 0.9]:")
    for cat, w in weights.items():
        print(f"  {cat:<15} : {w:.4f}")

    # Save
    output_path = Path(args.output_weights)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(weights, f, indent=2)

    print(f"\nSaved weights to: {output_path}")


if __name__ == "__main__":
    main()
