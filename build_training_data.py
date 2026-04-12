import argparse
import csv
import random
import nltk
from datasets import load_dataset

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


CATEGORY       = "Electronics"
DATASET_NAME   = "McAuley-Lab/Amazon-Reviews-2023"
DATASET_CONFIG = f"raw_review_{CATEGORY}"

MIN_WORDS = 6
MAX_WORDS = 60

def get_label(rating):
    """Map star rating to binary label. Returns None for 3-star."""
    if rating <= 2.0:
        return 0
    elif rating >= 4.0:
        return 1
    else:
        return None


def quality_ok(sentence):
    """Return True if sentence passes word count filter."""
    words = sentence.split()
    return MIN_WORDS <= len(words) <= MAX_WORDS


def tokenize_review(text):
    """Split review text into sentences."""
    try:
        return nltk.sent_tokenize(text)
    except Exception:
        return [text]

def collect_sentences(target_per_class, seed):
    """
    Stream Amazon reviews, extract sentences, filter, and collect
    until target_per_class is reached for both positive and negative.
    Returns two lists: positives, negatives.
    """
    positives = []
    negatives = []

    needed_pos = target_per_class
    needed_neg = target_per_class

    print(f"\nStreaming dataset: {DATASET_CONFIG}")
    print(f"Target: {target_per_class} sentences per class\n")

    dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split="full",
        streaming=True,
        trust_remote_code=True,
    )

    reviewed = 0
    for review in dataset:
        # Stop early if both classes are full
        if len(positives) >= needed_pos and len(negatives) >= needed_neg:
            break

        rating = review.get("rating")
        text   = review.get("text", "")

        if not text or not rating:
            continue

        label = get_label(float(rating))
        if label is None:
            continue

        # Skip if this class is already full
        if label == 1 and len(positives) >= needed_pos:
            continue
        if label == 0 and len(negatives) >= needed_neg:
            continue

        sentences = tokenize_review(text)
        for sent in sentences:
            sent = sent.strip()
            if not quality_ok(sent):
                continue

            record = {
                "sentence" : sent,
                "label"    : label,
                "category" : CATEGORY,
                "augmented": False,
            }

            if label == 1 and len(positives) < needed_pos:
                positives.append(record)
            elif label == 0 and len(negatives) < needed_neg:
                negatives.append(record)

        reviewed += 1
        if reviewed % 5000 == 0:
            print(f"  Reviews scanned: {reviewed:>7} | "
                  f"pos: {len(positives):>5}/{needed_pos} | "
                  f"neg: {len(negatives):>5}/{needed_neg}")

    return positives, negatives

def print_stats(records):
    pos = sum(1 for r in records if r["label"] == 1)
    neg = sum(1 for r in records if r["label"] == 0)
    print(f"\n  {'='*40}")
    print(f"  Total sentences : {len(records)}")
    print(f"  Positive        : {pos}")
    print(f"  Negative        : {neg}")
    print(f"  {'='*40}\n")

def main():
    parser = argparse.ArgumentParser(
        description="Build Amazon review training data for HedgeBERT."
    )
    parser.add_argument(
        "--output_path",
        default="training_data.tsv",
        help="Output TSV file path (default: training_data.tsv)"
    )
    parser.add_argument(
        "--target",
        type=int,
        default=1500,
        help="Total target sentences (split equally pos/neg, default: 1500)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    args = parser.parse_args()

    if args.target % 2 != 0:
        print("[ERROR] --target must be an even number")
        return

    random.seed(args.seed)
    target_per_class = args.target // 2

    positives, negatives = collect_sentences(target_per_class, args.seed)

    # Shuffle and combine
    random.shuffle(positives)
    random.shuffle(negatives)
    all_records = positives + negatives
    random.shuffle(all_records)

    print_stats(all_records)

    # Write TSV
    fieldnames = ["sentence", "label", "category", "augmented"]
    with open(args.output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(all_records)

    print(f"Saved {len(all_records)} sentences to: {args.output_path}")


if __name__ == "__main__":
    main()
