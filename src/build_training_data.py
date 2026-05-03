import argparse
import csv
import random
from pathlib import Path

import nltk
from datasets import get_dataset_config_names, load_dataset

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


DATASET_NAME = "McAuley-Lab/Amazon-Reviews-2023"
MIN_WORDS = 6
MAX_WORDS = 60
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def get_label(rating):
    """Map star rating to binary label. Returns None for 3-star."""
    if rating <= 2.0:
        return 0
    if rating >= 4.0:
        return 1
    return None


def quality_ok(sentence):
    words = sentence.split()
    return MIN_WORDS <= len(words) <= MAX_WORDS


def tokenize_review(text):
    try:
        return nltk.sent_tokenize(text)
    except Exception:
        return [text]


def discover_categories():
    """Discover all Amazon raw review categories exposed by the dataset."""
    config_names = get_dataset_config_names(
        DATASET_NAME,
        trust_remote_code=True,
    )
    categories = sorted(
        config.removeprefix("raw_review_")
        for config in config_names
        if config.startswith("raw_review_")
    )
    if not categories:
        raise ValueError("No raw review categories were found for the dataset.")
    return categories


def collect_category_sentences(category, target_per_class):
    """Collect a balanced set of positive and negative sentences for one category."""
    config_name = f"raw_review_{category}"
    positives = []
    negatives = []

    print(f"\nStreaming category: {category}")
    print(f"Target: {target_per_class} positive + {target_per_class} negative sentences")

    dataset = load_dataset(
        DATASET_NAME,
        config_name,
        split="full",
        streaming=True,
        trust_remote_code=True,
    )

    reviewed = 0
    for review in dataset:
        if len(positives) >= target_per_class and len(negatives) >= target_per_class:
            break

        rating = review.get("rating")
        text = review.get("text", "")
        if rating is None or not text:
            continue

        label = get_label(float(rating))
        if label is None:
            continue

        sentences = tokenize_review(text)
        for sent in sentences:
            sent = sent.strip()
            if not quality_ok(sent):
                continue

            record = {
                "sentence": sent,
                "label": label,
                "category": category,
                "augmented": False,
            }

            if label == 1 and len(positives) < target_per_class:
                positives.append(record)
            elif label == 0 and len(negatives) < target_per_class:
                negatives.append(record)

            if len(positives) >= target_per_class and len(negatives) >= target_per_class:
                break

        reviewed += 1
        if reviewed % 5000 == 0:
            print(
                f"  Reviews scanned: {reviewed:>7} | "
                f"pos: {len(positives):>4}/{target_per_class} | "
                f"neg: {len(negatives):>4}/{target_per_class}"
            )

    if len(positives) < target_per_class or len(negatives) < target_per_class:
        raise ValueError(
            f"Category '{category}' did not yield enough sentences. "
            f"Needed {target_per_class} per class, got "
            f"{len(positives)} positive and {len(negatives)} negative."
        )

    return positives, negatives


def print_stats(records):
    pos = sum(1 for r in records if r["label"] == 1)
    neg = sum(1 for r in records if r["label"] == 0)
    categories = sorted({r["category"] for r in records})
    print(f"\n  {'='*44}")
    print(f"  Total sentences : {len(records)}")
    print(f"  Positive        : {pos}")
    print(f"  Negative        : {neg}")
    print(f"  Categories      : {len(categories)}")
    print(f"  {'='*44}")
    for category in categories:
        cat_rows = [r for r in records if r["category"] == category]
        cat_pos = sum(1 for r in cat_rows if r["label"] == 1)
        cat_neg = sum(1 for r in cat_rows if r["label"] == 0)
        print(f"  {category:<24} total={len(cat_rows):>4}  pos={cat_pos:>4}  neg={cat_neg:>4}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Build balanced Amazon review training data across all categories."
    )
    parser.add_argument(
        "--output_path",
        default=str(PROCESSED_DIR / "training_data.tsv"),
        help="Output TSV file path"
    )
    parser.add_argument(
        "--target_per_category",
        type=int,
        default=200,
        help="Number of positive and negative sentences to sample per category"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    if args.target_per_category <= 0:
        raise ValueError("--target_per_category must be positive.")

    random.seed(args.seed)
    categories = discover_categories()
    print(f"Discovered {len(categories)} review categories.")

    all_records = []
    for category in categories:
        positives, negatives = collect_category_sentences(
            category,
            args.target_per_category,
        )
        random.shuffle(positives)
        random.shuffle(negatives)
        all_records.extend(positives)
        all_records.extend(negatives)

    random.shuffle(all_records)
    print_stats(all_records)

    fieldnames = ["sentence", "label", "category", "augmented"]
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(all_records)

    print(f"Saved {len(all_records)} sentences to: {output_path}")


if __name__ == "__main__":
    main()
