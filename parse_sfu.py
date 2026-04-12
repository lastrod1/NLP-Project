import os
import csv
import argparse
from xml.etree import ElementTree as ET

def get_label(filename):
    """Derive sentiment label from filename prefix.
    yes* -> 1 (positive), no* -> 0 (negative)
    """
    basename = os.path.basename(filename).lower()
    if basename.startswith("yes"):
        return 1
    elif basename.startswith("no"):
        return 0
    else:
        return None  # skip unknown


def reconstruct_sentence(sentence_element):
    """Reconstruct raw sentence text from all <W> tokens in a <SENTENCE> tag.
    Joins tokens with spaces and cleans up punctuation spacing.
    """
    tokens = [w.text for w in sentence_element.iter("W") if w.text]
    if not tokens:
        return ""

    text = ""
    punct_no_space_before = set(".,!?;:)'\"")
    punct_no_space_after  = set("(\"'")

    for i, token in enumerate(tokens):
        if i == 0:
            text += token
        elif token in punct_no_space_before:
            text += token
        elif tokens[i - 1] in punct_no_space_after:
            text += token
        else:
            text += " " + token

    return text.strip()


def sentence_has_speculation(sentence_element):
    """Return True if the sentence contains at least one speculation cue."""
    for cue in sentence_element.iter("cue"):
        if cue.attrib.get("type") == "speculation":
            return True
    return False

def parse_corpus(corpus_dir):
    """Walk corpus directory, parse all XML files, return list of sentence dicts."""
    records = []

    for domain in sorted(os.listdir(corpus_dir)):
        domain_path = os.path.join(corpus_dir, domain)
        if not os.path.isdir(domain_path):
            continue

        for fname in sorted(os.listdir(domain_path)):
            if not fname.endswith(".xml"):
                continue

            fpath = os.path.join(domain_path, fname)
            label = get_label(fname)
            if label is None:
                print(f"  [SKIP] unrecognised filename: {fname}")
                continue

            try:
                tree = ET.parse(fpath)
                root = tree.getroot()
            except ET.ParseError as e:
                print(f"  [ERROR] failed to parse {fpath}: {e}")
                continue

            for sentence_el in root.iter("SENTENCE"):
                text = reconstruct_sentence(sentence_el)

                # Skip empty or very short sentences (likely artefacts)
                if len(text.split()) < 4:
                    continue

                is_hedged = sentence_has_speculation(sentence_el)

                records.append({
                    "sentence"    : text,
                    "label"       : label,
                    "domain"      : domain,
                    "is_hedged"   : is_hedged,
                    "source_file" : fname,
                })

    return records

def print_stats(records):
    """Print a summary of the parsed corpus."""
    total     = len(records)
    hedged    = [r for r in records if r["is_hedged"]]
    direct    = [r for r in records if not r["is_hedged"]]
    positive  = [r for r in records if r["label"] == 1]
    negative  = [r for r in records if r["label"] == 0]

    print(f"\n{'='*50}")
    print(f"  Total sentences       : {total}")
    print(f"  Hedged sentences      : {len(hedged)}")
    print(f"  Direct sentences      : {len(direct)}")
    print(f"  Positive (yes) label  : {len(positive)}")
    print(f"  Negative (no) label   : {len(negative)}")
    print(f"{'='*50}")

    print("\n  Per-domain breakdown:")
    domains = sorted(set(r["domain"] for r in records))
    for domain in domains:
        d_records = [r for r in records if r["domain"] == domain]
        d_hedged  = sum(1 for r in d_records if r["is_hedged"])
        print(f"    {domain:<12} total={len(d_records):>4}  hedged={d_hedged:>4}")

    print(f"\n  Hedged breakdown by label:")
    print(f"    Hedged + positive : {sum(1 for r in hedged if r['label'] == 1)}")
    print(f"    Hedged + negative : {sum(1 for r in hedged if r['label'] == 0)}")
    print(f"    Direct + positive : {sum(1 for r in direct if r['label'] == 1)}")
    print(f"    Direct + negative : {sum(1 for r in direct if r['label'] == 0)}")
    print()

def main():
    parser = argparse.ArgumentParser(description="Parse SFU Review Corpus into benchmark TSV.")
    parser.add_argument(
        "--corpus_dir",
        required=True,
        help="Path to the SFU_Review_Corpus_Negation_Speculation directory"
    )
    parser.add_argument(
        "--output_path",
        default="sfu_benchmark.tsv",
        help="Output TSV file path (default: sfu_benchmark.tsv)"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.corpus_dir):
        print(f"[ERROR] corpus_dir not found: {args.corpus_dir}")
        return

    print(f"Parsing corpus from: {args.corpus_dir}")
    records = parse_corpus(args.corpus_dir)
    print_stats(records)

    # Write TSV (tab delimiter avoids column bleeding from commas in sentences)
    fieldnames = ["sentence", "label", "domain", "is_hedged", "source_file"]
    with open(args.output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(records)

    print(f"Saved {len(records)} sentences to: {args.output_path}")


if __name__ == "__main__":
    main()
