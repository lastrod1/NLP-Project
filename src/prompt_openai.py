"""
prompt_openai.py
----------------
Zero-shot sentiment classification using the OpenAI Chat Completions API.
Evaluates on the HedgeBERT benchmark and reports accuracy separately on
hedged vs. direct subsets — the same metrics as the fine-tuned models.

No fine-tuning. Each sentence is sent as a standalone prompt.

Usage:
    python prompt_openai.py \
        --bench_path  data/processed/benchmark.tsv \
        --output_dir  outputs/gpt4o_mini_output \
        --model_name  gpt-4o-mini \
        --api_key     sk-...

    # API key can also be set via environment variable:
    export OPENAI_API_KEY=sk-...
    python prompt_openai.py --bench_path ... --output_dir ...

Output:
    <output_dir>/results.json         -- all metrics
    <output_dir>/results_summary.txt  -- human-readable summary
    <output_dir>/predictions.tsv      -- per-sentence predictions

"""

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR   = PROJECT_ROOT / "outputs"

SYSTEM_PROMPT = (
    "You are a sentiment classifier. "
    "Classify the sentiment of the given sentence as either positive or negative. "
    "Respond with exactly one word: positive or negative. No explanation."
)


def normalize_bool(series):
    return series.map(
        lambda v: v if isinstance(v, bool) else str(v).strip().lower() == "true"
    )


def build_user_prompt(sentence: str) -> str:
    return f'Sentence: "{sentence}"\n\nSentiment:'


def parse_response(text: str) -> int | None:
    """Return 1 (positive), 0 (negative), or None if unparseable."""
    clean = text.strip().lower()
    if "positive" in clean:
        return 1
    if "negative" in clean:
        return 0
    return None


def classify_sentence(client: OpenAI, model: str, sentence: str,
                       retry_delay: float = 5.0) -> tuple[int | None, str]:
    """Call the API and return (predicted_label, raw_response_text)."""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": build_user_prompt(sentence)},
                ],
                max_tokens=5,
                temperature=0.0,
            )
            raw = response.choices[0].message.content or ""
            return parse_response(raw), raw.strip()
        except Exception as e:
            print(f"  [API error, attempt {attempt + 1}/3]: {e}")
            if attempt < 2:
                time.sleep(retry_delay)
    return None, "ERROR"


def evaluate_subset(df: pd.DataFrame, preds: list[int | None]) -> dict:
    """Compute accuracy and F1 for a subset given parallel pred list."""
    correct  = sum(p == l for p, l in zip(preds, df["label"]) if p is not None)
    total    = sum(1 for p in preds if p is not None)
    skipped  = sum(1 for p in preds if p is None)
    accuracy = round(correct / total, 4) if total > 0 else 0.0

    tp = sum(1 for p, l in zip(preds, df["label"]) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(preds, df["label"]) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(preds, df["label"]) if p == 0 and l == 1)
    tn = sum(1 for p, l in zip(preds, df["label"]) if p == 0 and l == 0)

    prec_pos = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_pos  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_pos   = (2 * prec_pos * rec_pos / (prec_pos + rec_pos)
                if (prec_pos + rec_pos) > 0 else 0.0)

    prec_neg = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec_neg  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_neg   = (2 * prec_neg * rec_neg / (prec_neg + rec_neg)
                if (prec_neg + rec_neg) > 0 else 0.0)

    macro_f1 = round((f1_pos + f1_neg) / 2, 4)

    return {
        "n":        len(df),
        "correct":  correct,
        "skipped":  skipped,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }


def write_summary(results: dict, output_dir: str, model_name: str):
    lines = [
        "=" * 60,
        f"  Zero-shot: {model_name}",
        "=" * 60,
        "",
        "[Benchmark — Overall]",
        f"  Accuracy : {results['benchmark_overall']['accuracy']}",
        f"  Macro F1 : {results['benchmark_overall']['macro_f1']}",
        f"  Skipped  : {results['benchmark_overall']['skipped']} unparseable responses",
        "",
        "[Benchmark — Hedged vs Direct Subsets]",
    ]
    for subset in ["hedged", "direct"]:
        s = results["benchmark_subsets"].get(subset, {})
        if s:
            lines.append(
                f"  {subset:<8}  n={s['n']:>3}  "
                f"acc={s['accuracy']:.4f}  macro_f1={s['macro_f1']:.4f}"
            )
    gap = results["benchmark_subsets"].get("accuracy_gap_direct_minus_hedged")
    if gap is not None:
        lines.append(f"\n  Accuracy gap (direct - hedged) : {gap:+.4f}")
    lines.append("=" * 60)

    path = os.path.join(output_dir, "results_summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nSummary saved to: {path}")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot sentiment prompting via OpenAI Chat Completions API."
    )
    parser.add_argument(
        "--bench_path",
        default=str(PROCESSED_DIR / "benchmark.tsv"),
    )
    parser.add_argument(
        "--output_dir",
        default=str(OUTPUTS_DIR / "gpt4o_mini_output"),
    )
    parser.add_argument(
        "--model_name",
        default="gpt-4o-mini",
        help="OpenAI model name (e.g. gpt-4o-mini, gpt-4o, gpt-3.5-turbo)",
    )
    parser.add_argument(
        "--api_key",
        default=os.environ.get("OPENAI_API_KEY"),
        help="OpenAI API key. Falls back to OPENAI_API_KEY env var.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds to wait between API calls to avoid rate limits (default: 0.5)",
    )
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError(
            "No API key found. Pass --api_key or set the OPENAI_API_KEY env var."
        )

    os.makedirs(args.output_dir, exist_ok=True)

    bench_df = pd.read_csv(args.bench_path, sep="\t")
    bench_df["is_hedged"] = normalize_bool(bench_df["is_hedged"])
    print(f"\nLoaded benchmark: {len(bench_df)} sentences")
    print(f"  Hedged : {bench_df['is_hedged'].sum()}")
    print(f"  Direct : {(~bench_df['is_hedged']).sum()}")

    client = OpenAI(api_key=args.api_key)

    print(f"\nModel  : {args.model_name}")
    print(f"Delay  : {args.delay}s between calls")
    print(f"\nRunning zero-shot classification on {len(bench_df)} sentences...\n")

    all_preds = []
    all_raws  = []

    for i, (_, row) in enumerate(bench_df.iterrows()):
        pred, raw = classify_sentence(client, args.model_name, row["sentence"])
        all_preds.append(pred)
        all_raws.append(raw)

        status = "✓" if pred == row["label"] else ("✗" if pred is not None else "?")
        if (i + 1) % 20 == 0 or i == 0:
            parsed_so_far = sum(1 for p in all_preds if p is not None)
            correct_so_far = sum(
                1 for p, l in zip(all_preds, bench_df["label"].iloc[:i+1])
                if p is not None and p == l
            )
            running_acc = correct_so_far / parsed_so_far if parsed_so_far else 0
            print(f"  [{i+1:>3}/{len(bench_df)}] {status}  running acc={running_acc:.3f}")

        if args.delay > 0:
            time.sleep(args.delay)

    # Overall metrics
    overall = evaluate_subset(bench_df, all_preds)
    print(f"\nOverall  acc={overall['accuracy']:.4f}  macro_f1={overall['macro_f1']:.4f}  "
          f"skipped={overall['skipped']}")

    # Subset metrics
    subset_results = {}
    for subset_name, mask in [("hedged", bench_df["is_hedged"] == True),
                               ("direct", bench_df["is_hedged"] == False)]:
        sub_df    = bench_df[mask].reset_index(drop=True)
        sub_preds = [all_preds[i] for i, m in enumerate(mask) if m]
        result    = evaluate_subset(sub_df, sub_preds)
        subset_results[subset_name] = result
        print(f"  {subset_name:<8}  n={result['n']:>3}  "
              f"acc={result['accuracy']:.4f}  macro_f1={result['macro_f1']:.4f}")

    if "hedged" in subset_results and "direct" in subset_results:
        gap = round(
            subset_results["direct"]["accuracy"] - subset_results["hedged"]["accuracy"], 4
        )
        subset_results["accuracy_gap_direct_minus_hedged"] = gap
        print(f"\n  Accuracy gap (direct - hedged): {gap:+.4f}")

    # Save predictions TSV
    bench_df["prediction"] = all_preds
    bench_df["raw_response"] = all_raws
    bench_df["correct"] = bench_df["prediction"] == bench_df["label"]
    pred_path = os.path.join(args.output_dir, "predictions.tsv")
    bench_df.to_csv(pred_path, sep="\t", index=False)
    print(f"\nPredictions saved to: {pred_path}")

    all_results = {
        "model":              args.model_name,
        "mode":               "zero_shot_prompting",
        "total_sentences":    len(bench_df),
        "benchmark_overall":  overall,
        "benchmark_subsets":  subset_results,
    }

    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to: {results_path}")

    write_summary(all_results, args.output_dir, args.model_name)


if __name__ == "__main__":
    main()
