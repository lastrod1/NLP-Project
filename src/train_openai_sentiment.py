import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI
from sklearn.metrics import accuracy_score, classification_report, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OPENAI_DIR = PROCESSED_DIR / "openai"

SYSTEM_PROMPT = (
    "You are a sentiment classifier for short review sentences. "
    "Reply with exactly one label: POSITIVE or NEGATIVE."
)


def normalize_bool(series):
    return series.map(
        lambda value: value if isinstance(value, bool)
        else str(value).strip().lower() == "true"
    )


def label_to_text(label):
    return "POSITIVE" if int(label) == 1 else "NEGATIVE"


def text_to_label(text):
    cleaned = (text or "").strip().upper()
    if "POSITIVE" in cleaned:
        return 1
    if "NEGATIVE" in cleaned:
        return 0
    return None


def example_to_messages(sentence, label):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Sentence: {sentence}"},
            {"role": "assistant", "content": label_to_text(label)},
        ]
    }


def prepare_jsonl(train_path, output_train, output_val, val_split=0.1, seed=42):
    df = pd.read_csv(train_path, sep="\t")
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    val_size = max(1, int(len(df) * val_split))
    val_df = df.iloc[:val_size].reset_index(drop=True)
    train_df = df.iloc[val_size:].reset_index(drop=True)

    output_train.parent.mkdir(parents=True, exist_ok=True)
    with output_train.open("w", encoding="utf-8") as train_file:
        for row in train_df.itertuples(index=False):
            train_file.write(json.dumps(example_to_messages(row.sentence, row.label)) + "\n")

    with output_val.open("w", encoding="utf-8") as val_file:
        for row in val_df.itertuples(index=False):
            val_file.write(json.dumps(example_to_messages(row.sentence, row.label)) + "\n")

    print(f"Prepared training JSONL: {output_train}")
    print(f"Prepared validation JSONL: {output_val}")


def upload_and_create_job(client, train_jsonl, val_jsonl, base_model, suffix=None):
    train_file = client.files.create(file=train_jsonl.open("rb"), purpose="fine-tune")
    val_file = client.files.create(file=val_jsonl.open("rb"), purpose="fine-tune")
    job = client.fine_tuning.jobs.create(
        model=base_model,
        training_file=train_file.id,
        validation_file=val_file.id,
        suffix=suffix,
    )
    return job


def wait_for_job(client, job_id, poll_seconds=30):
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        print(f"Job {job.id}: status={job.status}")
        if job.status in {"succeeded", "failed", "cancelled"}:
            return job
        time.sleep(poll_seconds)


def classify_sentence(client, model_name, sentence):
    response = client.responses.create(
        model=model_name,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Sentence: {sentence}"},
        ],
        max_output_tokens=8,
    )
    return response.output_text


def evaluate_model(client, model_name, bench_path, output_dir):
    bench_df = pd.read_csv(bench_path, sep="\t")
    bench_df["is_hedged"] = normalize_bool(bench_df["is_hedged"])

    predictions = []
    for row in bench_df.itertuples(index=False):
        text = classify_sentence(client, model_name, row.sentence)
        predictions.append(text_to_label(text))

    parsed_preds = [pred if pred is not None else 0 for pred in predictions]
    labels = bench_df["label"].tolist()
    overall_acc = accuracy_score(labels, parsed_preds)
    overall_f1 = f1_score(labels, parsed_preds, average="macro")

    results = {
        "model": model_name,
        "benchmark_overall": {
            "accuracy": round(overall_acc, 4),
            "macro_f1": round(overall_f1, 4),
        },
        "benchmark_subsets": {},
    }

    for subset_name, mask in [
        ("hedged", bench_df["is_hedged"] == True),
        ("direct", bench_df["is_hedged"] == False),
    ]:
        subset_indices = bench_df.index[mask].tolist()
        sub_df = bench_df.loc[subset_indices].reset_index(drop=True)
        sub_preds = [parsed_preds[i] for i in subset_indices]
        sub_labels = sub_df["label"].tolist()
        sub_acc = accuracy_score(sub_labels, sub_preds)
        sub_f1 = f1_score(sub_labels, sub_preds, average="macro")
        report = classification_report(
            sub_labels,
            sub_preds,
            target_names=["negative", "positive"],
            output_dict=True,
        )
        results["benchmark_subsets"][subset_name] = {
            "n": len(sub_df),
            "accuracy": round(sub_acc, 4),
            "macro_f1": round(sub_f1, 4),
            "report": report,
        }

    if "hedged" in results["benchmark_subsets"] and "direct" in results["benchmark_subsets"]:
        gap = (
            results["benchmark_subsets"]["direct"]["accuracy"]
            - results["benchmark_subsets"]["hedged"]["accuracy"]
        )
        results["benchmark_subsets"]["accuracy_gap_direct_minus_hedged"] = round(gap, 4)

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    summary_path = output_dir / "results_summary.txt"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    summary_lines = [
        "=" * 60,
        f"  OpenAI Fine-Tuned Model Results ({model_name})",
        "=" * 60,
        "",
        "[Benchmark — Overall]",
        f"  Accuracy : {results['benchmark_overall']['accuracy']}",
        f"  Macro F1 : {results['benchmark_overall']['macro_f1']}",
        "",
        "[Benchmark — Hedged vs Direct Subsets]",
    ]
    for subset in ["hedged", "direct"]:
        subset_result = results["benchmark_subsets"].get(subset, {})
        if subset_result:
            summary_lines.append(
                f"  {subset:<8}  n={subset_result['n']:>3}  "
                f"acc={subset_result['accuracy']:.4f}  macro_f1={subset_result['macro_f1']:.4f}"
            )
    gap = results["benchmark_subsets"].get("accuracy_gap_direct_minus_hedged")
    if gap is not None:
        summary_lines.append("")
        summary_lines.append(f"  Accuracy gap (direct - hedged) : {gap:+.4f}")

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print(f"Saved OpenAI benchmark results to: {results_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare, submit, and evaluate an OpenAI fine-tuned sentiment model."
    )
    parser.add_argument(
        "--train_path",
        default=str(PROCESSED_DIR / "training_combined.tsv"),
    )
    parser.add_argument(
        "--bench_path",
        default=str(PROCESSED_DIR / "benchmark.tsv"),
    )
    parser.add_argument(
        "--train_jsonl",
        default=str(OPENAI_DIR / "train.jsonl"),
    )
    parser.add_argument(
        "--val_jsonl",
        default=str(OPENAI_DIR / "val.jsonl"),
    )
    parser.add_argument(
        "--output_dir",
        default=str(OUTPUTS_DIR / "openai_gpt_output"),
    )
    parser.add_argument(
        "--base_model",
        default="gpt-4.1-mini-2025-04-14",
        help="OpenAI base model for supervised fine-tuning",
    )
    parser.add_argument("--job_id", default=None)
    parser.add_argument("--fine_tuned_model", default=None)
    parser.add_argument("--suffix", default="hedged-sentiment")
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--submit_only", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--evaluate_only", action="store_true")
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_jsonl = Path(args.train_jsonl)
    val_jsonl = Path(args.val_jsonl)
    output_dir = Path(args.output_dir)

    if not args.submit_only and not args.evaluate_only:
        prepare_jsonl(
            args.train_path,
            train_jsonl,
            val_jsonl,
            val_split=args.val_split,
            seed=args.seed,
        )
        if args.prepare_only:
            return

    client = OpenAI()

    if not args.evaluate_only:
        job = upload_and_create_job(
            client,
            train_jsonl,
            val_jsonl,
            args.base_model,
            suffix=args.suffix,
        )
        print(f"Created fine-tuning job: {job.id}")
        print(f"Base model: {args.base_model}")

        if not args.wait:
            return

        final_job = wait_for_job(client, job.id)
        print(f"Final status: {final_job.status}")
        if final_job.status != "succeeded":
            raise RuntimeError(f"Fine-tuning job did not succeed: {final_job.status}")
        model_name = final_job.fine_tuned_model
    else:
        if not args.fine_tuned_model:
            raise ValueError("--fine_tuned_model is required with --evaluate_only")
        model_name = args.fine_tuned_model

    evaluate_model(client, model_name, args.bench_path, output_dir)


if __name__ == "__main__":
    main()
