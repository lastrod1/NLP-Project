"""
prompt_llama.py
---------------
Zero-shot sentiment classification using a local LLaMA 2 (or any causal LM)
loaded from HuggingFace. No fine-tuning — each sentence is sent as a
standalone prompt and the model's generated text is parsed for
'positive' or 'negative'.

Evaluates on the HedgeBERT benchmark and reports the same hedged vs. direct
accuracy gap metrics as the fine-tuned models.

Usage:
    # LLaMA 2 7B Chat (requires HF access approval + token)
    python prompt_llama.py \
        --model_name  meta-llama/Llama-2-7b-chat-hf \
        --bench_path  data/processed/benchmark.tsv \
        --output_dir  outputs/llama2_output \
        --hf_token    hf_... \
        --load_in_4bit            # recommended for 24GB GPUs

    # TinyLlama (no token required, good for testing)
    python prompt_llama.py \
        --model_name  TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --output_dir  outputs/tinyllama_output

    # HF token can also be set via environment variable:
    export HF_TOKEN=hf_...

Output:
    <output_dir>/results.json         -- all metrics
    <output_dir>/results_summary.txt  -- human-readable summary
    <output_dir>/predictions.tsv      -- per-sentence predictions

Requirements:
    pip install transformers torch accelerate pandas
    pip install bitsandbytes   # only needed for --load_in_4bit
"""

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR   = PROJECT_ROOT / "outputs"

SYSTEM_PROMPT = (
    "You are a sentiment classifier. "
    "Classify the sentiment of the given sentence as either positive or negative. "
    "Respond with exactly one word: positive or negative. No explanation."
)

# LLaMA 2 Chat uses this exact template format
LLAMA2_CHAT_TEMPLATE = (
    "<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]"
)

# Generic instruction template for non-LLaMA chat models
GENERIC_TEMPLATE = (
    "### Instruction:\n{system}\n\n### Input:\n{user}\n\n### Response:\n"
)


def normalize_bool(series):
    return series.map(
        lambda v: v if isinstance(v, bool) else str(v).strip().lower() == "true"
    )


def is_llama2_chat(model_name: str) -> bool:
    name = model_name.lower()
    return "llama-2" in name and "chat" in name


def is_chat_model(model_name: str) -> bool:
    name = model_name.lower()
    return any(k in name for k in ["chat", "instruct", "it"])


def build_prompt(sentence: str, model_name: str) -> str:
    user_msg = f'Sentence: "{sentence}"\n\nSentiment (positive or negative):'
    if is_llama2_chat(model_name):
        return LLAMA2_CHAT_TEMPLATE.format(system=SYSTEM_PROMPT, user=user_msg)
    if is_chat_model(model_name):
        # Try HuggingFace chat template via tokenizer if available,
        # otherwise fall back to generic format
        return GENERIC_TEMPLATE.format(system=SYSTEM_PROMPT, user=user_msg)
    # Base / completion models
    return (
        f"Classify the sentiment of the following sentence as positive or negative.\n\n"
        f'Sentence: "{sentence}"\n\nSentiment:'
    )


def parse_response(generated_text: str, prompt: str) -> int | None:
    """
    Strip the prompt from the generated output, then look for
    'positive' or 'negative' in the first non-empty line of the response.
    Returns 1, 0, or None if unparseable.
    """
    # Remove the prompt prefix if the model echoed it back
    response = generated_text[len(prompt):].strip() if generated_text.startswith(prompt) else generated_text.strip()

    # Check first 50 characters only — the answer should be immediate
    snippet = response[:50].lower()
    if re.search(r"\bpositive\b", snippet):
        return 1
    if re.search(r"\bnegative\b", snippet):
        return 0

    # Wider search in case model added a preamble
    full_lower = response.lower()
    pos_idx = full_lower.find("positive")
    neg_idx = full_lower.find("negative")
    if pos_idx == -1 and neg_idx == -1:
        return None
    if pos_idx == -1:
        return 0
    if neg_idx == -1:
        return 1
    return 1 if pos_idx < neg_idx else 0


def resolve_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_and_tokenizer(model_name: str, hf_token: str | None,
                              load_in_4bit: bool, device: str):
    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {model_name}")
    if load_in_4bit:
        if device != "cuda":
            raise ValueError("--load_in_4bit requires a CUDA GPU.")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
        )
        if device != "cuda":
            model = model.to(device)

    model.eval()
    return model, tokenizer


def classify_sentence(model, tokenizer, prompt: str, device: str,
                       max_new_tokens: int = 10) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy — deterministic output
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def evaluate_subset(df: pd.DataFrame, preds: list[int | None]) -> dict:
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
        description="Zero-shot sentiment prompting with a local HuggingFace causal LM."
    )
    parser.add_argument(
        "--model_name",
        default="meta-llama/Llama-2-7b-chat-hf",
        help="HuggingFace model ID (default: meta-llama/Llama-2-7b-chat-hf)",
    )
    parser.add_argument(
        "--model_label",
        default=None,
        help="Display name for results (defaults to model_name)",
    )
    parser.add_argument(
        "--bench_path",
        default=str(PROCESSED_DIR / "benchmark.tsv"),
    )
    parser.add_argument(
        "--output_dir",
        default=str(OUTPUTS_DIR / "llama2_output"),
    )
    parser.add_argument(
        "--hf_token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
        help="HuggingFace token for gated models (LLaMA 2 requires this).",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4-bit quantization via bitsandbytes (requires CUDA).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=10,
        help="Max tokens to generate per sentence (default: 10).",
    )
    args = parser.parse_args()

    model_label = args.model_label or args.model_name
    os.makedirs(args.output_dir, exist_ok=True)

    device = resolve_device()
    print(f"Device : {device}")
    if args.load_in_4bit:
        print("Mode   : 4-bit quantization (QLoRA-style loading)")
    else:
        print("Mode   : full precision (bfloat16 on CUDA, float32 on CPU/MPS)")

    bench_df = pd.read_csv(args.bench_path, sep="\t")
    bench_df["is_hedged"] = normalize_bool(bench_df["is_hedged"])
    print(f"\nLoaded benchmark: {len(bench_df)} sentences")
    print(f"  Hedged : {bench_df['is_hedged'].sum()}")
    print(f"  Direct : {(~bench_df['is_hedged']).sum()}")

    model, tokenizer = load_model_and_tokenizer(
        args.model_name, args.hf_token, args.load_in_4bit, device
    )

    print(f"\nRunning zero-shot classification on {len(bench_df)} sentences...\n")

    all_preds  = []
    all_raws   = []
    all_prompts = []

    for i, (_, row) in enumerate(bench_df.iterrows()):
        prompt   = build_prompt(row["sentence"], args.model_name)
        raw_text = classify_sentence(
            model, tokenizer, prompt, device, args.max_new_tokens
        )
        pred = parse_response(raw_text, prompt)

        all_prompts.append(prompt)
        all_raws.append(raw_text[len(prompt):].strip() if raw_text.startswith(prompt) else raw_text.strip())
        all_preds.append(pred)

        status = "✓" if pred == row["label"] else ("✗" if pred is not None else "?")
        if (i + 1) % 20 == 0 or i == 0:
            parsed_so_far  = sum(1 for p in all_preds if p is not None)
            correct_so_far = sum(
                1 for p, l in zip(all_preds, bench_df["label"].iloc[:i+1])
                if p is not None and p == l
            )
            running_acc = correct_so_far / parsed_so_far if parsed_so_far else 0
            print(f"  [{i+1:>3}/{len(bench_df)}] {status}  running acc={running_acc:.3f}")

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
    bench_df["prediction"]   = all_preds
    bench_df["raw_response"] = all_raws
    bench_df["correct"]      = bench_df["prediction"] == bench_df["label"]
    pred_path = os.path.join(args.output_dir, "predictions.tsv")
    bench_df.to_csv(pred_path, sep="\t", index=False)
    print(f"\nPredictions saved to: {pred_path}")

    all_results = {
        "model":             args.model_name,
        "model_label":       model_label,
        "mode":              "zero_shot_prompting",
        "load_in_4bit":      args.load_in_4bit,
        "total_sentences":   len(bench_df),
        "benchmark_overall": overall,
        "benchmark_subsets": subset_results,
    }

    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to: {results_path}")

    write_summary(all_results, args.output_dir, model_label)


if __name__ == "__main__":
    main()
