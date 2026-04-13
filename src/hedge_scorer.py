import argparse
import json
import re
from pathlib import Path
import pandas as pd


# Check learn_weights.py
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
DEFAULT_WEIGHTS_PATH = PROJECT_ROOT / "data" / "processed" / "weights.json"


def _compile_cue_patterns():
    """Compile whole-word patterns so short cues do not match inside other words."""
    patterns = {}
    for category, cues in CUE_INVENTORY.items():
        patterns[category] = []
        for cue in cues:
            pattern = re.compile(rf"\b{re.escape(cue)}\b", re.IGNORECASE)
            patterns[category].append((cue, pattern))
    return patterns

class HedgeScorer:
    """
    Scores a sentence for hedge intensity using learned category weights
    and a diminishing-returns aggregation formula.
    """

    def __init__(self, weights_path=DEFAULT_WEIGHTS_PATH):
        with open(weights_path, "r") as f:
            self.weights = json.load(f)
        self.cue_patterns = _compile_cue_patterns()
        self._validate_weights()

    def _validate_weights(self):
        """Check all cue categories have a weight."""
        for cat in CUE_INVENTORY:
            if cat not in self.weights:
                raise ValueError(
                    f"Missing weight for category '{cat}' in weights file."
                )
            weight = self.weights[cat]
            if not isinstance(weight, (int, float)):
                raise ValueError(f"Weight for category '{cat}' must be numeric.")
            if not 0.0 <= float(weight) <= 1.0:
                raise ValueError(
                    f"Weight for category '{cat}' must be in [0.0, 1.0]."
                )

    def _split_at_conjunctions(self, sentence):
        """Split sentence into clauses at conjunction boundaries."""
        pattern = r'\b(' + '|'.join(
            re.escape(c) for c in CONJUNCTIONS) + r')\b'
        clauses = re.split(pattern, sentence.lower(), flags=re.IGNORECASE)
        clauses = [c.strip() for c in clauses
                   if c.strip() and c.strip().lower() not in CONJUNCTIONS]
        return clauses if clauses else [sentence.lower()]

    def _get_clause_with_cue(self, sentence_lower, cue):
        """Return the clause containing the cue."""
        clauses = self._split_at_conjunctions(sentence_lower)
        for clause in clauses:
            if cue in clause:
                return clause
        return sentence_lower

    def _detect_cues(self, sentence):
        """
        Detect all hedge cues in a sentence.
        Returns list of (category, cue, weight) tuples.
        Applies scope simplification per clause.
        """
        sent_lower = sentence.lower()
        detected = []
        seen_cues = set()  # avoid double-counting same cue

        # Sort cues by length descending so multi-word cues match first
        all_cues = []
        for cat, cues in CUE_INVENTORY.items():
            for cue in cues:
                all_cues.append((cat, cue))
        all_cues.sort(key=lambda x: len(x[1]), reverse=True)

        for cat, cue in all_cues:
            pattern = next(
                compiled for compiled_cue, compiled in self.cue_patterns[cat]
                if compiled_cue == cue
            )
            if pattern.search(sent_lower) and cue not in seen_cues:
                clause = self._get_clause_with_cue(sent_lower, cue)
                if pattern.search(clause):
                    weight = self.weights[cat]
                    detected.append((cat, cue, weight))
                    seen_cues.add(cue)

        return detected

    def score(self, sentence):
        """
        Score a sentence for hedge intensity.
        Returns float in [0.0, 1.0].

        Formula: score = 1 − ∏(1 − wᵢ) for all detected cues i
        """
        detected = self._detect_cues(sentence)

        if not detected:
            return 0.0

        product = 1.0
        for _, _, weight in detected:
            product *= (1.0 - weight)

        return round(1.0 - product, 4)

    def score_with_detail(self, sentence):
        """
        Score a sentence and return full breakdown.
        Returns dict with score, detected cues, and per-cue weights.
        """
        detected = self._detect_cues(sentence)
        final_score = self.score(sentence)

        return {
            "sentence" : sentence,
            "score"    : final_score,
            "cues"     : [
                {"category": cat, "cue": cue, "weight": w}
                for cat, cue, w in detected
            ],
            "n_cues"   : len(detected),
        }


_DEFAULT_SCORER = None


def score_hedge(sentence, weights_path=DEFAULT_WEIGHTS_PATH):
    """
    Backward-compatible helper used by training scripts.
    Lazily caches the default scorer for repeated calls.
    """
    global _DEFAULT_SCORER
    if Path(weights_path) == DEFAULT_WEIGHTS_PATH:
        if _DEFAULT_SCORER is None:
            _DEFAULT_SCORER = HedgeScorer(weights_path)
        scorer = _DEFAULT_SCORER
    else:
        scorer = HedgeScorer(weights_path)
    return scorer.score(sentence)

def main():
    parser = argparse.ArgumentParser(
        description="Score sentences for hedge intensity."
    )
    parser.add_argument(
        "--weights",
        default=str(DEFAULT_WEIGHTS_PATH),
        help="Path to weights.json from learn_weights.py"
    )
    parser.add_argument(
        "--sentence",
        default=None,
        help="Single sentence to score (optional)"
    )
    parser.add_argument(
        "--input_tsv",
        default=None,
        help="TSV file to score (must have 'sentence' column)"
    )
    parser.add_argument(
        "--output_tsv",
        default=None,
        help="Output TSV with added 'hedge_score' column"
    )
    args = parser.parse_args()

    scorer = HedgeScorer(args.weights)

    # Single sentence mode
    if args.sentence:
        result = scorer.score_with_detail(args.sentence)
        print(f"\nSentence : {result['sentence']}")
        print(f"Score    : {result['score']}")
        print(f"Cues ({result['n_cues']}):")
        for cue_info in result["cues"]:
            print(f"  [{cue_info['category']}] "
                  f"'{cue_info['cue']}' → weight={cue_info['weight']}")
        return

    # TSV batch mode
    if args.input_tsv:
        if not args.output_tsv:
            print("[ERROR] --output_tsv required when using --input_tsv")
            return

        print(f"\nScoring TSV: {args.input_tsv}")
        df = pd.read_csv(args.input_tsv, sep="\t")

        df["hedge_score"] = df["sentence"].apply(scorer.score)

        print(f"  Scored {len(df)} sentences")
        print(f"  Mean score         : {df['hedge_score'].mean():.4f}")
        print(f"  Sentences score>0  : {(df['hedge_score'] > 0).sum()}")
        print(f"  Sentences score>0.5: {(df['hedge_score'] > 0.5).sum()}")

        df.to_csv(args.output_tsv, sep="\t", index=False)
        print(f"  Saved to: {args.output_tsv}")
        return

    print("Provide --sentence or --input_tsv to score.")


if __name__ == "__main__":
    main()
