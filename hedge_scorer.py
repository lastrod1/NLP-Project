"""
hedge_scorer.py

Rule-based hedge intensity scorer for HedgeBERT.
Loads learned cue category weights from weights.json and scores
any sentence as a float in [0.0, 1.0].

The aggregation formula is:
    score = 1 − ∏(1 − wᵢ)  for all detected cue instances i

This gives diminishing returns — each additional cue adds hedging
but the score never exceeds 1.0.

Scope simplification: cues are scored only within the clause they
appear in, not across conjunction boundaries.

Usage as a module:
    from hedge_scorer import HedgeScorer
    scorer = HedgeScorer("weights.json")
    score = scorer.score("I guess this product is sort of okay")
    print(score)  # e.g. 0.891

Usage from command line (score a single sentence):
    python hedge_scorer.py --weights weights.json
                           --sentence "I guess this is sort of decent"

Usage from command line (score a TSV file):
    python hedge_scorer.py --weights weights.json
                           --input_tsv training_data.tsv
                           --output_tsv training_data_scored.tsv
"""

import argparse
import json
import re
import pandas as pd


# ---------------------------------------------------------------------------
# Cue inventory — must match learn_weights.py exactly
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# HedgeScorer class
# ---------------------------------------------------------------------------

class HedgeScorer:
    """
    Scores a sentence for hedge intensity using learned category weights
    and a diminishing-returns aggregation formula.
    """

    def __init__(self, weights_path="weights.json"):
        with open(weights_path, "r") as f:
            self.weights = json.load(f)
        self._validate_weights()

    def _validate_weights(self):
        """Check all cue categories have a weight."""
        for cat in CUE_INVENTORY:
            if cat not in self.weights:
                raise ValueError(
                    f"Missing weight for category '{cat}' in weights file."
                )

    # ------------------------------------------------------------------
    # Scope helpers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------

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
            if cue in sent_lower and cue not in seen_cues:
                clause = self._get_clause_with_cue(sent_lower, cue)
                if cue in clause:
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Score sentences for hedge intensity."
    )
    parser.add_argument(
        "--weights",
        default="weights.json",
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
