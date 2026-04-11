import pandas as pd
from collections import Counter
from sklearn.metrics import cohen_kappa_score
import numpy as np
import random
import argparse

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Sentiment labeling pipeline")
parser.add_argument("--score-based",   action="store_true", help="Use star rating as annotator 1")
parser.add_argument("--rule-based",    action="store_true", help="Use rule-based NLP as an annotator")
parser.add_argument("--random-label",  action="store_true", help="Use random labels as an annotator")
parser.add_argument("--input",         type=str, default="all_cleaned.csv", help="Input CSV file")
parser.add_argument("--output",        type=str, default="labeled_reviews.csv", help="Output CSV file")
parser.add_argument("--size",          type=int, default=300, help="Number of records to label (default: all)")
args = parser.parse_args()

# Require at least 2 annotators for majority voting + Kappa
active_flags = [args.score_based, args.rule_based, args.random_label]
if sum(active_flags) < 2:
    parser.error("You must enable at least 2 annotators (e.g. --rule-based --random-label)")

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(args.input)

if args.size is not None:
    if args.size > len(df):
        print(f"Warning: --size {args.size} exceeds dataset length {len(df)}, using full dataset")
    else:
        df = df.sample(n=args.size, random_state=42).reset_index(drop=True)
        print(f"[+] Sampled {args.size} records from dataset")

print(f"[+] Working on {len(df)} records\n")
annotators = {}   # will hold {"annotator_name": pd.Series}

# ── Annotator: score-based ────────────────────────────────────────────────────
if args.score_based:
    if "score" not in df.columns:
        parser.error("--score-based requires a 'score' column in the data")

    def score_to_sentiment(score):
        if score <= 2:   return "Negative"
        elif score == 3: return "Neutral"
        else:            return "Positive"

    annotators["score_based"] = df["score"].apply(score_to_sentiment)
    print("[+] Annotator added: score-based")

# ── Annotator: rule-based NLP ─────────────────────────────────────────────────
if args.rule_based:
    if "content" not in df.columns:
        parser.error("--rule-based requires a 'content' column in the data")

    positive_words = {"good", "great", "excellent", "love", "amazing", "best",
                      "fast", "happy", "recommend", "perfect", "nice", "satisfied"}
    negative_words = {"worst", "problem", "bad", "terrible", "horrible", "hate",
                      "never", "poor", "broken", "failed", "wrong", "fake",
                      "counterfeit", "hang", "dont", "not"}

    def text_sentiment(text):
        words = set(str(text).lower().split())
        pos = len(words & positive_words)
        neg = len(words & negative_words)
        if neg > pos:   return "Negative"
        elif pos > neg: return "Positive"
        else:           return "Neutral"

    annotators["rule_based"] = df["content"].apply(text_sentiment)
    print("[+] Annotator added: rule-based NLP")

# ── Annotator: random labels ──────────────────────────────────────────────────
if args.random_label:
    random.seed(42)
    labels = ["Positive", "Negative", "Neutral"]
    annotators["random_label"] = pd.Series(
        [random.choice(labels) for _ in range(len(df))], index=df.index
    )
    print("[+] Annotator added: random labels")

# ── Add annotator columns to df ───────────────────────────────────────────────
annotator_names = list(annotators.keys())
for name, series in annotators.items():
    df[name] = series

# ── Majority voting ───────────────────────────────────────────────────────────
def majority_vote(row):
    votes = [row[name] for name in annotator_names]
    counts = Counter(votes)
    top = counts.most_common()
    # Full tie → trust the first annotator mentioned in the command
    if top[0][1] == 1:
        return votes[0]
    return top[0][0]

df["final_label"] = df.apply(majority_vote, axis=1)

# ── Cohen's Kappa for every pair ──────────────────────────────────────────────
print("\n" + "=" * 50)
print("Inter-Annotator Agreement (Cohen's Kappa)")
print("=" * 50)

kappas = []
for i in range(len(annotator_names)):
    for j in range(i + 1, len(annotator_names)):
        a, b = annotator_names[i], annotator_names[j]
        k = cohen_kappa_score(df[a], df[b])
        kappas.append(k)
        print(f"{a:20s} vs {b:20s}  -> kappa = {k:.3f}")

avg_kappa = np.mean(kappas)
print(f"\nAverage Kappa: {avg_kappa:.3f}")

def interpret_kappa(k):
    if k < 0.2:   return "Slight"
    elif k < 0.4: return "Fair"
    elif k < 0.6: return "Moderate"
    elif k < 0.8: return "Substantial"
    else:         return "Almost perfect"

print(f"Interpretation: {interpret_kappa(avg_kappa)}")

# ── Label distribution ────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("Final Label Distribution")
print("=" * 50)
print(df["final_label"].value_counts())
print(f"\nTotal records labeled: {len(df)}")

# ── Save ──────────────────────────────────────────────────────────────────────
df.to_csv(args.output, index=False)
print(f"\n[+] Saved to {args.output}")