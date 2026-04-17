import argparse
import pandas as pd
import json
from pathlib import Path
import re
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Download required NLTK data
import nltk
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# ──────────────────────────────────────────────────────────────────────────────
# ── ARGUMENT PARSING ──────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Lexical-Based Sentiment Models: VADER and AFINN")
parser.add_argument("--input",           type=str, default="labeled_reviews.csv", help="Input CSV file with labels")
parser.add_argument("--output",          type=str, default="lexical_results.csv", help="Output CSV file")
parser.add_argument("--output-report",   type=str, default="lexical_report.json", help="Output report file")
parser.add_argument("--vader",           action="store_true", help="Use VADER sentiment analyzer")
parser.add_argument("--afinn",           action="store_true", help="Use AFINN dictionary-based model")
parser.add_argument("--afinn-negation",  action="store_true", help="Enable negation handling for AFINN")

args = parser.parse_args()

# Require at least 1 model
if not args.vader and not args.afinn:
    parser.error("You must enable at least 1 model (--vader and/or --afinn)")


CLASS_LABELS = ["Negative", "Neutral", "Positive"]
NEGATION_WORDS = {
    "not", "no", "never", "neither", "nobody", "nothing", "none",
    "dont", "don't", "doesnt", "doesn't", "didnt", "didn't",
    "hasnt", "hasn't", "havent", "haven't", "cannot", "cant", "can't"
}


def normalize_label(label):
    text = str(label).strip().lower()
    if "negative" in text:
        return "Negative"
    if "positive" in text:
        return "Positive"
    return "Neutral"


def evaluate_predictions(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "labels": CLASS_LABELS,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASS_LABELS).tolist(),
    }


def load_afinn_lexicon():
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "AFINN" / "AFINN-111.txt",
        script_dir / "AFINN" / "AFINN-96.txt",
    ]

    afinn_path = next((p for p in candidates if p.exists()), None)
    if afinn_path is None:
        raise FileNotFoundError(
            "AFINN lexicon not found. Expected one of: "
            f"{candidates[0]} or {candidates[1]}"
        )

    lexicon = {}
    with afinn_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            word, score = line.split("\t")
            lexicon[word.lower()] = int(score)

    return lexicon, afinn_path


def tokenize_for_afinn(text):
    # Keep words and sentence-ending punctuation that resets negation scope.
    return re.findall(r"[a-zA-Z']+|[.!?]", str(text).lower())


def afinn_sentiment_score(text, afinn_dict, use_negation=False, negation_window=3):
    tokens = tokenize_for_afinn(text)
    total_score = 0
    negation_scope = 0

    for token in tokens:
        if token in {".", "!", "?"}:
            negation_scope = 0
            continue

        cleaned = token.strip("'")
        if use_negation and cleaned in NEGATION_WORDS:
            negation_scope = negation_window
            continue

        if cleaned in afinn_dict:
            word_score = afinn_dict[cleaned]
            if use_negation and negation_scope > 0:
                word_score *= -1
            total_score += word_score

        if use_negation and negation_scope > 0:
            negation_scope -= 1

    return total_score


def score_to_label(score):
    if score > 0:
        return "Positive"
    if score < 0:
        return "Negative"
    return "Neutral"

# ──────────────────────────────────────────────────────────────────────────────
# ── LOAD DATA ─────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

df = pd.read_csv(args.input)

if "content" not in df.columns:
    raise ValueError("Input CSV must contain a 'content' column")
if "final_label" not in df.columns:
    raise ValueError("Input CSV must contain a 'final_label' column (ground truth)")

print(f"✓ Loaded {len(df)} records from {args.input}")
print(f"Label distribution:\n{df['final_label'].value_counts()}\n")

results = df.copy()
reports = {}
y_true = df["final_label"].apply(normalize_label)
results["final_label_3class"] = y_true

# ──────────────────────────────────────────────────────────────────────────────
# ── VADER SENTIMENT ANALYZER ──────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

if args.vader:
    print("=" * 60)
    print("VADER Sentiment Analyzer")
    print("=" * 60)
    
    sia = SentimentIntensityAnalyzer()
    
    def vader_sentiment(text):
        scores = sia.polarity_scores(str(text))
        compound = scores['compound']
        
        if compound >= 0.05:
            return "Positive"
        elif compound <= -0.05:
            return "Negative"
        else:
            return "Neutral"
    
    results["vader_prediction"] = df["content"].apply(vader_sentiment)

    metrics = evaluate_predictions(y_true, results["vader_prediction"])
    reports["vader"] = {"model": "VADER", **metrics}
    
    print(f"✓ Accuracy:  {metrics['accuracy']:.3f}")
    print(f"✓ Precision: {metrics['precision']:.3f}")
    print(f"✓ Recall:    {metrics['recall']:.3f}")
    print(f"✓ F1-Score:  {metrics['f1_score']:.3f}\n")

# ──────────────────────────────────────────────────────────────────────────────
# ── AFINN DICTIONARY-BASED MODEL ──────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

if args.afinn:
    print("=" * 60)
    print("AFINN Dictionary-Based Model")
    print("=" * 60)
    
    afinn_dict, afinn_path = load_afinn_lexicon()

    afinn_scores = df["content"].apply(
        lambda x: afinn_sentiment_score(x, afinn_dict, use_negation=args.afinn_negation)
    )
    results["afinn_score"] = afinn_scores

    results["afinn_prediction"] = afinn_scores.apply(score_to_label)

    metrics = evaluate_predictions(y_true, results["afinn_prediction"])
    negation_status = "enabled" if args.afinn_negation else "disabled"

    reports["afinn"] = {
        "model": "AFINN",
        "afinn_lexicon_file": str(afinn_path),
        "negation_handling": negation_status,
        "dictionary_size": len(afinn_dict),
        **metrics,
    }
    
    print(f"✓ Dictionary size: {len(afinn_dict)} words")
    print(f"✓ Lexicon file: {afinn_path}")
    print(f"✓ Negation handling: {negation_status}")
    print(f"✓ Accuracy:  {metrics['accuracy']:.3f}")
    print(f"✓ Precision: {metrics['precision']:.3f}")
    print(f"✓ Recall:    {metrics['recall']:.3f}")
    print(f"✓ F1-Score:  {metrics['f1_score']:.3f}\n")

# ──────────────────────────────────────────────────────────────────────────────
# ── SAVE RESULTS ──────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

output_path = Path(args.output)
report_path = Path(args.output_report)
output_path.parent.mkdir(parents=True, exist_ok=True)
report_path.parent.mkdir(parents=True, exist_ok=True)

results.to_csv(output_path, index=False)
print("=" * 60)
print(f"✓ Results saved → {output_path}\n")

# Save report as JSON
with open(report_path, "w") as f:
    json.dump(reports, f, indent=2)

print(f"✓ Report saved → {report_path}")
print("=" * 60)
