import argparse
import pandas as pd
import numpy as np
import json
from pathlib import Path
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Download required NLTK data
import nltk
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

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
    
    # Calculate metrics
    accuracy = accuracy_score(df["final_label"], results["vader_prediction"])
    precision = precision_score(df["final_label"], results["vader_prediction"], average='weighted', zero_division=0)
    recall = recall_score(df["final_label"], results["vader_prediction"], average='weighted', zero_division=0)
    f1 = f1_score(df["final_label"], results["vader_prediction"], average='weighted', zero_division=0)
    
    reports["vader"] = {
        "model": "VADER",
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": confusion_matrix(df["final_label"], results["vader_prediction"]).tolist()
    }
    
    print(f"✓ Accuracy:  {accuracy:.3f}")
    print(f"✓ Precision: {precision:.3f}")
    print(f"✓ Recall:    {recall:.3f}")
    print(f"✓ F1-Score:  {f1:.3f}\n")

# ──────────────────────────────────────────────────────────────────────────────
# ── AFINN DICTIONARY-BASED MODEL ──────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

if args.afinn:
    print("=" * 60)
    print("AFINN Dictionary-Based Model")
    print("=" * 60)
    
    # AFINN lexicon (subset - can be extended)
    afinn_dict = {
        # Positive words
        'good': 3, 'great': 3, 'excellent': 4, 'amazing': 4, 'awesome': 4,
        'love': 3, 'best': 3, 'perfect': 3, 'wonderful': 4, 'fantastic': 4,
        'happy': 2, 'glad': 2, 'beautiful': 3, 'brilliant': 4, 'terrific': 3,
        'superb': 4, 'outstanding': 4, 'recommend': 2, 'satisfied': 2, 'nice': 2,
        'fast': 2, 'quick': 2, 'easy': 2, 'simple': 1, 'clean': 2, 'smooth': 2,
        'helpful': 2, 'useful': 2, 'reliable': 2, 'trustworthy': 2,
        
        # Negative words
        'bad': -3, 'terrible': -4, 'awful': -4, 'horrible': -4, 'worst': -4,
        'hate': -3, 'poor': -3, 'broken': -2, 'failed': -3, 'problem': -2,
        'issue': -1, 'wrong': -2, 'fake': -3, 'counterfeit': -4, 'slow': -2,
        'hang': -2, 'crash': -3, 'don\'t': -1, 'not': -1, 'never': -2,
        'useless': -3, 'garbage': -4, 'waste': -2, 'scam': -4, 'fraud': -4,
        'disappointing': -2, 'unsatisfied': -2, 'unacceptable': -3,
    } # affin dictionary wasn't used
    
    def negation_words():
        return {'not', 'no', 'never', 'neither', 'nobody', 'nothing', 'neither', "don't", "doesn't", "didn't", "hasn't", "haven't"}
    
    def afinn_sentiment(text, use_negation=args.afinn_negation):
        words = str(text).lower().split()
        score = 0
        
        for i, word in enumerate(words):
            # Remove punctuation from word for matching
            word_cleaned = ''.join(c for c in word if c.isalnum())
            
            # Check for negation in previous word
            is_negated = False
            if use_negation and i > 0:
                prev_word = ''.join(c for c in words[i-1] if c.isalnum())
                if prev_word in negation_words():
                    is_negated = True
            
            # Apply AFINN score
            if word_cleaned in afinn_dict:
                word_score = afinn_dict[word_cleaned]
                if is_negated:
                    word_score *= -1  # Flip sentiment on negation
                score += word_score
        
        return score
    
    # Get scores and convert to labels
    afinn_scores = df["content"].apply(lambda x: afinn_sentiment(x))
    results["afinn_score"] = afinn_scores
    
    def score_to_label(score):
        if score > 0:
            return "Positive"
        elif score < 0:
            return "Negative"
        else:
            return "Neutral"
    
    results["afinn_prediction"] = afinn_scores.apply(score_to_label)
    
    # Calculate metrics
    accuracy = accuracy_score(df["final_label"], results["afinn_prediction"])
    precision = precision_score(df["final_label"], results["afinn_prediction"], average='weighted', zero_division=0)
    recall = recall_score(df["final_label"], results["afinn_prediction"], average='weighted', zero_division=0)
    f1 = f1_score(df["final_label"], results["afinn_prediction"], average='weighted', zero_division=0)
    
    negation_status = "enabled" if use_negation else "disabled"
    
    reports["afinn"] = {
        "model": "AFINN",
        "negation_handling": negation_status,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": confusion_matrix(df["final_label"], results["afinn_prediction"]).tolist(),
        "dictionary_size": len(afinn_dict)
    }
    
    print(f"✓ Dictionary size: {len(afinn_dict)} words")
    print(f"✓ Negation handling: {negation_status}")
    print(f"✓ Accuracy:  {accuracy:.3f}")
    print(f"✓ Precision: {precision:.3f}")
    print(f"✓ Recall:    {recall:.3f}")
    print(f"✓ F1-Score:  {f1:.3f}\n")

# ──────────────────────────────────────────────────────────────────────────────
# ── SAVE RESULTS ──────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

results.to_csv(args.output, index=False)
print("=" * 60)
print(f"✓ Results saved → {args.output}\n")

# Save report as JSON
with open(args.output_report, "w") as f:
    json.dump(reports, f, indent=2)

print(f"✓ Report saved → {args.output_report}")
print("=" * 60)
