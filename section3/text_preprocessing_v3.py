"""
Improved Text Preprocessing v3 - Preserves Sentiment-Critical Words
Fixes: Stopwords issue, negation handling, sentiment preservation
"""

import argparse
import pandas as pd
import re
import os
from pathlib import Path
from textblob import TextBlob
from nltk.corpus import stopwords
import nltk

# Download stopwords if not available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Standard English stopwords
STANDARD_STOP_WORDS = set(stopwords.words('english'))

# CRITICAL FIX: Remove sentiment-critical words from stopwords
# These words are essential for sentiment analysis
SENTIMENT_CRITICAL_WORDS = {
    # Negation words
    'not', 'no', 'nor', 'neither', 'never', "n't", "don't", "didn't", "doesn't",
    "doesn't", "won't", "wasn't", "aren't", "isn't", "haven't", "hasn't",
    "shouldn't", "couldn't", "mightn't", "mustn't",
    
    # Intensifiers (strengthen sentiment)
    'very', 'too', 'so', 'extremely', 'incredibly', 'absolutely', 'quite',
    'really', 'truly', 'totally', 'completely', 'utterly', 'far', 'much',
    'more', 'most', 'less', 'least',
    
    # Positive sentiment words
    'good', 'better', 'best', 'great', 'excellent', 'amazing', 'brilliant',
    'fantastic', 'wonderful', 'awesome', 'beautiful', 'pretty', 'nice',
    
    # Negative sentiment words  
    'bad', 'worse', 'worst', 'terrible', 'horrible', 'awful', 'poor',
    'wrong',' hate', 'wrong', 'awful', 'ugly',
    
    # Sentiment modifiers
    'only', 'just', 'almost', 'barely', 'hardly', 'scarcely'
}

# Refined stopwords: Remove sentiment-critical words
FILTERED_STOP_WORDS = STANDARD_STOP_WORDS - SENTIMENT_CRITICAL_WORDS

print(f"[*] Standard NLTK stopwords: {len(STANDARD_STOP_WORDS)}")
print(f"[*] Removed for sentiment: {len(SENTIMENT_CRITICAL_WORDS)}")
print(f"[*] Final stopwords list: {len(FILTERED_STOP_WORDS)}")
print(f"[!] Preserved sentiment words: {sorted(SENTIMENT_CRITICAL_WORDS)[:10]}...")

# Constants for addressing overfitting
MIN_WORD_LENGTH = 2  # Remove single characters
MAX_WORD_LENGTH = 50  # Remove overly long words (likely typos/spam)

# ---------------- CLEANING FUNCTIONS ---------------- #

def lowercase(text):
    """Convert text to lowercase"""
    return str(text).lower()

def remove_urls(text):
    """Remove URLs from text"""
    return re.sub(r'http\S+|www\S+', '', str(text))

def remove_punctuation(text):
    """Remove punctuation but preserve spaces"""
    return re.sub(r'[^\w\s]', '', str(text))

def remove_emojis(text):
    """Remove emoji characters"""
    text = str(text)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002700-\U000027BF"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

def remove_numbers(text):
    """Remove numbers to reduce overfitting"""
    return re.sub(r'\d+', '', str(text))

def lemmatize_text(text):
    """Convert words to their base form"""
    blob = TextBlob(str(text))
    return " ".join([word.lemmatize() for word in blob.words])

def fix_spelling(text):
    """Correct common spelling mistakes"""
    blob = TextBlob(str(text))
    return str(blob.correct())

def remove_stopwords_improved(text, use_filtered=True):
    """
    Remove stopwords while preserving sentiment-critical words
    
    Args:
        text: Input text
        use_filtered: If True, use sentiment-aware filtered stopwords
                    If False, use standard NLTK stopwords (not recommended)
    """
    stopword_set = FILTERED_STOP_WORDS if use_filtered else STANDARD_STOP_WORDS
    words = str(text).lower().split()
    
    # Remove stopwords and clean words
    filtered = [
        word for word in words
        if word not in stopword_set and len(word) >= MIN_WORD_LENGTH
    ]
    
    return " ".join(filtered)

def remove_extra_whitespace(text):
    """Remove multiple spaces, tabs, newlines"""
    text = str(text)
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces → single space
    return text.strip()

def extract_tags(df):
    """Ensure app column exists"""
    if "app" not in df.columns:
        df["app"] = "unknown"
    return df

def add_source(df, input_path):
    """Add source column based on filename"""
    source = Path(input_path).stem
    df["source"] = source
    return df

# Anti-overfitting: Filter out noisy/rare words
def filter_word_frequency(df, min_samples=2):
    """
    Remove words that appear too frequently (likely stopwords we missed)
    or too infrequently (likely typos/spam) - helps reduce overfitting
    """
    # Count word frequencies
    from collections import Counter
    all_words = ' '.join(df['content'].astype(str)).split()
    word_freq = Counter(all_words)
    
    # Keep words that appear in 2+ samples and less than 50% of data
    min_freq = min_samples
    max_freq = len(df) * 0.5
    
    valid_words = {w for w, count in word_freq.items() if min_freq <= count <= max_freq}
    
    print(f"[*] Total unique words: {len(word_freq)}")
    print(f"[*] Valid words (after frequency filter): {len(valid_words)}")
    print(f"[*] Removed: {len(word_freq) - len(valid_words)} words (too rare or too common)")
    
    def filter_words(text):
        words = str(text).split()
        return " ".join([w for w in words if w in valid_words])
    
    df['content'] = df['content'].apply(filter_words)
    return df

# Anti-overfitting: Add data validation
def validate_text(text):
    """Validate and clean individual records"""
    text = str(text).strip()
    # Remove texts that are too short (likely noise)
    if len(text.split()) < 2:
        return ""
    # Remove extremely long texts (likely malformed)
    if len(text.split()) > 500:
        return " ".join(text.split()[:500])  # Truncate
    return text

# Anti-overfitting: Cross-dataset consistency check
def check_consistency(df1, df2, label="Training vs Validation"):
    """Check if distributions are similar (helps detect overfitting)"""
    print(f"\n[*] Consistency Check: {label}")
    print(f"    Dataset 1 size: {len(df1)}")
    print(f"    Dataset 2 size: {len(df2)}")
    
    # Compare label distributions
    if 'final_label' in df1.columns and 'final_label' in df2.columns:
        print(f"    Dataset 1 labels: {dict(df1['final_label'].value_counts())}")
        print(f"    Dataset 2 labels: {dict(df2['final_label'].value_counts())}")

# Example test cases for sentiment validation
TEST_CASES = [
    ("not good", "Negative"),      # Negation of positive
    ("not bad", "Positive"),       # Negation of negative  
    ("very good", "Positive"),     # Intensified positive
    ("too bad", "Negative"),       # Intensified negative
    ("extremely bad", "Negative"), # Strong negative
    ("not very good", "Negative"), # Negation + intensifier
    ("just okay", "Neutral"),      # Mild positive
]

def validate_test_cases(model, vectorizer, label_encoder):
    """Test model on known sentiment cases to catch issues"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    print("\n" + "="*80)
    print("SENTIMENT TEST VALIDATION")
    print("="*80)
    
    for text, expected_label in TEST_CASES:
        try:
            # Vectorize
            text_vec = vectorizer.transform([text])
            # Predict
            pred_encoded = model.predict(text_vec)[0]
            pred_label = label_encoder.inverse_transform([pred_encoded])[0]
            
            status = "✓" if pred_label == expected_label else "✗"
            print(f"{status} '{text}'")
            print(f"    Expected: {expected_label:10s} | Predicted: {pred_label:10s}")
        except Exception as e:
            print(f"✗ '{text}' - Error: {str(e)}")

# Main preprocessing pipeline
def preprocess(df, args):
    """Main preprocessing pipeline"""
    
    if args.remove_emojis:
        df["content"] = df["content"].apply(remove_emojis)
    
    if args.lowercase:
        df["content"] = df["content"].apply(lowercase)
    
    if args.remove_urls:
        df["content"] = df["content"].apply(remove_urls)
    
    if args.remove_numbers:
        df["content"] = df["content"].apply(remove_numbers)
    
    if args.remove_punctuation:
        df["content"] = df["content"].apply(remove_punctuation)
    
    # IMPROVED: Use sentiment-aware stopwords removal
    if args.remove_stopwords:
        df["content"] = df["content"].apply(
            lambda x: remove_stopwords_improved(x, use_filtered=True)
        )
    
    if args.lemmatize:
        df["content"] = df["content"].apply(lemmatize_text)
    
    if args.fix_spelling:
        df["content"] = df["content"].apply(fix_spelling)
    
    # Clean up whitespace
    df["content"] = df["content"].apply(remove_extra_whitespace)
    
    # Validate texts
    df["content"] = df["content"].apply(validate_text)
    
    if args.extract_tags:
        df = extract_tags(df)
    
    return df

# Collect files helper
def collect_files(inputs):
    """Collect CSV files from paths and folders"""
    files = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            found = list(p.glob("*.csv"))
            print(f"  Folder '{p}' → found {len(found)} CSV files")
            files.extend(found)
        elif p.is_file() and p.suffix == ".csv":
            files.append(p)
        else:
            print(f"  Warning: '{item}' is not a CSV file or folder — skipping")
    return files

# Main function
def main():
    parser = argparse.ArgumentParser(
        description="Text Preprocessing v3 - Fixes stopwords issue, preserves sentiment"
    )
    
    parser.add_argument("--input", required=True, nargs="+",
                        help="One or more CSV files and/or folders")
    parser.add_argument("--output", required=True,
                        help="Output file path")
    
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--remove_urls", action="store_true")
    parser.add_argument("--remove_emojis", action="store_true")
    parser.add_argument("--remove_numbers", action="store_true", 
                        help="[NEW] Remove numbers to reduce overfitting")
    parser.add_argument("--remove_punctuation", action="store_true")
    parser.add_argument("--remove_stopwords", action="store_true",
                        help="[IMPROVED] Use sentiment-aware stopword removal")
    parser.add_argument("--lemmatize", action="store_true")
    parser.add_argument("--fix_spelling", action="store_true")
    parser.add_argument("--extract_tags", action="store_true")
    
    args = parser.parse_args()
    
    # Collect files
    all_files = collect_files(args.input)
    if not all_files:
        print("No CSV files found.")
        return
    
    print(f"\nProcessing {len(all_files)} files...")
    
    # Process each file
    dfs = []
    for file in all_files:
        print(f"  {file.name}")
        df = pd.read_csv(file)
        df = add_source(df, str(file))
        df = preprocess(df, args)
        dfs.append(df)
    
    # Combine and save
    final_df = pd.concat(dfs, ignore_index=True)
    
    # Remove duplicate or near-duplicate rows (anti-overfitting)
    initial_count = len(final_df)
    final_df = final_df.drop_duplicates(subset=['content'], keep='first')
    dropped = initial_count - len(final_df)
    if dropped > 0:
        print(f"[*] Removed {dropped} duplicate texts (anti-overfitting)")
    
    os.makedirs(Path(args.output).parent, exist_ok=True)
    final_df.to_csv(args.output, index=False)
    
    print(f"\n✓ Saved to: {args.output}")
    print(f"  Total records: {len(final_df)}")
    print(f"  Unique texts: {final_df['content'].nunique()}")

if __name__ == "__main__":
    main()
