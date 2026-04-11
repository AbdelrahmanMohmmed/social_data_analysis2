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

# Get English stopwords
STOP_WORDS = set(stopwords.words('english'))

# ---------------- CLEANING FUNCTIONS ---------------- #
def lowercase(text):
    return str(text).lower()

def remove_urls(text):
    return re.sub(r'http\S+|www\S+', '', str(text))

def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', str(text))

def remove_emojis(text):
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

def lemmatize_text(text):
    blob = TextBlob(str(text))
    return " ".join([word.lemmatize() for word in blob.words])

def fix_spelling(text):
    blob = TextBlob(str(text))
    return str(blob.correct())

def remove_stopwords(text):
    words = str(text).lower().split()
    filtered = [word for word in words if word not in STOP_WORDS]
    return " ".join(filtered)

def extract_tags(df):
    if "app" not in df.columns:
        df["app"] = "unknown"
    return df

def add_source(df, input_path):
    source = Path(input_path).stem          # e.g. amazon_reviews
    df["source"] = source
    return df

# ---------------- PIPELINE ---------------- #
def preprocess(df, args):
    if args.lowercase:
        df["content"] = df["content"].apply(lowercase)
    if args.remove_urls:
        df["content"] = df["content"].apply(remove_urls)
    if args.remove_emojis:
        df["content"] = df["content"].apply(remove_emojis)
    if args.remove_punctuation:
        df["content"] = df["content"].apply(remove_punctuation)
    if args.remove_stopwords:
        df["content"] = df["content"].apply(remove_stopwords)
    if args.lemmatize:
        df["content"] = df["content"].apply(lemmatize_text)
    if args.fix_spelling:
        df["content"] = df["content"].apply(fix_spelling)
    if args.extract_tags:
        df = extract_tags(df)
    return df

# ---------------- COLLECT FILES HELPER ---------------- #
def collect_files(inputs):
    """
    inputs: list of paths — each can be a .csv file or a folder.
    Returns a flat list of Path objects pointing to CSV files.
    """
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

# ---------------- MAIN ---------------- #
def main():
    parser = argparse.ArgumentParser()

    # --input accepts 1 file, multiple files, or folders — all mixed together
    parser.add_argument("--input",  required=True, nargs="+",
                        help="One or more CSV files and/or folders")
    parser.add_argument("--output", required=True,
                        help="Output file path, e.g. section3/all_cleaned.csv")

    parser.add_argument("--lowercase",          action="store_true")
    parser.add_argument("--remove_urls",         action="store_true")
    parser.add_argument("--remove_emojis",       action="store_true")
    parser.add_argument("--remove_punctuation",  action="store_true")
    parser.add_argument("--remove_stopwords",    action="store_true")
    parser.add_argument("--lemmatize",           action="store_true")
    parser.add_argument("--fix_spelling",        action="store_true")
    parser.add_argument("--extract_tags",        action="store_true")

    args = parser.parse_args()

    # Collect all CSV files from whatever was passed
    all_files = collect_files(args.input)

    if not all_files:
        print("No CSV files found. Check your --input paths.")
        return

    print(f"\nTotal files to process: {len(all_files)}")

    # Process each file and collect into list
    dfs = []
    for file in all_files:
        print(f"  Processing: {file.name}")
        df = pd.read_csv(file)
        df = add_source(df, str(file))
        df = preprocess(df, args)
        dfs.append(df)

    # Concatenate everything into one dataframe
    final_df = pd.concat(dfs, ignore_index=True)

    # Save single output file
    os.makedirs(Path(args.output).parent, exist_ok=True)
    final_df.to_csv(args.output, index=False)

    print(f"\nDone! {len(final_df)} total rows saved -> {args.output}")

if __name__ == "__main__":
    main()