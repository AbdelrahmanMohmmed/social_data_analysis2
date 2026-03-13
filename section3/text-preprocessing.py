import argparse
import os
import pandas as pd
import re
from textblob import TextBlob

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
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002700-\U000027BF"  # dingbats
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

def extract_tags(df):
    if "app" not in df.columns:
        df["app"] = "unknown"
    return df

def add_source(df, input_path):
    filename = os.path.basename(input_path)         # e.g. amazon_reviews.csv
    source = os.path.splitext(filename)[0]          # e.g. amazon_reviews
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

    if args.lemmatize:
        df["content"] = df["content"].apply(lemmatize_text)

    if args.fix_spelling:
        df["content"] = df["content"].apply(fix_spelling)

    if args.extract_tags:
        df = extract_tags(df)

    return df

# ---------------- MAIN ---------------- #

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument("--lowercase", action="store_true", default=False)
    parser.add_argument("--remove_urls", action="store_true", default=False)
    parser.add_argument("--remove_emojis", action="store_true", default=False)
    parser.add_argument("--remove_punctuation", action="store_true", default=False)
    parser.add_argument("--lemmatize", action="store_true", default=False)
    parser.add_argument("--fix_spelling", action="store_true", default=False)
    parser.add_argument("--extract_tags", action="store_true", default=False)

    args = parser.parse_args()

    df = pd.read_csv(args.input)

    df = preprocess(df, args)

    df.to_csv(args.output, index=False)

    print("Preprocessing complete!")

if __name__ == "__main__":
    main()