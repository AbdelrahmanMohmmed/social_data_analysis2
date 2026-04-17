import argparse
import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim.downloader as api

# ──────────────────────────────────────────────────────────────────────────────
# ── ARGUMENT PARSING ──────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Text Representation: TF-IDF and GloVe embeddings")
parser.add_argument("--input",            type=str, default="labeled_reviews.csv", help="Input CSV file or directory containing CSV files")
parser.add_argument("--output-dir",       type=str, default=".", help="Output directory for representations")
parser.add_argument("--tfidf",            action="store_true", help="Generate TF-IDF vectors")
parser.add_argument("--glove",            action="store_true", help="Generate GloVe embeddings")
parser.add_argument("--max-features",     type=int, default=5000, help="Max features for TF-IDF (default: 5000)")
parser.add_argument("--ngram-range",      type=int, nargs=2, default=[1, 2], help="N-gram range for TF-IDF (default: 1 2)")
parser.add_argument("--glove-dim",        type=int, default=100, help="GloVe embedding dimension (default: 100)")
parser.add_argument("--min-df",           type=int, default=2, help="Min document frequency for TF-IDF (default: 2)")
parser.add_argument("--max-df",           type=float, default=0.95, help="Max document frequency for TF-IDF (default: 0.95)")

args = parser.parse_args()

# Require at least 1 method
if not args.tfidf and not args.glove:
    parser.error("You must enable at least 1 representation method (--tfidf and/or --glove)")

if len(args.ngram_range) != 2:
    parser.error("--ngram-range must contain exactly 2 integers, e.g. --ngram-range 1 2")

if args.min_df < 1:
    parser.error("--min-df must be >= 1")

if not (0 < args.max_df <= 1):
    parser.error("--max-df must be in (0, 1]")

# Create output directory
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

input_path = Path(args.input)
if input_path.is_dir():
    input_files = sorted(input_path.glob("*.csv"))
elif input_path.is_file() and input_path.suffix.lower() == ".csv":
    input_files = [input_path]
else:
    raise ValueError("--input must be a CSV file or a directory containing CSV files")

if not input_files:
    raise ValueError(f"No CSV files found in: {input_path}")

print(f"[+] Files to process: {len(input_files)}")

glove_model = None
chosen_dim = None

if args.glove:
    # Map dimension to pretrained model
    dim_model_map = {
        50: "glove-wiki-gigaword-50",
        100: "glove-wiki-gigaword-100",
        200: "glove-wiki-gigaword-200",
        300: "glove-wiki-gigaword-300",
    }

    if args.glove_dim not in dim_model_map:
        print(f"Note: GloVe {args.glove_dim}d not available. Using 100d instead.")
        chosen_dim = 100
    else:
        chosen_dim = args.glove_dim

    model_name = dim_model_map[chosen_dim]
    print(f"Downloading GloVe model ({chosen_dim}d)... this may take a moment")
    glove_model = api.load(model_name)
    print(f"[+] Loaded GloVe model: {model_name}\n")


def get_embedding(text, model, dim):
    words = str(text).lower().split()
    embeddings = [model[word] for word in words if word in model.key_to_index]
    if embeddings:
        return np.mean(embeddings, axis=0)
    return np.zeros(dim)

# ──────────────────────────────────────────────────────────────────────────────
# ── REPRESENTATION BUILD LOOP ─────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
for file_path in input_files:
    print("=" * 60)
    print(f"Processing: {file_path.name}")
    print("=" * 60)

    df = pd.read_csv(file_path)
    if "content" not in df.columns:
        raise ValueError(f"Input CSV must contain a 'content' column: {file_path}")

    texts = df["content"].fillna("").astype(str)
    file_stem = file_path.stem

    if args.tfidf:
        print("TF-IDF Vectorization")
        tfidf = TfidfVectorizer(
            max_features=args.max_features,
            ngram_range=tuple(args.ngram_range),
            min_df=args.min_df,
            max_df=args.max_df,
            lowercase=True,
            stop_words="english",
        )

        tfidf_matrix = tfidf.fit_transform(texts)
        tfidf_dense = tfidf_matrix.toarray()

        tfidf_output = output_dir / f"{file_stem}_tfidf_matrix.csv"
        feature_names_output = output_dir / f"{file_stem}_tfidf_features.json"
        vectorizer_output = output_dir / f"{file_stem}_tfidf_vectorizer.pkl"

        tfidf_df = pd.DataFrame(
            tfidf_dense,
            columns=[f"tfidf_{i}" for i in range(tfidf_dense.shape[1])],
        )
        tfidf_df.to_csv(tfidf_output, index=False)

        feature_info = {
            "feature_names": tfidf.get_feature_names_out().tolist(),
            "n_features": len(tfidf.get_feature_names_out()),
        }
        with open(feature_names_output, "w") as f:
            json.dump(feature_info, f, indent=2)

        with open(vectorizer_output, "wb") as f:
            pickle.dump(tfidf, f)

        print(f"[+] TF-IDF matrix shape: {tfidf_dense.shape}")
        print(f"[+] Saved TF-IDF matrix -> {tfidf_output}")
        print(f"[+] Saved TF-IDF features -> {feature_names_output}")
        print(f"[+] Saved TF-IDF vectorizer -> {vectorizer_output}")

    if args.glove:
        print("GloVe Embeddings")
        glove_embeddings = np.array([
            get_embedding(text, glove_model, chosen_dim)
            for text in texts
        ])

        glove_output = output_dir / f"{file_stem}_glove_embeddings.csv"
        glove_df = pd.DataFrame(
            glove_embeddings,
            columns=[f"glove_{i}" for i in range(glove_embeddings.shape[1])],
        )
        glove_df.to_csv(glove_output, index=False)

        print(f"[+] GloVe embeddings shape: {glove_embeddings.shape}")
        print(f"[+] Saved GloVe embeddings -> {glove_output}")

    print()

print("=" * 60)
print(f"[+] Completed {len(input_files)} file(s)")
print(f"[+] Outputs saved in -> {output_dir}")
print("=" * 60)
