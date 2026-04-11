import argparse
import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim.downloader as api
from gensim.models import KeyedVectors

# ──────────────────────────────────────────────────────────────────────────────
# ── ARGUMENT PARSING ──────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Text Representation: TF-IDF and GloVe embeddings")
parser.add_argument("--input",            type=str, default="labeled_reviews.csv", help="Input CSV file")
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

# ──────────────────────────────────────────────────────────────────────────────
# ── LOAD DATA ─────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

df = pd.read_csv(args.input)

if "content" not in df.columns:
    raise ValueError("Input CSV must contain a 'content' column")

print(f"[+] Loaded {len(df)} records from {args.input}")
print(f"[+] Working on column 'content'\n")

# Create output directory
Path(args.output_dir).mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# ── TF-IDF VECTORIZATION ──────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

if args.tfidf:
    print("=" * 60)
    print("TF-IDF Vectorization")
    print("=" * 60)
    
    tfidf = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=tuple(args.ngram_range),
        min_df=args.min_df,
        max_df=args.max_df,
        lowercase=True,
        stop_words='english'
    )
    
    tfidf_matrix = tfidf.fit_transform(df["content"].astype(str))
    
    # Convert to dense for easier handling (optional, can keep sparse)
    tfidf_dense = tfidf_matrix.toarray()
    
    # Save TF-IDF matrix and feature names
    tfidf_output = Path(args.output_dir) / "tfidf_matrix.csv"
    feature_names_output = Path(args.output_dir) / "tfidf_features.json"
    
    # Save dense matrix as CSV
    tfidf_df = pd.DataFrame(
        tfidf_dense,
        columns=[f"tfidf_{i}" for i in range(tfidf_dense.shape[1])]
    )
    tfidf_df.to_csv(tfidf_output, index=False)
    
    # Save feature names and vocabulary
    feature_info = {
        "feature_names": tfidf.get_feature_names_out().tolist(),
        "n_features": len(tfidf.get_feature_names_out()),
        "vocabulary": {k: int(v) for k, v in tfidf.vocabulary_.items()}
    }
    with open(feature_names_output, "w") as f:
        json.dump(feature_info, f, indent=2)
    
    print(f"[+] TF-IDF matrix shape: {tfidf_dense.shape}")
    print(f"[+] Max features: {args.max_features}")
    print(f"[+] N-gram range: {tuple(args.ngram_range)}")
    print(f"[+] Saved TF-IDF matrix -> {tfidf_output}")
    print(f"[+] Saved feature names -> {feature_names_output}\n")
    
    # Also save the vectorizer itself for later use
    vectorizer_output = Path(args.output_dir) / "tfidf_vectorizer.pkl"
    with open(vectorizer_output, "wb") as f:
        pickle.dump(tfidf, f)
    print(f"[+] Saved TF-IDF vectorizer -> {vectorizer_output}\n")

# ──────────────────────────────────────────────────────────────────────────────
# ── GLOVE EMBEDDINGS ──────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

if args.glove:
    print("=" * 60)
    print("GloVe Embeddings")
    print("=" * 60)
    
    # Map dimension to pretrained model
    dim_model_map = {100: "glove-wiki-gigaword-100", 300: "glove-wiki-gigaword-300"}
    
    if args.glove_dim not in dim_model_map:
        print(f"Note: GloVe {args.glove_dim}d not available. Using 100d instead.")
        chosen_dim = 100
    else:
        chosen_dim = args.glove_dim
    
    model_name = dim_model_map[chosen_dim]
    
    print(f"Downloading GloVe model ({chosen_dim}d)... this may take a moment")
    glove_model = api.load(model_name)
    print(f"[+] Loaded GloVe model: {model_name}\n")
    
    # Create embedding matrix: average embedding for each review
    def get_embedding(text, model, dim):
        words = str(text).lower().split()
        embeddings = []
        for word in words:
            if word in model:
                embeddings.append(model[word])
        
        if embeddings:
            return np.mean(embeddings, axis=0)
        else:
            return np.zeros(dim)
    
    glove_embeddings = np.array([
        get_embedding(text, glove_model, chosen_dim)
        for text in df["content"].astype(str)
    ])
    
    # Save GloVe embeddings
    glove_output = Path(args.output_dir) / "glove_embeddings.csv"
    glove_df = pd.DataFrame(
        glove_embeddings,
        columns=[f"glove_{i}" for i in range(glove_embeddings.shape[1])]
    )
    glove_df.to_csv(glove_output, index=False)
    
    print(f"[+] GloVe embeddings shape: {glove_embeddings.shape}")
    print(f"[+] Embedding dimension: {chosen_dim}")
    print(f"[+] Method: Average pooling of word embeddings")
    print(f"[+] Saved GloVe embeddings -> {glove_output}\n")

# ──────────────────────────────────────────────────────────────────────────────
# ── COMBINE AND SAVE ──────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

combined_dfs = [df.copy()]

if args.tfidf:
    combined_dfs.append(tfidf_df)

if args.glove:
    combined_dfs.append(glove_df)

combined = pd.concat(combined_dfs, axis=1)

combined_output = Path(args.output_dir) / "representations_combined.csv"
combined.to_csv(combined_output, index=False)

print("=" * 60)
print(f"[+] Combined representation saved -> {combined_output}")
print(f"[+] Total shape: {combined.shape}")
print("=" * 60)
