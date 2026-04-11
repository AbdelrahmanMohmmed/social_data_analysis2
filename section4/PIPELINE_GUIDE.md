# Sentiment Analysis Pipeline - Usage Guide

This directory contains scripts for a complete sentiment analysis project covering labeling, text representation, and modeling.

## Scripts Overview

### 1. label_data.py
**Purpose**: Ground truth labeling with multiple annotators and inter-annotator agreement measurement

**Usage**:
```bash
python label_data.py \
    --input all_cleaned.csv \
    --output labeled_reviews.csv \
    --size 300 \
    --score-based \
    --rule-based \
    --random-label
```

**Arguments**:
- `--input`: Input CSV file (default: all_cleaned.csv)
- `--output`: Output CSV file (default: labeled_reviews.csv)
- `--size`: Number of records to label (default: all)
- `--score-based`: Use star rating column as annotator
- `--rule-based`: Use rule-based NLP lexicon as annotator
- `--random-label`: Use random labels as annotator
- Requires at least 2 annotators

**Output**: CSV with original data + columns for each annotator + `final_label` (majority voted)

---

### 2. text_preprocessing_v2.py
**Purpose**: Text preprocessing with multiple cleaning options

**Usage**:
```bash
python text_preprocessing_v2.py \
    --input section2/*.csv \
    --output section3/all_cleaned.csv \
    --lowercase \
    --remove_urls \
    --remove_emojis \
    --remove_punctuation \
    --lemmatize
```

**Arguments**:
- `--input`: One or more CSV files and/or folders (required)
- `--output`: Output file path (required)
- `--lowercase`: Convert to lowercase
- `--remove_urls`: Remove URLs
- `--remove_emojis`: Remove emoji characters
- `--remove_punctuation`: Remove punctuation
- `--lemmatize`: Apply lemmatization
- `--fix_spelling`: Correct spelling
- `--extract_tags`: Extract app tags

**Output**: Single CSV file with combined and preprocessed data

---

### 3. text_representation.py
**Purpose**: Convert text to numerical representations using TF-IDF and/or GloVe embeddings

**Usage**:
```bash
python text_representation.py \
    --input labeled_reviews.csv \
    --output-dir representations \
    --tfidf \
    --glove \
    --max-features 5000 \
    --glove-dim 100
```

**Arguments**:
- `--input`: Input CSV file (default: labeled_reviews.csv)
- `--output-dir`: Output directory path (default: .)
- `--tfidf`: Generate TF-IDF vectors
- `--glove`: Generate GloVe embeddings
- `--max-features`: Max features for TF-IDF (default: 5000)
- `--ngram-range`: N-gram range for TF-IDF (default: 1 2)
- `--glove-dim`: GloVe embedding dimension: 100 or 300 (default: 100)
- `--min-df`: Min document frequency (default: 2)
- `--max-df`: Max document frequency (default: 0.95)
- Requires at least 1 representation method

**Output**: 
- `tfidf_matrix.csv` - TF-IDF vectors
- `glove_embeddings.csv` - GloVe embeddings
- `representations_combined.csv` - All original columns + vectors (for ML models)
- `tfidf_vectorizer.pkl` - Serialized TF-IDF vectorizer
- `tfidf_features.json` - Feature names and vocabulary

---

### 4. lexical_based_models.py
**Purpose**: Sentiment analysis using lexicon-based methods (VADER and AFINN)

**Usage**:
```bash
python lexical_based_models.py \
    --input labeled_reviews.csv \
    --output lexical_results.csv \
    --output-report lexical_report.json \
    --vader \
    --afinn \
    --afinn-negation
```

**Arguments**:
- `--input`: Input CSV with labels (default: labeled_reviews.csv)
- `--output`: Output CSV file (default: lexical_results.csv)
- `--output-report`: Output report JSON (default: lexical_report.json)
- `--vader`: Use VADER sentiment analyzer
- `--afinn`: Use AFINN dictionary-based model
- `--afinn-negation`: Enable negation handling for AFINN (recommend: yes)
- Requires at least 1 model

**Output**:
- CSV with predictions from each model
- JSON report with accuracy metrics and confusion matrices

**Models**:
- **VADER**: Pre-trained lexicon + rules for social media text
- **AFINN**: Dictionary-based with optional negation handling (flips sentiment on "not", "never", etc.)

---

### 5. ml_based_models.py
**Purpose**: Train machine learning models using text representations

**Usage**:
```bash
python ml_based_models.py \
    --features representations/representations_combined.csv \
    --labels labeled_reviews.csv \
    --output-dir ml_models \
    --feature-type tfidf \
    --svm \
    --logistic \
    --svm-kernel rbf \
    --svm-c 1.0 \
    --test-size 0.2
```

**Arguments**:
- `--features`: CSV with text representations (required)
- `--labels`: CSV with ground truth labels (default: labeled_reviews.csv)
- `--output-dir`: Output directory (default: .)
- `--feature-type`: Which features to use: tfidf | glove | both (default: tfidf)
- `--svm`: Train SVM classifier
- `--logistic`: Train Logistic Regression classifier
- `--test-size`: Test set size (default: 0.2)
- `--svm-kernel`: SVM kernel: linear | rbf | poly (default: rbf)
- `--svm-c`: SVM C parameter (default: 1.0)
- Requires at least 1 model

**Output**:
- `svm_model.pkl` - Trained SVM model
- `logistic_regression_model.pkl` - Trained LR model
- `ml_models_report.json` - Detailed metrics and confusion matrices
- `training_config.json` - Training configuration
- `label_encoder.pkl` - Label encoder for inference

---

## Complete Pipeline Example

### Step 1: Label your data
```bash
python label_data.py \
    --input all_cleaned.csv \
    --output labeled_reviews.csv \
    --size 200 \
    --score-based \
    --rule-based
```

### Step 2: Generate text representations
```bash
python text_representation.py \
    --input labeled_reviews.csv \
    --output-dir representations \
    --tfidf \
    --glove \
    --max-features 5000 \
    --glove-dim 100
```

### Step 3: Test lexical-based models
```bash
python lexical_based_models.py \
    --input labeled_reviews.csv \
    --output lexical_results.csv \
    --output-report lexical_report.json \
    --vader \
    --afinn \
    --afinn-negation
```

### Step 4: Train ML-based models
```bash
python ml_based_models.py \
    --features representations/representations_combined.csv \
    --labels labeled_reviews.csv \
    --output-dir ml_models \
    --feature-type tfidf \
    --svm \
    --logistic \
    --svm-kernel rbf \
    --test-size 0.2
```

### Step 5: Compare results
```bash
# Compare lexical vs ML model performance
# Review lexical_report.json and ml_models/ml_models_report.json
```

---

## Expected Output Structure

```
section4/
├── labeled_reviews.csv                          # Ground truth labels
├── lexical_results.csv                          # Lexical model predictions
├── lexical_report.json                          # Lexical model metrics
├── representations/
│   ├── tfidf_matrix.csv                        # TF-IDF vectors
│   ├── glove_embeddings.csv                    # GloVe vectors
│   ├── representations_combined.csv            # Combined for ML models
│   ├── tfidf_vectorizer.pkl                    # Saved vectorizer
│   └── tfidf_features.json                     # Feature metadata
└── ml_models/
    ├── svm_model.pkl                           # Trained SVM
    ├── logistic_regression_model.pkl           # Trained LR
    ├── label_encoder.pkl                       # Label encoder
    ├── ml_models_report.json                   # ML metrics
    └── training_config.json                    # Training config
```

---

## Key Features

### Labeling
- Multiple annotators for robust ground truth
- Majority voting with ties resolved to first annotator
- Cohen's Kappa for inter-annotator agreement measurement

### Text Representation
- **TF-IDF**: Frequency-based approach, captures term importance
- **GloVe**: Semantic embeddings, captures word meaning
- Configurable n-grams, dimensions, and frequencies

### Lexical Models
- **VADER**: Pre-trained, rules-based, great for social media
- **AFINN**: Dictionary-based, customizable, with negation handling

### ML Models
- **SVM**: Non-linear classification with RBF/poly kernels
- **Logistic Regression**: Linear classifier, interpretable weights
- Both support multiple feature types and train/test splits

---

## Tips & Best Practices

1. **Start small**: Test with --size 100-200 first
2. **Combine annotators**: --score-based + --rule-based gives good variety
3. **Try different preprocessing**: Run multiple instances with different flag combinations
4. **Feature engineering**: Use --feature-type both to combine TF-IDF + GloVe
5. **Hyperparameter tuning**: Experiment with --svm-kernel and --svm-c
6. **Negation matters**: Always use --afinn-negation for better lexical results

---

## Dependencies

```
pandas
scikit-learn
nltk
gensim
numpy
textblob
```

Install with:
```bash
pip install pandas scikit-learn nltk gensim numpy textblob
```
