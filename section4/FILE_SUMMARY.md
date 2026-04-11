# Sentiment Analysis Project - Complete File Summary

## Directory Structure

```
section3/
└── text_preprocessing_v2.py         ← Enhanced with stop words removal

section4/
├── label_data.py                    ← Ground truth labeling (existing)
├── text_representation.py           ← TF-IDF + GloVe vectors (new)
├── lexical_based_models.py          ← VADER + AFINN models (new)
├── ml_based_models.py               ← SVM + Logistic Regression (new)
├── main_sentiment_pipeline.py       ← Master orchestrator (new)
│
├── Documentation/
│   ├── PIPELINE_GUIDE.md            ← Individual script usage guide
│   ├── MAIN_PIPELINE_README.md      ← Comprehensive pipeline documentation
│   ├── QUICK_START.md               ← Quick start guide (START HERE)
│   └── FILE_SUMMARY.md              ← This file
│
├── Outputs/ (Generated when running)
│   ├── dataset_01_minimal_emojis.csv
│   ├── dataset_02_stopwords.csv
│   ├── dataset_03_lemmatization.csv
│   ├── dataset_04_full.csv
│   │
│   ├── labeled_dataset_*.csv        (4 files)
│   │
│   ├── representations_minimal/
│   ├── representations_stopwords/
│   ├── representations_lemmatization/
│   ├── representations_full/
│   │
│   ├── ml_results_*.../             (32 directories for all combinations)
│   │
│   └── pipeline_results/
│       ├── comparison_all_results_*.csv
│       └── summary_report_*.json
```

---

## Script Files Created

### 1. **text_representation.py** (NEW)
**Purpose:** Convert text to numerical vectors

**Features:**
- TF-IDF vectorization (configurable n-grams, features, document frequency)
- GloVe word embeddings (100d or 300d with average pooling)
- Saves matrices, feature names, and combined representations

**Usage:**
```bash
python text_representation.py \
    --input labeled_reviews.csv \
    --output-dir representations \
    --tfidf --glove \
    --max-features 5000 \
    --glove-dim 100
```

**Outputs:**
- `tfidf_matrix.csv` - TF-IDF vectors
- `glove_embeddings.csv` - GloVe embeddings
- `representations_combined.csv` - All features for ML models
- `tfidf_vectorizer.pkl` - Serialized vectorizer
- `tfidf_features.json` - Feature metadata

---

### 2. **lexical_based_models.py** (NEW)
**Purpose:** Apply lexicon-based sentiment analysis

**Models:**
- **VADER**: Pre-trained lexicon + social media adaptation rules
- **AFINN**: Dictionary-based with optional negation handling

**Features:**
- Automatic negation detection ("not good" → negative)
- Cohen's Kappa inter-annotator agreement computation
- Accuracy, precision, recall, F1-score metrics
- Confusion matrix analysis

**Usage:**
```bash
python lexical_based_models.py \
    --input labeled_reviews.csv \
    --output lexical_results.csv \
    --vader --afinn --afinn-negation
```

**Outputs:**
- `lexical_results.csv` - Predictions from both models
- `lexical_report.json` - Detailed metrics

---

### 3. **ml_based_models.py** (NEW)
**Purpose:** Train machine learning sentiment classifiers

**Models:**
- **SVM**: Support Vector Machine with configurable kernels (linear, rbf, poly)
- **Logistic Regression**: Linear multi-class classifier

**Features:**
- Multiple feature types (TF-IDF, GloVe, or both combined)
- Train/test split with stratification
- Comprehensive metrics: accuracy, precision, recall, F1, confusion matrix
- Model persistence (pickle files)

**Usage:**
```bash
python ml_based_models.py \
    --features representations_combined.csv \
    --labels labeled_reviews.csv \
    --output-dir ml_models \
    --feature-type both \
    --svm --logistic \
    --svm-kernel rbf
```

**Outputs:**
- `svm_model.pkl` - Trained SVM model
- `logistic_regression_model.pkl` - Trained LR model
- `ml_models_report.json` - Detailed metrics
- `training_config.json` - Configuration used
- `label_encoder.pkl` - Label encoder for later use

---

### 4. **main_sentiment_pipeline.py** (NEW - MASTER ORCHESTRATOR)
**Purpose:** Automate the entire sentiment analysis workflow

**Workflow (5 Steps):**
1. **Create 4 preprocessed datasets** with different strategies
2. **Label each dataset** with 200 records (score-based + rule-based)
3. **Generate text representations** (TF-IDF + GloVe for each)
4. **Train ML models** (32 combinations = 4 datasets × 8 ML combinations)
5. **Generate comparison report** with rankings and statistics

**Usage:**
```bash
# Full pipeline
python main_sentiment_pipeline.py

# Skip ML training (quick test)
python main_sentiment_pipeline.py --skip-ml

# Continue from step 3
python main_sentiment_pipeline.py --step 3
```

**Outputs:**
- All 4 preprocessed datasets
- All 4 labeled datasets
- All text representations (TF-IDF + GloVe)
- 32 trained ML models with reports
- `pipeline_results/comparison_all_results_*.csv` - All 32 results ranked
- `pipeline_results/summary_report_*.json` - Statistics and best combinations

---

### 5. **text_preprocessing_v2.py** (ENHANCED)
**Purpose:** Text preprocessing with multiple cleaning strategies

**NEW Feature:**
- `--remove_stopwords` flag for removing common words

**All Flags:**
- `--lowercase` - Convert to lowercase
- `--remove_urls` - Remove HTTP/WWW links
- `--remove_emojis` - Remove emoji characters
- `--remove_punctuation` - Remove punctuation
- `--remove_stopwords` - Remove stop words (NEW)
- `--lemmatize` - Reduce words to root form (TextBlob)
- `--fix_spelling` - Correct spelling errors
- `--extract_tags` - Extract app tags

**Usage Examples:**
```bash
# Minimal preprocessing
python text_preprocessing_v2.py \
    --input section2/*.csv \
    --output dataset_minimal.csv \
    --remove_emojis

# Full preprocessing
python text_preprocessing_v2.py \
    --input section2/*.csv \
    --output dataset_full.csv \
    --lowercase --remove_urls --remove_emojis \
    --remove_punctuation --remove_stopwords --lemmatize
```

---

## Documentation Files

### 1. **QUICK_START.md** - NEW
**Start here!** 10-minute guide to run the complete pipeline.

**Contains:**
- Installation instructions
- 3 run options (full, quick test, resume)
- Expected outputs and reading results
- Simple troubleshooting

---

### 2. **MAIN_PIPELINE_README.md** - NEW
**Comprehensive guide** to the master orchestration script.

**Contains:**
- Detailed workflow explanation
- Command-line options explained
- Output structure documentation
- Understanding the 4 datasets
- Understanding 8 ML combinations
- Result interpretation guide
- Performance expectations
- Customization instructions
- Troubleshooting section

---

### 3. **PIPELINE_GUIDE.md** - (Updated)
**Reference guide** for all individual scripts.

**Contains:**
- Each script's purpose, arguments, outputs
- Complete pipeline workflow example
- Expected output structure
- Tips and best practices
- Dependencies list

---

### 4. **FILE_SUMMARY.md** - This File
**Complete overview** of all created files and their purposes.

---

## Workflow Summary

### What Gets Created

#### Step 1: 4 Preprocessed Datasets
```
dataset_01_minimal_emojis.csv        ~200 records, min preprocessing
dataset_02_stopwords.csv             ~200 records, stop words removed
dataset_03_lemmatization.csv         ~200 records, lemmatized
dataset_04_full.csv                  ~200 records, fully preprocessed
```

#### Step 2: 4 Labeled Datasets
```
labeled_dataset_01_minimal_emojis.csv
labeled_dataset_02_stopwords.csv
labeled_dataset_03_lemmatization.csv
labeled_dataset_04_full.csv
```

Each has:
- Original columns + `score_based` + `rule_based` + `final_label`
- Cohen's Kappa inter-annotator agreement score
- Majority voted labels

#### Step 3: Text Representations (4 directories)
```
representations_minimal/
representations_stopwords/
representations_lemmatization/
representations_full/
```

Each contains:
- `tfidf_matrix.csv` - 5000 TF-IDF features per record
- `glove_embeddings.csv` - 100d semantic embeddings per record
- `representations_combined.csv` - Original data + both representations

#### Step 4: ML Models (32 directories)
```
ml_results_minimal_tfidf_svm_rbf/
ml_results_minimal_tfidf_svm_linear/
ml_results_minimal_tfidf_logistic/
ml_results_minimal_glove_svm_rbf/
ml_results_minimal_glove_svm_linear/
ml_results_minimal_glove_logistic/
ml_results_minimal_both_svm_rbf/
ml_results_minimal_both_logistic/
...and 24 more (4 datasets × 8 combinations)
```

Each contains:
- Trained model (`.pkl` file)
- Performance metrics (`.json` report)
- Training configuration (`.json` file)

#### Step 5: Final Comparison
```
pipeline_results/
├── comparison_all_results_20260411_143022.csv
│   └── 32 rows: Dataset + Combination + Metrics (Accuracy, Precision, Recall, F1)
└── summary_report_20260411_143022.json
    └── Best combinations, statistics, rankings
```

---

## Performance Metrics Explained

| Metric | What It Measures | Range | Good | Excellent |
|--------|-----------------|-------|------|-----------|
| **Accuracy** | % of correct predictions | 0.0-1.0 | >0.70 | >0.80 |
| **Precision** | % of positive predictions that were correct | 0.0-1.0 | >0.70 | >0.85 |
| **Recall** | % of actual positives that were found | 0.0-1.0 | >0.70 | >0.85 |
| **F1-Score** | Harmonic mean (best overall metric) | 0.0-1.0 | >0.70 | >0.80 |

**Primary metric: F1-Score** (balances precision & recall)

---

## How to Use Each Script Independently

### Pre-labeled, at-scale training:
```bash
python text_representation.py --input my_data.csv --tfidf --glove
python ml_based_models.py --features representations_combined.csv \
    --labels my_data.csv --svm --logistic
```

### Just testing lexical models on labeled data:
```bash
python lexical_based_models.py --input my_labeled_data.csv \
    --vader --afinn --afinn-negation
```

### Custom preprocessing:
```bash
python text_preprocessing_v2.py \
    --input raw_reviews.csv \
    --output cleaned_reviews.csv \
    --lowercase --lemmatize --remove_punctuation
```

### Complete run with orchestration:
```bash
python main_sentiment_pipeline.py  # Full 50-90 minute run
```

---

## Key Features Summary

| Feature | Script | Status |
|---------|--------|--------|
| Ground truth labeling with multiple annotators | label_data.py | Existing |
| Cohen's Kappa agreement measurement | label_data.py | Existing |
| Text preprocessing (5 options) | text_preprocessing_v2.py | Enhanced |
| Stop words removal | text_preprocessing_v2.py | NEW ✓ |
| TF-IDF vectorization | text_representation.py | NEW ✓ |
| GloVe embeddings | text_representation.py | NEW ✓ |
| VADER sentiment analysis | lexical_based_models.py | NEW ✓ |
| AFINN with negation handling | lexical_based_models.py | NEW ✓ |
| SVM classifier | ml_based_models.py | NEW ✓ |
| Logistic Regression | ml_based_models.py | NEW ✓ |
| Mixed pipeline combinations | ml_based_models.py | NEW ✓ |
| Master orchestration | main_sentiment_pipeline.py | NEW ✓ |
| Automatic comparison reporting | main_sentiment_pipeline.py | NEW ✓ |

---

## Next Steps

1. **Run the pipeline**: `python main_sentiment_pipeline.py`
2. **Review results**: Check `pipeline_results/comparison_all_results_*.csv`
3. **Deploy best model**: Use the highest-scoring ML model
4. **Iterate**: Modify parameters and rerun as needed

---

## Questions?

- **Getting started?** → Read `QUICK_START.md`
- **Need details?** → Read `MAIN_PIPELINE_README.md`
- **Script reference?** → Read `PIPELINE_GUIDE.md`
- **Individual script help?** → `python script.py --help`

---

**Happy sentiment analyzing! 🚀**
