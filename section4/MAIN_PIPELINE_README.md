# Main Sentiment Analysis Pipeline - Complete Guide

This is the **master orchestration script** that automates the entire sentiment analysis workflow in one command.

## What It Does

The `main_sentiment_pipeline.py` script automates all 5 steps:

```
Step 1: Create 4 Preprocessed Datasets
    ├─ Dataset 1: Minimal (only remove emojis)
    ├─ Dataset 2: Stop Words Removed (lowercase + stop words)
    ├─ Dataset 3: Lemmatization (lowercase + punctuation + lemmatization)
    └─ Dataset 4: Full Preprocessing (all options combined)
         ↓
Step 2: Label Each Dataset
    └─ 200 records each labeled with score-based + rule-based annotators
         ↓
Step 3: Create Text Representations  
    └─ For each labeled dataset: TF-IDF + GloVe (100d)
         ↓
Step 4: Train Mixed Pipeline ML Models
    ├─ TF-IDF + SVM (RBF kernel)
    ├─ TF-IDF + SVM (Linear kernel)
    ├─ TF-IDF + Logistic Regression
    ├─ GloVe + SVM (RBF kernel)
    ├─ GloVe + SVM (Linear kernel)
    ├─ GloVe + Logistic Regression
    ├─ Both (TF-IDF + GloVe) + SVM (RBF kernel)
    └─ Both + Logistic Regression
         ↓
Step 5: Generate Comprehensive Comparison Report
    ├─ All results compiled into CSV
    ├─ Top 10 best combinations
    ├─ Best combination per dataset
    ├─ Best combination per model type
    └─ Statistical summary
```

## Installation

Make sure you have all required dependencies:

```bash
pip install pandas scikit-learn nltk gensim numpy textblob
```

## Usage

### Basic Usage - Run Full Pipeline

```bash
cd section4
python main_sentiment_pipeline.py
```

This will:
1. Create all 4 datasets
2. Label each one
3. Generate text representations for all
4. Train 32 different ML model combinations (4 datasets × 8 combinations)
5. Generate comprehensive comparison report with all results

### Advanced Usage

#### Start from a specific step:
```bash
# Continue from step 3 (skip dataset creation and labeling)
python main_sentiment_pipeline.py --step 3
```

#### Skip ML training for quick testing:
```bash
# Run only up to step 3 (create representations)
python main_sentiment_pipeline.py --skip-ml
```

#### Combine options:
```bash
# Start from step 2, skip ML training
python main_sentiment_pipeline.py --step 2 --skip-ml
```

## Output Structure

```
section4/
├── dataset_01_minimal_emojis.csv                # Original datasets
├── dataset_02_stopwords.csv
├── dataset_03_lemmatization.csv
├── dataset_04_full.csv
│
├── labeled_dataset_*.csv                        # Labeled versions
│
├── representations_minimal/                     # Text representations
├── representations_stopwords/
├── representations_lemmatization/
├── representations_full/
│   ├── tfidf_matrix.csv
│   ├── glove_embeddings.csv
│   ├── representations_combined.csv
│   ├── tfidf_vectorizer.pkl
│   └── tfidf_features.json
│
├── ml_results_minimal_tfidf_svm_rbf/           # ML model results
├── ml_results_minimal_tfidf_svm_linear/
├── ml_results_minimal_tfidf_logistic/
├── ml_results_minimal_glove_svm_rbf/
│   ├── svm_model.pkl
│   ├── ml_models_report.json
│   └── training_config.json
│ ... (32 total ML results directories)
│
└── pipeline_results/
    ├── comparison_all_results_20260411_143022.csv    # Master results
    └── summary_report_20260411_143022.json           # Summary statistics
```

## Understanding the Results

### Comparison CSV (`comparison_all_results_*.csv`)

Contains one row per model combination with columns:
- **Dataset**: Which preprocessing strategy (minimal, stopwords, lemmatization, full)
- **Combination**: Feature + model type (e.g., "tfidf_svm_rbf")
- **Model**: The actual model (e.g., "SVM (rbf)")
- **Accuracy**: Overall accuracy score
- **Precision**: Weighted precision
- **Recall**: Weighted recall
- **F1-Score**: Weighted F1 score (primary metric)

### Console Output

When the pipeline completes, you'll see:

**TOP 10 BEST PERFORMING COMBINATIONS**
```
Dataset                 Combination              F1-Score  Accuracy
Full Preprocessing      both_svm_rbf             0.8234    0.8100
Full Preprocessing      tfidf_svm_rbf            0.8156    0.8050
...
```

**BEST COMBINATION PER DATASET**
```
Dataset                  Combination              F1-Score  Accuracy
Full Preprocessing       both_svm_rbf             0.8234    0.8100
Lemmatization Only       tfidf_logistic           0.7843    0.7750
...
```

**BEST COMBINATION PER MODEL TYPE**
```
Model                                Dataset              Combination        F1-Score
SVM (rbf)                           Full Preprocessing   both_svm_rbf        0.8234
Logistic Regression                 Full Preprocessing   both_logistic       0.8012
...
```

## Dataset Descriptions

### 1. Minimal (Emojis Only)
- **Preprocessing**: Only removes emoji characters
- **Use case**: Test if emoji removal alone helps
- **Expected**: Baseline, likely lower performance

### 2. Stop Words Removed  
- **Preprocessing**: Lowercase + remove stop words (a, the, is, etc.)
- **Use case**: Reduce noise with frequency-based approach
- **Expected**: Moderate improvement from minimal

### 3. Lemmatization Only
- **Preprocessing**: Lowercase + remove punctuation + lemmatization (words → root form)
- **Use case**: Reduce sparsity by normalizing word forms
- **Expected**: Good for capturing word meanings

### 4. Full Preprocessing
- **Preprocessing**: Lowercase + URLs + emojis + punctuation + stop words + lemmatization
- **Use case**: Maximum cleaning for production use
- **Expected**: Often the best overall performer

## Model Combinations Explained

### Features:
- **TF-IDF**: Term frequency-inverse document frequency (sparse, interpretable)
- **GloVe**: Word embeddings (dense, semantic meaning)
- **Both**: Concatenate both representations

### Models:
- **SVM RBF**: Support Vector Machine with RBF kernel (non-linear)
- **SVM Linear**: Support Vector Machine with linear kernel (faster)
- **Logistic Regression**: Fast linear classifier (baseline)

## Interpreting Results

### What makes a "good" combination?
1. **High F1-Score** (primary metric) – balances precision/recall
2. **Consistent across runs** – no huge variance
3. **Reasonable training time** – TF-IDF usually faster than GloVe

### Common Findings:
- **Full preprocessing + Both features + SVM (RBF)** is often best but slowest
- **TF-IDF + SVM (RBF)** often provides best speed/accuracy tradeoff
- **Logistic Regression** is fastest but may have lower accuracy
- **GloVe alone** sometimes outperforms TF-IDF for semantic understanding

## Progress Tracking

The script provides real-time feedback:

```
[14:30:22] ℹ STEP 1: Creating 4 preprocessed datasets
[14:30:23] ▶ Creating: Minimal (Emojis Only)
[14:30:45] ✓ Created: Minimal (Emojis Only)
...
[14:32:10] ▶ Training ML: dataset_01_minimal + tfidf_svm_rbf
[14:33:05] ✓ tfidf_svm_rbf: Trained successfully
...
```

## Troubleshooting

### Issue: "Failed to create datasets"
**Solution**: Ensure `../section3/all_cleaned.csv` exists with columns: userName, score, content, at, thumbsUpCount, source, app

### Issue: "Labeling fails with memory error"
**Solution**: Reduce `--size` parameter in Step 2, or increase system RAM

### Issue: "Text representation very slow"
**Solution**: 
- Reduce `--max-features` parameter (currently 5000)
- Use only `--tfidf` or `--glove`, not both (use `--skip-ml` to test)

### Issue: "ML training fails with numpy/sklearn error"
**Solution**: Ensure compatibility: `pip install --upgrade scikit-learn numpy pandas`

## Customization

To modify the pipeline, edit the `PipelineConfig` class:

```python
class PipelineConfig:
    def __init__(self):
        # Change default input file
        self.raw_data = "your_data.csv"
        
        # Modify dataset strategies
        self.datasets = {
            "custom": {
                "name": "My Custom Preprocessing",
                "flags": ["--lowercase", "--lemmatize"],
                "output": "dataset_custom.csv"
            }
            # ... add more
        }
        
        # Add/remove ML combinations
        self.ml_combinations = [
            {"features": "tfidf", "model": "svm", "kernel": "rbf"},
            # ... modify as needed
        ]
```

## Performance Expectations

On a typical machine (8GB RAM, modern CPU):
- **Step 1 (Datasets)**: ~2-5 min total
- **Step 2 (Labeling)**: ~3-8 min total (200 records × 4)
- **Step 3 (Representations)**: ~10-20 min total (TF-IDF + GloVe)
- **Step 4 (ML Training)**: ~30-60 min total (32 combinations)
- **Total**: ~45-90 minutes for full pipeline

Use `--skip-ml` to test quickly (~15-25 min for steps 1-3 only).

## Next Steps After Running

1. **Review `comparison_all_results_*.csv`**: Rank all combinations
2. **Check specific model reports**: Open individual `ml_models_report.json` files
3. **Focus on best combination**: Production deployment uses top-performing model
4. **Iterate**: Modify preprocessing or model parameters based on results

## Example: Production Deployment

Once you've identified the best combination (e.g., "Full + Both Features + SVM RBF"):

```bash
# Train final model on full dataset
python ml_based_models.py \
    --features representations_full/representations_combined.csv \
    --labels labeled_dataset_04_full.csv \
    --output-dir production_model \
    --feature-type both \
    --svm \
    --svm-kernel rbf

# Load and use the model
import pickle
model = pickle.load(open('production_model/svm_model.pkl', 'rb'))
predictions = model.predict(new_features)
```
