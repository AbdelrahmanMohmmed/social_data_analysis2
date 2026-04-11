# Quick Start Guide - Main Sentiment Pipeline

## First Time Setup

### 1. Install Dependencies
```bash
pip install pandas scikit-learn nltk gensim numpy textblob
```

### 2. Prepare Your Data
Ensure you have preprocessed data in `section3/`:
```
section3/
└── all_cleaned.csv  ← Required input file
```

The file should have columns: `userName`, `score`, `content`, `at`, `thumbsUpCount`, `source`, `app`

### 3. Navigate to section4
```bash
cd section4
```

## Running the Pipeline

### Option A: Full Automatic Pipeline (Recommended)
```bash
python main_sentiment_pipeline.py
```

**What happens:**
- ✓ Creates 4 different preprocessed datasets
- ✓ Labels each with 200 records
- ✓ Generates TF-IDF and GloVe representations
- ✓ Trains 32 ML model combinations
- ✓ Generates final comparison report

**Duration:** ~45-90 minutes

**Output:** `pipeline_results/comparison_all_results_*.csv` + `summary_report_*.json`

---

### Option B: Quick Test (No ML Training)
```bash
python main_sentiment_pipeline.py --skip-ml
```

**What happens:**
- ✓ Creates all 4 datasets
- ✓ Labels all 4 datasets  
- ✓ Generates representations

**Duration:** ~15-25 minutes

**Use case:** Verify everything works before full run

---

### Option C: Resume from Step N
```bash
# Continue from step 3 (skip dataset creation/labeling)
python main_sentiment_pipeline.py --step 3

# For testing with --skip-ml
python main_sentiment_pipeline.py --step 3 --skip-ml
```

---

## Understanding the 4 Datasets

| # | Name | Preprocessing | Expected Use |
|---|------|---|---|
| 1 | Minimal | Only removes emojis | Baseline |
| 2 | Stop Words | Lowercase + removes stop words | Frequency-based |
| 3 | Lemmatization | Lowercase + punctuation + lemmatization | Morphological |
| 4 | Full | All options combined | Production quality |

---

## 8 ML Model Combinations (Per Dataset = 32 Total)

| Features | Model Type | Speed | Accuracy | Best For |
|----------|-----------|-------|----------|----------|
| TF-IDF | SVM (RBF) | Slow | High | Best tradeoff |
| TF-IDF | SVM (Linear) | Fast | Medium-High | Quick results |
| TF-IDF | Logistic | Very Fast | Medium | Baselines |
| GloVe | SVM (RBF) | Slow | High | Semantic |
| GloVe | SVM (Linear) | Fast | Medium-High | Fast semantic |
| GloVe | Logistic | Very Fast | Medium | Semantic baseline |
| Both | SVM (RBF) | Very Slow | Highest | Best (slowest) |
| Both | Logistic | Slow | Medium-High | Balanced |

---

## Reading the Results

### 1. Console Output (Immediate)
```
TOP 10 BEST PERFORMING COMBINATIONS
Dataset                 Combination        F1-Score  Accuracy
Full Preprocessing      both_svm_rbf       0.8234    0.8100
Full Preprocessing      tfidf_svm_rbf      0.8156    0.8050
Lemmatization Only      tfidf_logistic     0.7843    0.7750
...

BEST COMBINATION PER DATASET
Dataset                 Combination        F1-Score
Full Preprocessing      both_svm_rbf       0.8234
Lemmatization Only      tfidf_logistic     0.7843
Stop Words Removed      glove_svm_rbf      0.7624
Minimal                 tfidf_svm_rbf      0.7234
```

### 2. Files Generated
```
pipeline_results/
├── comparison_all_results_20260411_143022.csv  ← Full results table
└── summary_report_20260411_143022.json         ← Statistics
```

### 3. Individual Model Reports
Each trained model has detailed metrics:
```
ml_results_full_both_svm_rbf/
├── svm_model.pkl              ← Trained model  
├── ml_models_report.json       ← Metrics
└── training_config.json        ← Configuration
```

---

## What to Do With Results

### 1. Find the Best Combination
Look at the top result from console output. Note:
- Dataset name (e.g., "Full Preprocessing")
- Combination name (e.g., "both_svm_rbf")
- F1-Score (primary metric)

### 2. Open Detailed Report
```bash
# View all 32 combinations in a spreadsheet
excel pipeline_results/comparison_all_results_*.csv

# Or analyze in Python
import pandas as pd
df = pd.read_csv("pipeline_results/comparison_all_results_*.csv")
print(df.nlargest(10, "F1-Score"))
```

### 3. Deploy the Best Model
```bash
# Best model is already trained and saved in:
# ml_results_<dataset>_<combination>/svm_model.pkl

import pickle

# Load the model
with open("ml_results_full_both_svm_rbf/svm_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the features
import pandas as pd
features = pd.read_csv("representations_full/representations_combined.csv")

# Make predictions
predictions = model.predict(features)
```

---

## Common Patterns & Insights

### Usually Best Performers:
1. **Full Preprocessing + Both Features + SVM (RBF)** - Highest accuracy, slowest
2. **Full Preprocessing + TF-IDF + SVM (RBF)** - Best speed/accuracy tradeoff
3. **Full Preprocessing + TF-IDF + Logistic** - Fast and reasonable

### Usually Weakest:
1. **Minimal + Logistic** - Least preprocessing, simplest model
2. **Minimal/Stop Words + Logistic** - Limited feature engineering

### Dataset Quality Ranking:
1. **Full Preprocessing** - Usually best (most cleaned)
2. **Lemmatization** - Usually 2nd (good normalization)
3. **Stop Words** - Usually 3rd (frequency-based)
4. **Minimal** - Usually weakest (least cleaned)

---

## Troubleshooting

### "all_cleaned.csv not found"
```bash
# Ensure you're in section4 directory
cd section4

# Verify file exists
ls ../section3/all_cleaned.csv
```

### Pipeline runs slow
```bash
# Quick test first
python main_sentiment_pipeline.py --skip-ml

# If that works, run full pipeline
python main_sentiment_pipeline.py
```

### Out of memory error
**Solution:** Reduce dataset size in main_sentiment_pipeline.py:
```python
"--size", "100",  # Change from "200" to "100"
```

### NLTK data missing error
```bash
# Download required data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

---

## Timeline Example

For 200-record datasets on a typical machine:

```
14:30:00 - Start
14:35:00 - Datasets created (5 min)
14:43:00 - All labeled (8 min)
15:05:00 - Representations complete (22 min)
16:20:00 - ML training complete (75 min)
16:22:00 - Report generated (2 min)
        = ~50 minutes total
```

---

## Next: Production Deployment

Once you have the best combination:

```bash
# Create a simple inference script
python -c "
import pickle
import pandas as pd

# Load model and preprocessing
model = pickle.load(open('ml_results_full_both_svm_rbf/svm_model.pkl', 'rb'))
encoder = pickle.load(open('ml_results_full_both_svm_rbf/label_encoder.pkl', 'rb'))

# Load new data
new_data = pd.read_csv('new_reviews.csv')

# Make predictions
predictions = model.predict(new_data)
predicted_labels = encoder.inverse_transform(predictions)

print(predicted_labels)
"
```

---

## Help & More Info

- **Individual script help**: `python label_data.py --help`
- **Detailed pipeline docs**: See `MAIN_PIPELINE_README.md`
- **Original guides**: See `PIPELINE_GUIDE.md`

---

**Ready? Run the pipeline:**
```bash
python main_sentiment_pipeline.py
```

Go grab a coffee ☕ - it'll run for 45-90 minutes!
