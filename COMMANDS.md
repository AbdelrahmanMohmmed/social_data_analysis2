# 🎯 COPY-PASTE COMMANDS (Ready to Run)

Save this file and copy commands one by one. They're ready to go!

---

## ⏱️ 30-MINUTE FULL IMPLEMENTATION

### Command 1: Generate Better Training Data (5 min)

```bash
cd /media/abdo/Games/social_data_analysis/section3

python text_preprocessing_v3.py \
  --input ../section2/* \
  --output all_cleaned_v2.csv \
  --lowercase --remove_urls --remove_emojis --remove_numbers \
  --remove_punctuation --remove_stopwords --lemmatize --extract_tags
```

**Expected output:**
```
[*] Loading data from ../section2/...
[*] Processing 3000+ records...
[*] Applying sentiment-aware preprocessing...
✓ Saved: all_cleaned_v2.csv (5MB+)
✓ Complete
```

**After running:**
- [ ] File exists: `ls -lh section3/all_cleaned_v2.csv`
- [ ] Size > 1MB

---

### Command 2: Retrain All 12 Models (15 min)

```bash
cd /media/abdo/Games/social_data_analysis/section4

python main_sentiment_pipeline.py \
  --section3 ../section3 \
  --section4 . \
  --output pipeline_results_v2
```

**Expected output:**
```
[*] Creating v2 datasets...
[*] Training 12 models with improved preprocessing...
✓ Model 1/12: minimal_tfidf_svm_rbf_v2 ✓
✓ Model 2/12: minimal_tfidf_svm_linear_v2 ✓
✓ Model 3/12: minimal_tfidf_logistic_v2 ✓
... (9 more models)
✓ All 12 v2 models trained successfully
```

**After running:**
- [ ] Command completed (no errors)
- [ ] Check: `ls -d section4/ml_results_*_v2/ | wc -l` should show `12`

---

### Command 3: Test With Validation Framework (2 min)

```bash
cd /media/abdo/Games/social_data_analysis

python section4/model_validation_v2.py \
  --model-dir section4/ml_results_stopwords_v2_tfidf_svm_rbf/
```

**Expected output:**
```
Testing: ml_results_stopwords_v2_tfidf_svm_rbf/

✓ "not good"             → Negative  [PASS]
✓ "not bad"              → Positive  [PASS]
✓ "very good"            → Positive  [PASS]
✓ "pretty bad"           → Negative  [PASS]
✓ "really good"          → Positive  [PASS]
✓ "so bad"               → Negative  [PASS]
✓ "last night positive"  → Negative  [PASS]

Validation Results:
  Passed: 7/7 (100%)
  Status: ✓ PASS - All tests successful!
```

**After running:**
- [ ] Check: All 7 tests show `[PASS]`
- [ ] Final line shows: `✓ PASS`
- [ ] Success means "not good" now correctly returns NEGATIVE

---

### Command 4: Simple Interactive Test (1 min)

```bash
python3 << 'PYTHON_TEST'
import sys
sys.path.insert(0, '/media/abdo/Games/social_data_analysis')

from section4.model_loader import SentimentModelLoader

# Load new v2 model
loader = SentimentModelLoader(
    model_dir="/media/abdo/Games/social_data_analysis/section4/ml_results_stopwords_v2_tfidf_svm_rbf"
)

# Test cases
test_cases = [
    "not good",
    "not bad",
    "very good",
    "pretty bad",
]

print("Testing v2 models:")
print("-" * 50)
for test in test_cases:
    result = loader.predict(test)
    print(f"'{test}' → {result['sentiment']}")
print("-" * 50)
print("✓ Tests complete")
PYTHON_TEST
```

**Expected output:**
```
Testing v2 models:
--------------------------------------------------
'not good' → Negative
'not bad' → Positive
'very good' → Positive
'pretty bad' → Negative
--------------------------------------------------
✓ Tests complete
```

---

## 🔧 OPTIONAL: Database Update to v2 Models

If you want to use v2 models in production (optional):

### Command 5: Update Flask Configuration

```bash
# Edit the configuration to use v2 models
cd /media/abdo/Games/social_data_analysis

# Backup old config
cp section5/model_loader.py section5/model_loader.py.backup

# Update to use v2 model
sed -i 's/ml_results_full_tfidf_svm_rbf/ml_results_full_tfidf_svm_rbf_v2/g' section5/model_loader.py

# Verify change
grep "ml_results_full_tfidf_svm_rbf" section5/model_loader.py
# Should show: ml_results_full_tfidf_svm_rbf_v2
```

---

### Command 6: Restart Services

```bash
# Kill old processes
pkill -f flask_app
pkill -f streamlit
sleep 2

# Start with new models
cd /media/abdo/Games/social_data_analysis

# Terminal 1: Flask API
python section5/flask_app.py

# Terminal 2: Streamlit UI (in another terminal)
streamlit run section5/streamlit_app.py
```

---

## 🧪 FINAL VERIFICATION TESTS

### Test 1: Command Line

```bash
# Direct prediction test
python3 -c "
from section4.model_loader import SentimentModelLoader
loader = SentimentModelLoader('section4/ml_results_stopwords_v2_tfidf_svm_rbf')

tests = ['not good', 'not bad', 'very good']
for t in tests:
    print(f'{t}: {loader.predict(t)[\"sentiment\"]}')"
```

Expected:
```
not good: Negative
not bad: Positive
very good: Positive
```

### Test 2: Flask API

```bash
# Test via HTTP
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "not good"}'
```

Expected:
```json
{"sentiment": "Negative", "confidence": 0.85, ...}
```

### Test 3: Web UI

Open in browser:
```
http://localhost:8501
```

Type in text field: `not good`
Expected: Shows **Negative** sentiment

---

## ⏮️ ROLLBACK (If Needed)

### Go Back to v1 Models

```bash
# Restore backup
cp section5/model_loader.py.backup section5/model_loader.py

# Or manually edit to old path:
sed -i 's/_v2//g' section5/model_loader.py

# Restart services
pkill -f flask_app
pkill -f streamlit
```

---

## 📊 Performance Comparison Commands

### Before (v1): Check accuracy of old models

```bash
cd /media/abdo/Games/social_data_analysis

python section4/model_validation_v2.py \
  --model-dir section4/ml_results_stopwords_tfidf_svm_rbf/ 2>/dev/null | tail -20
```

Expected: 5/7 tests pass

### After (v2): Check accuracy of new models

```bash
cd /media/abdo/Games/social_data_analysis

python section4/model_validation_v2.py \
  --model-dir section4/ml_results_stopwords_v2_tfidf_svm_rbf/ 2>/dev/null | tail -20
```

Expected: 7/7 tests pass

---

## 🎯 ONE-LINER: Full Implementation Chain

Copy and run all at once (takes ~30 min):

```bash
cd /media/abdo/Games/social_data_analysis && \
echo "Step 1: Generate data..." && \
python section3/text_preprocessing_v3.py --input section2/* --output section3/all_cleaned_v2.csv --lowercase --remove_urls --remove_emojis --remove_numbers --remove_punctuation --remove_stopwords --lemmatize --extract_tags && \
echo "✓ Data ready" && \
echo "" && \
echo "Step 2: Train models (this takes 15 min)..." && \
python section4/main_sentiment_pipeline.py --section3 section3 --section4 section4 --output pipeline_results_v2 && \
echo "✓ Models trained" && \
echo "" && \
echo "Step 3: Validate..." && \
python section4/model_validation_v2.py --model-dir section4/ml_results_stopwords_v2_tfidf_svm_rbf/ && \
echo "" && \
echo "✓✓✓ ALL COMPLETE ✓✓✓"
```

---

## 📝 Quick Reference

**Command to run first:**
```bash
cd /media/abdo/Games/social_data_analysis/section3
python text_preprocessing_v3.py --input ../section2/* --output all_cleaned_v2.csv \
  --lowercase --remove_urls --remove_emojis --remove_numbers \
  --remove_punctuation --remove_stopwords --lemmatize --extract_tags
```

**Command to test results:**
```bash
cd /media/abdo/Games/social_data_analysis
python section4/model_validation_v2.py --model-dir section4/ml_results_stopwords_v2_tfidf_svm_rbf/
```

**Expected success indicator:**
```
✓ "not good" → Negative [PASS]
✓ "not bad"  → Positive [PASS]
7/7 tests pass (100%)
Status: ✓ PASS
```

---

## ❓ Troubleshooting Commands

### Check if Python dependencies are installed
```bash
python -c "import nltk, pandas, sklearn, textblob; print('✓ All packages installed')"
```

### Check if files exist
```bash
ls -lh section3/text_preprocessing_v3.py section4/model_validation_v2.py
```

### Check if v3 preprocessing works
```bash
cd section3
python -c "from text_preprocessing_v3 import preprocess_text; print(preprocess_text('not good'))"
# Should output: "not good" (both preserved)
```

### Check if v2 training data was created
```bash
wc -l section3/all_cleaned_v2.csv
du -h section3/all_cleaned_v2.csv
```

### Check if v2 models were created
```bash
ls -d section4/ml_results_*_v2/ | head -5
# Should show at least 5 directories
```

### Run a model test manually
```bash
python3 -c "
import pickle
import os

# Load model
model_path = 'section4/ml_results_stopwords_v2_tfidf_svm_rbf/svm_model.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Load vectorizer  
vec_path = 'section4/ml_results_stopwords_v2_tfidf_svm_rbf/vectorizer.pkl'
with open(vec_path, 'rb') as f:
    vec = pickle.load(f)

# Test
text = 'not good'
X = vec.transform([text])
pred = model.predict(X)[0]
print(f'{text} → {pred}')
"
```

---

## 🎉 Success Verification Checklist

Verify after each command:

### After Command 1 (Data Generation)
- [ ] File created: `section3/all_cleaned_v2.csv` exists
- [ ] File size > 1MB
- [ ] Contains both "not" and "good" together in rows

### After Command 2 (Training)
- [ ] 12 v2 model directories created
- [ ] Verify: `ls -d section4/ml_results_*_v2 | wc -l` shows `12`
- [ ] No errors in training output

### After Command 3 (Validation)
- [ ] All 7 tests show `[PASS]`
- [ ] Final output: `7/7 (100%) Status: ✓ PASS`
- [ ] Specifically check:
  - `"not good" → Negative [PASS]`
  - `"not bad" → Positive [PASS]`

### After Commands 4-6 (Web Deployment)
- [ ] Flask restarted successfully
- [ ] Streamlit restarted successfully
- [ ] Web UI accessible
- [ ] Manual test shows: `"not good" → Negative`

---

## 💾 Save & Reference

**Save this file location:**
```
/media/abdo/Games/social_data_analysis/COMMANDS.md
```

**Quick access:**
```bash
cat /media/abdo/Games/social_data_analysis/COMMANDS.md | head -50
```

**All commands in one place for reference & re-running**
