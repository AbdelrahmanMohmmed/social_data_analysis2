# Task 4: Quick Start Guide

## 🚀 Run Everything in One Command

```bash
cd /media/abdo/Games/social_data_analysis/section5
python task4_orchestrator.py
```

This runs all 5 steps automatically! ✅

---

## 📋 What Each Step Does

### Step 1: Model Evaluation (5-10 min)
- Evaluates all 12 trained models
- Calculates: Accuracy, Precision, Recall, F1-Score, MCC, ROC-AUC
- Creates visualization comparing all models
- **Output:** `model_evaluation_results.json`, `model_comparison.png`

### Step 2: Error Analysis (2-3 min)
- Finds incorrect predictions (~40 cases)
- Analyzes why model failed
- Identifies patterns (e.g., "Neutral confused with Positive")
- **Output:** `error_analysis_report.json`, `error_analysis_visualization.png`

### Step 3: Hyperparameter Optimization (30-60 min)
- Tests many hyperparameter combinations using:
  - Grid Search (systematic)
  - Random Search (faster)
  - Genetic Algorithm (evolutionary)
- Saves best model
- **Output:** `optimized_svm_model.pkl`, `optimization_results.csv`

### Step 4: Model Comparison (5 min)
- Compares original vs optimized model
- Shows improvements in each metric
- Checks: Does it beat random (1/3 = 0.33) by 20%?
- **Output:** `model_comparison_report.json`, `model_comparison.png`

### Step 5: Deployment (Already Done)
- Flask REST API ready to use
- Streamlit dashboard ready to use
- Shows how to deploy

---

## 📊 Expected Results

After running, you'll see something like:

```
Step 1: MODEL EVALUATION
✓ Baseline SVM RBF: F1=0.7705, Accuracy=0.80

Step 2: ERROR ANALYSIS
✓ Found 40 misclassified cases
  Main issue: Neutral confused with Positive (8 cases)

Step 3: HYPERPARAMETER OPTIMIZATION
✓ Grid Search Best: F1=0.8100
✓ Random Search: F1=0.8050
✓ Genetic Algorithm: F1=0.8120

Step 4: MODEL COMPARISON
✓ Baseline F1: 0.7705
✓ Optimized F1: 0.8120
✓ Improvement: +5.4%
✓ PASSED: 143% above random chance!

Step 5: DEPLOYMENT
✓ Flask API: python flask_app.py
✓ Streamlit UI: streamlit run streamlit_app.py
```

---

## 🔧 Run Only Specific Steps

```bash
# Just evaluation
python task4_orchestrator.py --step 1

# Just error analysis
python task4_orchestrator.py --step 2

# Just optimization
python task4_orchestrator.py --step 3

# Just comparison
python task4_orchestrator.py --step 4

# Skip a step (e.g., skip step 3)
python task4_orchestrator.py --skip 3
```

---

## 📁 Output Files

Everything goes to `task4_results/` folder:

```
📊 evaluation results
├── model_evaluation_results.json
├── model_evaluation_results.csv
└── model_comparison.png

🔍 error analysis
├── error_analysis_report.json
└── error_analysis_visualization.png

⚙️ optimization results
└── optimization_results.csv

📈 comparison
├── model_comparison_report.json
└── model_comparison.png

🎯 saved model
└── optimized_svm_model.pkl
```

---

## 🚀 Deploy After Optimization

### Start REST API:
```bash
python flask_app.py
```

Then test:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!"}'
```

### Start Dashboard:
```bash
streamlit run streamlit_app.py
```

Open: http://localhost:8501

---

## ⏱️ Estimated Times

| Step | Time |
|------|------|
| Evaluation | 5-10 min |
| Error Analysis | 2-3 min |
| Optimization | 30-60 min |
| Comparison | 5 min |
| **Total** | **45-80 min** |

To speed up: use `--skip 3` to skip optimization (30-60 min)

---

## 🎯 Success Criteria

Your Task 4 is complete if:

✅ Model Evaluation: All 12 models evaluated  
✅ Error Analysis: Failed cases identified & analyzed  
✅ Optimization: Hyperparameters tuned  
✅ Comparison: F1-Score improved (ideally by >5%)  
✅ Deployment: Flask & Streamlit both working  
✅ Threshold: **>0.40 F1-Score** (20% above random 0.33)

---

## 📈 Key Metrics to Track

| Metric | Target | Status |
|--------|--------|--------|
| Baseline F1-Score | - | 0.7705 |
| Optimized F1-Score | >0.40 | ? |
| Improvement | >5% | ? |
| Accuracy | >80% | ? |
| Error Rate Analysis | All cases | ? |

---

## 🆘 Troubleshooting

**Q: "Module not found"**  
A: Run `pip install -r requirements.txt` first

**Q: "Model not found" in section4**  
A: Make sure you ran the training pipeline in section4 first

**Q: Takes too long (>2 hours)**  
A: Skip optimization: `python task4_orchestrator.py --skip 3`

**Q: Out of memory**  
A: Reduce CV folds or add `--skip 3`

---

## 📚 Full Documentation

For detailed information, see: `TASK4_COMPLETE_GUIDE.md`

---

## 🎉 That's It!

```bash
# Just run this one command:
python task4_orchestrator.py

# Then follow the output
# All results go to: task4_results/
```

Questions? Check the full guide or run individual modules!
