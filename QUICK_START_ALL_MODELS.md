# 🚀 QUICK START - 4 MODELS + STREAMLIT APP

## ✅ What's Ready

✅ **4 Sentiment Analysis Models Trained**
- SVM (RBF) - 80% accuracy ⭐ BEST
- Logistic Regression - 80% accuracy  
- Decision Tree - 70% accuracy
- Random Forest - 65% accuracy

✅ **Streamlit Dashboard Running**
- http://localhost:8501
- Model comparison
- ROC curves visualization
- Single & batch analysis

---

## 🎯 Try It Now

### Option 1: Open in Browser
```
http://localhost:8501
```

### Option 2: Test a Prediction
```bash
python3 << 'EOF'
from section5.model_loader import get_model
model = get_model()
result = model.predict("not good")
print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.2%}")
EOF
```

---

## 📊 Model Performance

| Model | Accuracy | Best For |
|-------|----------|----------|
| **SVM (RBF)** | 80% | ⭐ Production |
| **Logistic Regression** | 80% | Interpretability |
| **Decision Tree** | 70% | Quick insight |
| **Random Forest** | 65% | Ensemble |

---

## 🎮 Dashboard Features

### 🔍 Single Text Mode
Test one phrase at a time
- Sentiment classification
- Confidence score
- Class probabilities

### 📦 Batch Analysis
Analyze multiple reviews
- Upload CSV or paste text
- Sentiment distribution
- Download results

### 🤖 Model Info
Compare all 4 models
- Model architectures
- Performance metrics
- **ROC Curves** with AUC scores
- Training statistics

---

## 📂 Trained Models Location

```
/media/abdo/Games/social_data_analysis/section4/ml_results_full_tfidf_all_models/

Files:
  • svm_model.pkl (305 KB)
  • logistic_regression_model.pkl (7.9 KB)
  • decision_tree_model.pkl (3.7 KB)
  • random_forest_model.pkl (853 KB)
  • label_encoder.pkl
  • tfidf_vectorizer.pkl
  • ml_models_report.json
  • training_config.json
```

---

## 🔧 Troubleshooting

### App not loading?
```bash
# Kill old process
pkill -f "streamlit run"

# Restart
cd /media/abdo/Games/social_data_analysis/section5
streamlit run streamlit_app.py &
```

### Want to retrain?
```bash
cd /media/abdo/Games/social_data_analysis
python train_all.py
```

### Want to train specific models?
```bash
cd /media/abdo/Games/social_data_analysis/section4
python ml_based_models.py \
  --features representations_full/tfidf_matrix.csv \
  --labels labeled_dataset_04_full.csv \
  --output-dir my_models \
  --svm --decision-tree
```

---

## 💡 Tips

1. **Best Results**: Use SVM model (default)
2. **Fastest**: Decision Tree (lowest latency)
3. **Most Data**: Random Forest (uses all training data via bagging)
4. **Most Interpretable**: Logistic Regression or Decision Tree

---

## 📈 Training Data Used

- **Samples**: 200 reviews
- **Train**: 160 | **Test**: 40
- **Features**: 303 TF-IDF dimensions
- **Classes**: Positive (111), Negative (75), Neutral (14)

---

## 🎯 Next Steps

1. ✅ **Open Dashboard**: http://localhost:8501
2. ✅ **Test Models**: Try "not good" or "very bad"
3. ✅ **View Comparison**: Check Model Info tab
4. ✅ **See ROC Curves**: Scroll down in Model Info
5. ✅ **Download Results**: Use batch analysis mode

---

**Status**: 🟢 All systems operational
**Streamlit**: 🟢 Running on port 8501
**Models**: 🟢 4/4 trained and loaded
