# 🚀 Quick Start Guide - Sentiment Analysis Web Apps

## What You Have

✅ **Flask REST API** (`flask_app.py`) - Backend server for predictions  
✅ **Streamlit Dashboard** (`streamlit_app.py`) - Interactive UI  
✅ **API Client** (`api_client.py`) - Python client for integration  
✅ **Performance Tools** (`run_benchmark.py`) - Testing & benchmarking  
✅ **Model Loader** (`model_loader.py`) - Unified model interface  
✅ **Utilities** (`web_utils.py`) - Text processing & analytics  

## Installation (One-Time Setup)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify model files exist
ls ml_results_full_tfidf_svm_rbf/
ls representations_full/
```

## Running the Apps

### Option 1: Just the REST API

```bash
python flask_app.py
# Access at: http://localhost:5000
# API Docs at: http://localhost:5000/
```

**Test it:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This is amazing!"}'
```

### Option 2: Just the Streamlit Dashboard

```bash
streamlit run streamlit_app.py
# Access at: http://localhost:8501
```

### Option 3: Both (RECOMMENDED)

**Terminal 1:**
```bash
python flask_app.py
```

**Terminal 2:**
```bash
streamlit run streamlit_app.py
```

Now you have:
- REST API at `http://localhost:5000`
- Dashboard at `http://localhost:8501`

## Common Workflows

### 🔍 Analyze Single Text

**Via Streamlit Dashboard:**
1. Open `http://localhost:8501`
2. Enter text
3. Click "Predict"

**Via REST API:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here"}'
```

**Via Python:**
```python
from api_client import SentimentAPIClient
client = SentimentAPIClient()
result = client.predict("Your text here")
print(result['sentiment'])  # Output: Positive/Negative/Neutral
```

### 📦 Analyze Multiple Texts

**Via Streamlit Dashboard:**
1. Select "Batch Analysis" mode
2. Paste texts (one per line) OR upload CSV
3. Click "Analyze All"

**Via REST API:**
```bash
curl -X POST http://localhost:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Text 1", "Text 2", "Text 3"]}'
```

**Via Python:**
```python
from api_client import SentimentAPIClient
client = SentimentAPIClient()

# From list
results = client.predict_batch(["Great!", "Terrible!", "Okay"])

# From CSV file
df = client.predict_csv('reviews.csv', output_filepath='results.csv')
```

### 📊 Benchmark Performance

```bash
python run_benchmark.py
```

This will:
- ⏱️ Test prediction speed
- 🎯 Check consistency
- 💪 Test edge cases
- 📈 Generate report

## Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 80% |
| **F1-Score** | 0.7705 |
| **Prediction Time** | <50ms |
| **Classes** | Positive, Negative, Neutral |

## API Endpoints Summary

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | API documentation |
| `/health` | GET | Health check |
| `/model/info` | GET | Model information |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch predictions |

## Example Request/Response

**Request:**
```json
{
  "text": "This product is amazing!"
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "text": "This product is amazing!",
    "sentiment": "Positive",
    "confidence": 0.92,
    "class_scores": {
      "Positive": 0.92,
      "Negative": 0.05,
      "Neutral": 0.03
    }
  }
}
```

## Using with Your Code

### Django/FastAPI Integration

```python
import requests

# Get sentiment for a review
response = requests.post('http://localhost:5000/predict', 
    json={'text': 'Great product!'})
result = response.json()['data']

print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.0%}")
```

### Process CSV Files

```python
from api_client import SentimentAPIClient

client = SentimentAPIClient()

# Process file and save results
df = client.predict_csv(
    'input_reviews.csv',
    text_column='review_text',
    output_filepath='output_predictions.csv'
)
```

## Troubleshooting

### "Connection refused" on localhost:5000
```bash
# Make sure Flask is running in a terminal:
python flask_app.py
```

### "Port already in use"
```bash
# Kill the process using port 5000:
kill -9 $(lsof -t -i:5000)

# Or run on different port:
python -c "from flask_app import app; app.run(port=5001)"
```

### Model loading errors
```bash
# Verify files exist:
ls ml_results_full_tfidf_svm_rbf/svm_model.pkl
ls representations_full/tfidf_vectorizer.pkl

# Ensure you're in the section4 directory
cd section4/
```

## Files Reference

| File | Uses | When to Use |
|------|------|-------------|
| `flask_app.py` | REST API | Building web services, integrations |
| `streamlit_app.py` | Web UI | Interactive analysis, dashboards |
| `api_client.py` | Python wrapper | Testing, scripting, integrations |
| `model_loader.py` | Model interface | Direct Python access to model |
| `web_utils.py` | Utilities | Text processing, analytics |
| `run_benchmark.py` | Testing | Performance evaluation |

## What Model Is Being Used?

```
🤖 SVM with RBF Kernel + TF-IDF
├── Training Data: 200 reviews
├── Test Accuracy: 80%
├── F1-Score: 0.7705
└── Features: 303 TF-IDF features
```

**Why this model?**
- ✅ Best performance among all tested combinations
- ✅ Consistent across all preprocessing methods
- ✅ Fast predictions (<50ms)
- ✅ Production-ready

## Next Steps

1. **Try Flask API**: `python flask_app.py` → http://localhost:5000
2. **Try Dashboard**: `streamlit run streamlit_app.py` → http://localhost:8501
3. **Integrate with your app**: Use `api_client.py` or direct REST calls
4. **Test performance**: `python run_benchmark.py`
5. **Deploy**: See deployment section in full README

## Support

- 📖 Full documentation: `README_WEB_APPS.md`
- 🧪 Test API: Use curl, Postman, or `api_client.py`
- 📊 Check performance: `run_benchmark.py`
- 🔍 Debug: Check console output and Flask logs

---

**You're all set!** 🎉 Start with:
```bash
python flask_app.py
```

Then in another terminal:
```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501` to see your dashboard! 🚀
