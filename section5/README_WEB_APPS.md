# 🎯 Sentiment Analysis Web Applications

Complete web solution for sentiment prediction using the best trained ML model (SVM + TF-IDF).

## 📋 Files Overview

| File | Purpose |
|------|---------|
| **model_loader.py** | Load and manage the SVM model, vectorizer, and label encoder |
| **flask_app.py** | REST API server with Flask (single & batch predictions) |
| **streamlit_app.py** | Interactive web dashboard with Streamlit (UI for analysis) |
| **api_client.py** | Python client for testing/integrating the Flask API |
| **web_utils.py** | Shared utilities (text processing, analytics, data export) |
| **run_benchmark.py** | Performance benchmarking and comparison tool |
| **requirements.txt** | All Python dependencies |

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Flask API (REST Server)

```bash
python flask_app.py
```

**Output:**
```
======================================================================
Sentiment Analysis API (Flask)
======================================================================
SVM with RBF Kernel + TF-IDF
Accuracy: 80% | F1-Score: 0.7705

Starting server on http://localhost:5000
API Documentation: http://localhost:5000/
======================================================================
```

**Test the API:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is amazing!"}'
```

### 3. Run Streamlit Dashboard (in separate terminal)

```bash
streamlit run streamlit_app.py
```

**Output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

## 📚 Detailed Usage Guide

### Flask REST API

#### **Endpoints**

##### 1. Health Check
```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2026-04-12T10:30:00.000000",
  "service": "Sentiment Analysis API (Flask + SVM + TF-IDF)"
}
```

##### 2. Model Information
```bash
GET /model/info
```

Response:
```json
{
  "model_type": "SVM with RBF Kernel + TF-IDF",
  "classes": ["Negative", "Neutral", "Positive"],
  "n_features": 303,
  "accuracy": 0.80,
  "f1_score": 0.7705,
  "precision": 0.7458,
  "recall": 0.80,
  "train_samples": 160,
  "test_samples": 40
}
```

##### 3. Single Text Prediction
```bash
POST /predict
Content-Type: application/json

{
  "text": "This product is amazing!"
}
```

Response:
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
    },
    "model_info": {
      "type": "SVM with RBF Kernel",
      "features": "tfidf",
      "accuracy": 0.80,
      "f1_score": 0.7705
    }
  },
  "timestamp": "2026-04-12T10:30:00.000000"
}
```

##### 4. Batch Text Prediction
```bash
POST /predict/batch
Content-Type: application/json

{
  "texts": [
    "This is great!",
    "I don't like it",
    "It's okay"
  ]
}
```

Response:
```json
{
  "status": "success",
  "data": [
    {"text": "This is great!", "sentiment": "Positive", "confidence": 0.88, ...},
    {"text": "I don't like it", "sentiment": "Negative", "confidence": 0.95, ...},
    {"text": "It's okay", "sentiment": "Neutral", "confidence": 0.72, ...}
  ],
  "processed": 3,
  "timestamp": "2026-04-12T10:30:00.000000"
}
```

### Python API Client

```python
from api_client import SentimentAPIClient

# Initialize client
client = SentimentAPIClient("http://localhost:5000")

# Single prediction
result = client.predict("This product is amazing!")
print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']*100:.2f}%")

# Batch prediction
results = client.predict_batch([
    "This is great!",
    "I don't like it",
    "It's okay"
])

# Process CSV file
df = client.predict_csv('reviews.csv', text_column='review', 
                        output_filepath='predictions.csv')
```

### Streamlit Dashboard

**Features:**
- 🔍 **Single Text Analysis** - Real-time sentiment prediction with visualization
- 📦 **Batch Analysis** - Upload CSV or paste multiple texts
- 📊 **Model Information** - View architecture, performance metrics, class distribution
- 📈 **Visual Analytics** - Confidence scores, sentiment distribution charts
- 📥 **Export Results** - Download predictions as CSV

**Usage:**
1. Select mode from left sidebar
2. Enter text or upload file
3. Click "Predict" or "Analyze All"
4. View results and download if needed

## 🔬 Performance Benchmarking

Run the benchmark suite to test performance:

```bash
python run_benchmark.py
```

**Tests:**
- Single prediction latency (100 iterations)
- Batch prediction throughput (20 iterations)
- Prediction consistency (50 runs)
- Edge case handling (special inputs)
- API performance (if running)

**Output:**
```
======================================================================
  Single Prediction Benchmark
======================================================================
  iterations............................... 100
  avg_time_ms.......................... 25.34 ms
  min_time_ms.......................... 22.10 ms
  max_time_ms.......................... 45.20 ms
  std_time_ms.......................... 5.12 ms
======================================================================
```

## 📊 Model Selection Rationale

### ✅ Why SVM with RBF Kernel?

| Criterion | Score | Notes |
|-----------|-------|-------|
| **Accuracy** | 80% | Highest among all tested models |
| **F1-Score** | 0.7705 | Best balance of precision and recall |
| **Consistency** | Very High | Performs consistently across all preprocessing methods |
| **Speed** | Fast | Prediction latency < 50ms |
| **Memory** | Efficient | Small model size (~1MB) |

### 📈 Performance Comparison

```
┌─────────────────────┬──────────┬──────────┬──────────┐
│ Model               │ Accuracy │ F1-Score │ Speed    │
├─────────────────────┼──────────┼──────────┼──────────┤
│ SVM (RBF)           │ 0.80     │ 0.7705   │ ⚡ Fast  │
│ SVM (Linear)        │ 0.80     │ 0.7686   │ ⚡ Fast  │
│ Logistic Regression │ 0.80     │ 0.7686   │ ⚡ Fast  │
└─────────────────────┴──────────┴──────────┴──────────┘
```

## 🛠️ Development & Customization

### Using the Model Loader

```python
from model_loader import get_model

# Load model (singleton pattern)
model = get_model()

# Get model info
info = model.get_info()

# Single prediction
result = model.predict("Your text here")

# Batch prediction
results = model.predict_batch(["Text 1", "Text 2"])
```

### Custom Text Processing

```python
from web_utils import TextProcessor

# Clean text
cleaned = TextProcessor.clean_text(raw_text)

# Validate text
is_valid, error = TextProcessor.validate_text(text)

# Get text statistics
stats = TextProcessor.get_text_stats(text)
```

### Export Results

```python
from web_utils import DataExporter

# Export to CSV
csv_string = DataExporter.to_csv(results, filepath='output.csv')

# Export to JSON
json_string = DataExporter.to_json(results, filepath='output.json')

# Convert to DataFrame
df = DataExporter.to_dataframe(results)
```

### Analytics

```python
from web_utils import Analytics

# Get sentiment summary
summary = Analytics.sentiment_summary(results)
print(f"Positive: {summary['positive_percentage']:.1f}%")
print(f"Negative: {summary['negative_percentage']:.1f}%")

# Confidence distribution
dist = Analytics.confidence_distribution(results)
```

## 🔧 Configuration

Edit `web_utils.py` to customize:

```python
class Config:
    MODEL_DIR = "ml_results_full_tfidf_svm_rbf"  # Model directory
    VECTORIZER_DIR = "representations_full"      # Vectorizer directory
    
    API_HOST = "0.0.0.0"                         # API host
    API_PORT = 5000                              # API port
    
    MIN_TEXT_LENGTH = 3                          # Minimum text length
    MAX_TEXT_LENGTH = 10000                      # Maximum text length
    
    MAX_BATCH_SIZE = 100                         # Max texts per batch
```

## 📋 Example Use Cases

### 1. Analyze Product Reviews

```bash
# Flask API
curl -X POST http://localhost:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great product!", "Waste of money", "Average"]}'
```

### 2. Integration with Existing System

```python
from api_client import SentimentAPIClient

client = SentimentAPIClient()
sentiment = client.predict("User review text")

# Store in database
db.save_sentiment(text=..., sentiment=sentiment['sentiment'])
```

### 3. Batch Processing

```bash
# Process large CSV file
python api_client.py
# Then use client.predict_csv() in Python
```

### 4. Real-time Dashboard

```bash
# Open Streamlit dashboard
streamlit run streamlit_app.py
# Access at http://localhost:8501
```

## ⚠️ Troubleshooting

### Flask API Won't Start
```bash
# Check if port 5000 is in use
lsof -i :5000

# Kill process if necessary
kill -9 <PID>

# Run on different port
python -c "from flask_app import app; app.run(port=5001)"
```

### Streamlit Not Connecting to API
```bash
# Ensure Flask is running first
python flask_app.py  # Terminal 1

# Then start Streamlit
streamlit run streamlit_app.py  # Terminal 2
```

### Model Loading Error
```bash
# Verify model files exist
ls ml_results_full_tfidf_svm_rbf/
ls representations_full/

# Check file permissions
chmod 644 ml_results_full_tfidf_svm_rbf/*.pkl
```

### Out of Memory
```bash
# Set batch size limit in web_utils.py
MAX_BATCH_SIZE = 50  # Down from 100

# Or limit predictions in client
results = client.predict_batch(texts[:50])
```

## 📈 Performance Optimization Tips

1. **Batch Processing**: Use batch predictions for multiple texts
2. **Caching**: Results are automatically cached in Flask
3. **Connection Pooling**: API client uses persistent sessions
4. **Async Processing**: Consider using Celery for large datasets

## 🤝 Integration Examples

### With Django

```python
import requests

def get_sentiment(text):
    response = requests.post(
        'http://localhost:5000/predict',
        json={'text': text}
    )
    return response.json()['data']['sentiment']
```

### With FastAPI

```python
from fastapi import FastAPI
import httpx

app = FastAPI()

@app.post("/analyze")
async def analyze(text: str):
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            'http://localhost:5000/predict',
            json={'text': text}
        )
    return resp.json()
```

## 📞 Support & Issues

- Check logs: `Flask app prints errors to console`
- Enable debug: Set `debug=True` in `flask_app.py`
- Test API: Use `curl` or Postman
- Run benchmarks: `python run_benchmark.py`

## 📦 Deployment

### Docker

```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "flask_app.py"]
```

### Heroku

```bash
heroku create sentiment-analyzer
git push heroku main
heroku open
```

### AWS Lambda

Use API Gateway + Lambda for REST endpoints

## 📝 License & Attribution

- **Model**: Trained on custom dataset with SVM + TF-IDF
- **Libraries**: scikit-learn, Flask, Streamlit, pandas, numpy
- **Dataset**: 200 labeled reviews, 80% positive, 20% split

## ✨ Key Statistics

- **Model Accuracy**: 80%
- **F1-Score**: 0.7705
- **Precision**: 0.7458
- **Recall**: 0.80%
- **Average Prediction Time**: <50ms
- **Supported Classes**: Positive, Negative, Neutral
- **Features**: 303 TF-IDF features

---

**Last Updated**: April 12, 2026
**Version**: 1.0
**Status**: Production Ready ✅
