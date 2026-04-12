"""
Flask-based REST API for Sentiment Analysis
Provides endpoints for single and batch sentiment predictions
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from datetime import datetime
import traceback

from model_loader import get_model

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model on startup
try:
    model = get_model()
    logger.info("✓ Model loaded successfully")
    app.model = model
except Exception as e:
    logger.error(f"✗ Failed to load model: {e}")
    app.model = None


# ──────────────────────────────────────────────────────────────────────────────
# ── HEALTH CHECK ──────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy" if app.model else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Sentiment Analysis API (Flask + SVM + TF-IDF)"
    }), 200 if app.model else 503


# ──────────────────────────────────────────────────────────────────────────────
# ── MODEL INFO ────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information and performance metrics"""
    if not app.model:
        return jsonify({"error": "Model not loaded"}), 503
    
    info = app.model.get_info()
    info["timestamp"] = datetime.now().isoformat()
    return jsonify(info), 200


# ──────────────────────────────────────────────────────────────────────────────
# ── SINGLE PREDICTION ─────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict sentiment for a single text
    
    Request body:
    {
        "text": "This product is amazing!"
    }
    
    Response:
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
            "model_info": {...}
        },
        "timestamp": "2026-04-12T..."
    }
    """
    try:
        if not app.model:
            return jsonify({"error": "Model not loaded"}), 503
        
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field in request"}), 400
        
        text = data['text'].strip()
        
        if not text:
            return jsonify({"error": "Text cannot be empty"}), 400
        
        if len(text) > 10000:
            return jsonify({"error": "Text too long (max 10000 characters)"}), 400
        
        # Make prediction
        result = app.model.predict(text)
        
        return jsonify({
            "status": "success",
            "data": result,
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500


# ──────────────────────────────────────────────────────────────────────────────
# ── BATCH PREDICTION ──────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Predict sentiment for multiple texts
    
    Request body:
    {
        "texts": [
            "This is great!",
            "I don't like it",
            "It's okay"
        ]
    }
    
    Response:
    {
        "status": "success",
        "data": [
            {"text": "...", "sentiment": "...", "confidence": ...},
            ...
        ],
        "processed": 3,
        "timestamp": "2026-04-12T..."
    }
    """
    try:
        if not app.model:
            return jsonify({"error": "Model not loaded"}), 503
        
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({"error": "Missing 'texts' field in request"}), 400
        
        texts = data['texts']
        
        if not isinstance(texts, list):
            return jsonify({"error": "'texts' must be a list"}), 400
        
        if len(texts) > 100:
            return jsonify({"error": "Too many texts (max 100)"}), 400
        
        if not texts:
            return jsonify({"error": "Empty texts list"}), 400
        
        # Filter empty texts
        texts = [t.strip() for t in texts if isinstance(t, str) and t.strip()]
        
        # Make predictions
        results = app.model.predict_batch(texts)
        
        return jsonify({
            "status": "success",
            "data": results,
            "processed": len(results),
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500


# ──────────────────────────────────────────────────────────────────────────────
# ── ROOT ENDPOINT ─────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

@app.route('/', methods=['GET'])
def root():
    """API documentation"""
    return jsonify({
        "service": "Sentiment Analysis API (Flask)",
        "version": "1.0",
        "model": "SVM with RBF Kernel + TF-IDF",
        "endpoints": {
            "GET /health": "Health check",
            "GET /model/info": "Get model information",
            "POST /predict": "Predict sentiment for single text",
            "POST /predict/batch": "Predict sentiment for multiple texts"
        },
        "example_request": {
            "endpoint": "POST /predict",
            "body": {"text": "This product is amazing!"}
        }
    }), 200


# ──────────────────────────────────────────────────────────────────────────────
# ── ERROR HANDLERS ────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "status": "error",
        "error": "Endpoint not found",
        "timestamp": datetime.now().isoformat()
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "status": "error",
        "error": "Method not allowed",
        "timestamp": datetime.now().isoformat()
    }), 405


if __name__ == '__main__':
    # Run with: python flask_app.py
    # Or: flask --app flask_app run --debug
    print("\n" + "=" * 70)
    print("Sentiment Analysis API (Flask)")
    print("=" * 70)
    print("SVM with RBF Kernel + TF-IDF")
    print("Accuracy: 80% | F1-Score: 0.7705")
    print("\nStarting server on http://localhost:5000")
    print("API Documentation: http://localhost:5000/")
    print("=" * 70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
