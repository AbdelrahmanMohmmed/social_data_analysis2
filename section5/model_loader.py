"""
Model Loader Module
Handles loading the trained SVM model, vectorizer, and label encoder for sentiment prediction.
"""

import pickle
import json
from pathlib import Path
import numpy as np
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


class SentimentModelLoader:
    """Load and manage sentiment analysis model components"""
    
    def __init__(self, model_dir: str = None, vectorizer_dir: str = None):
        """
        Initialize model loader
        
        Args:
            model_dir: Directory containing SVM model and label encoder (defaults to ../section4/ml_results_full_tfidf_svm_rbf)
            vectorizer_dir: Directory containing TF-IDF vectorizer (defaults to ../section4/representations_full)
        """
        if model_dir is None:
            model_dir = str(Path(__file__).parent.parent / "section4" / "ml_results_full_tfidf_svm_rbf")
        if vectorizer_dir is None:
            vectorizer_dir = str(Path(__file__).parent.parent / "section4" / "representations_full")
        
        self.model_dir = Path(model_dir)
        self.vectorizer_dir = Path(vectorizer_dir)
        
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.config = None
        
        self._load_components()
    
    def _load_components(self):
        """Load all model components"""
        try:
            # Load SVM model
            model_path = self.model_dir / "svm_model.pkl"
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"✓ Loaded SVM model from {model_path}")
            
            # Load label encoder
            encoder_path = self.model_dir / "label_encoder.pkl"
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            print(f"✓ Loaded label encoder from {encoder_path}")
            
            # Load TF-IDF vectorizer
            vectorizer_path = self.vectorizer_dir / "tfidf_vectorizer.pkl"
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            print(f"✓ Loaded TF-IDF vectorizer from {vectorizer_path}")
            
            # Load training config
            config_path = self.model_dir / "training_config.json"
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            print(f"✓ Loaded training config from {config_path}")
            
        except FileNotFoundError as e:
            print(f"✗ Error loading model components: {e}")
            raise
    
    def predict(self, text: str) -> Dict[str, any]:
        """
        Predict sentiment for a single text
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary with prediction results
        """
        if not all([self.model, self.vectorizer, self.label_encoder]):
            raise RuntimeError("Model components not loaded properly")
        
        # Vectorize text and convert sparse to dense
        text_features = self.vectorizer.transform([text]).toarray()
        
        # Get prediction and confidence
        prediction = self.model.predict(text_features)[0]
        
        # Get decision function scores (confidence-like)
        decision_scores = self.model.decision_function(text_features)[0]
        
        # Decode label
        sentiment_label = self.label_encoder.inverse_transform([prediction])[0]
        
        # Calculate confidence (convert decision scores to probability-like scores)
        confidence_scores = self._softmax(decision_scores)
        
        # Create result dictionary
        result = {
            "text": text,
            "sentiment": sentiment_label,
            "confidence": float(confidence_scores.max()),
            "class_scores": {
                label: float(score)
                for label, score in zip(self.label_encoder.classes_, confidence_scores)
            },
            "model_info": {
                "type": "SVM with RBF Kernel",
                "features": self.config.get("feature_type", "tfidf"),
                "accuracy": 0.80,
                "f1_score": 0.7705
            }
        }
        
        return result
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, any]]:
        """
        Predict sentiment for multiple texts
        
        Args:
            texts: List of texts to classify
            
        Returns:
            List of prediction results
        """
        return [self.predict(text) for text in texts]
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Convert decision scores to probability distribution"""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    def get_info(self) -> Dict[str, any]:
        """Get model information"""
        return {
            "model_type": "SVM with RBF Kernel + TF-IDF",
            "classes": list(self.label_encoder.classes_),
            "n_features": self.config.get("n_features", "unknown"),
            "accuracy": 0.80,
            "f1_score": 0.7705,
            "precision": 0.7458,
            "recall": 0.80,
            "train_samples": self.config.get("train_samples", "unknown"),
            "test_samples": self.config.get("test_samples", "unknown"),
        }


# Global model instance (lazy loading)
_model_instance = None

def get_model() -> SentimentModelLoader:
    """Get or create model instance (singleton pattern)"""
    global _model_instance
    if _model_instance is None:
        _model_instance = SentimentModelLoader()
    return _model_instance
