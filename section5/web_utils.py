"""
Web Utilities
Shared functions and utilities for Flask and Streamlit applications
"""

import re
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any
from datetime import datetime
import json


# ──────────────────────────────────────────────────────────────────────────────
# ── TEXT PREPROCESSING ────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

class TextProcessor:
    """Utility class for text preprocessing and validation"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        return text.strip()
    
    @staticmethod
    def validate_text(text: str, min_length: int = 3, max_length: int = 10000) -> Tuple[bool, str]:
        """
        Validate text for prediction
        
        Args:
            text: Text to validate
            min_length: Minimum text length
            max_length: Maximum text length
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not text or not isinstance(text, str):
            return False, "Text must be a non-empty string"
        
        text = text.strip()
        
        if len(text) < min_length:
            return False, f"Text must be at least {min_length} characters"
        
        if len(text) > max_length:
            return False, f"Text must not exceed {max_length} characters"
        
        return True, ""
    
    @staticmethod
    def get_text_stats(text: str) -> Dict[str, Any]:
        """
        Get statistics about the text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with text statistics
        """
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        return {
            "characters": len(text),
            "words": len(words),
            "sentences": len([s for s in sentences if s.strip()]),
            "avg_word_length": np.mean([len(w) for w in words]) if words else 0,
            "unique_words": len(set(words))
        }


# ──────────────────────────────────────────────────────────────────────────────
# ── RESULT FORMATTING ─────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

class ResultFormatter:
    """Format and present prediction results"""
    
    SENTIMENT_EMOJI = {
        'Positive': '🟢',
        'Negative': '🔴',
        'Neutral': '🟡'
    }
    
    SENTIMENT_COLORS = {
        'Positive': '#27AE60',
        'Negative': '#E74C3C',
        'Neutral': '#95A5A6'
    }
    
    @staticmethod
    def format_single_result(result: Dict[str, Any]) -> Dict[str, Any]:
        """Format single prediction result for display"""
        return {
            'text': result['text'],
            'sentiment': result['sentiment'],
            'emoji': ResultFormatter.SENTIMENT_EMOJI.get(result['sentiment'], '⚪'),
            'confidence': f"{result['confidence']*100:.2f}%",
            'confidence_value': result['confidence'],
            'class_scores': {
                k: f"{v*100:.2f}%" for k, v in result['class_scores'].items()
            },
            'model': result['model_info']['type'],
            'color': ResultFormatter.SENTIMENT_COLORS.get(result['sentiment'], '#95A5A6')
        }
    
    @staticmethod
    def format_batch_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format batch prediction results with statistics"""
        if not results:
            return {}
        
        sentiments = [r['sentiment'] for r in results]
        confidences = [r['confidence'] for r in results]
        
        sentiment_counts = pd.Series(sentiments).value_counts()
        
        return {
            'total': len(results),
            'sentiment_distribution': sentiment_counts.to_dict(),
            'avg_confidence': np.mean(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'results': [ResultFormatter.format_single_result(r) for r in results]
        }


# ──────────────────────────────────────────────────────────────────────────────
# ── DATA EXPORT ───────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

class DataExporter:
    """Export prediction results in various formats"""
    
    @staticmethod
    def to_csv(results: List[Dict[str, Any]], filepath: str = None) -> str:
        """
        Export results to CSV format
        
        Args:
            results: List of prediction results
            filepath: Optional path to save CSV
            
        Returns:
            CSV string
        """
        df = pd.DataFrame([
            {
                'text': r['text'],
                'sentiment': r['sentiment'],
                'confidence': r['confidence'],
                **{f'score_{k}': v for k, v in r['class_scores'].items()}
            }
            for r in results
        ])
        
        csv_string = df.to_csv(index=False)
        
        if filepath:
            df.to_csv(filepath, index=False)
        
        return csv_string
    
    @staticmethod
    def to_json(results: List[Dict[str, Any]], filepath: str = None) -> str:
        """
        Export results to JSON format
        
        Args:
            results: List of prediction results
            filepath: Optional path to save JSON
            
        Returns:
            JSON string
        """
        output = {
            'timestamp': datetime.now().isoformat(),
            'total_results': len(results),
            'results': results
        }
        
        json_string = json.dumps(output, indent=2)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_string)
        
        return json_string
    
    @staticmethod
    def to_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert results to pandas DataFrame"""
        return pd.DataFrame([
            {
                'text': r['text'],
                'sentiment': r['sentiment'],
                'confidence': r['confidence'],
                **{f'score_{k}': v for k, v in r['class_scores'].items()}
            }
            for r in results
        ])


# ──────────────────────────────────────────────────────────────────────────────
# ── STATISTICS & ANALYTICS ───────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

class Analytics:
    """Generate analytics and insights from predictions"""
    
    @staticmethod
    def sentiment_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate sentiment summary statistics"""
        if not results:
            return {}
        
        sentiments = [r['sentiment'] for r in results]
        confidences = [r['confidence'] for r in results]
        
        sentiment_counts = pd.Series(sentiments).value_counts()
        
        return {
            'total_predictions': len(results),
            'positive_count': (sentiment_counts.get('Positive', 0)),
            'negative_count': (sentiment_counts.get('Negative', 0)),
            'neutral_count': (sentiment_counts.get('Neutral', 0)),
            'positive_percentage': (sentiment_counts.get('Positive', 0) / len(results) * 100),
            'negative_percentage': (sentiment_counts.get('Negative', 0) / len(results) * 100),
            'neutral_percentage': (sentiment_counts.get('Neutral', 0) / len(results) * 100),
            'avg_confidence': float(np.mean(confidences)),
            'min_confidence': float(np.min(confidences)),
            'max_confidence': float(np.max(confidences)),
            'std_confidence': float(np.std(confidences))
        }
    
    @staticmethod
    def confidence_distribution(results: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get distribution of predictions by confidence level"""
        if not results:
            return {}
        
        confidences = [r['confidence'] for r in results]
        
        bins = [0, 0.3, 0.5, 0.7, 0.9, 1.0]
        bin_labels = ['0-30%', '30-50%', '50-70%', '70-90%', '90-100%']
        
        distribution = pd.cut(confidences, bins=bins, labels=bin_labels).value_counts().sort_index()
        
        return distribution.to_dict()


# ──────────────────────────────────────────────────────────────────────────────
# ── EXAMPLE DATA ──────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

class ExampleData:
    """Sample texts for testing and demonstration"""
    
    POSITIVE_EXAMPLES = [
        "This product is absolutely amazing! I love it!",
        "Best purchase ever, highly recommend to everyone",
        "Outstanding quality and excellent customer service",
        "I'm extremely satisfied with my purchase",
        "Fantastic experience, will definitely buy again"
    ]
    
    NEGATIVE_EXAMPLES = [
        "Terrible product, complete waste of money",
        "Worst experience ever, do not recommend",
        "Poor quality and horrible customer support",
        "I'm very disappointed with this purchase",
        "Absolutely horrible, returning immediately"
    ]
    
    NEUTRAL_EXAMPLES = [
        "The product is okay, nothing special",
        "It's average, has pros and cons",
        "Not bad, but there are better options",
        "It does what it's supposed to do",
        "Acceptable but could be improved"
    ]
    
    @staticmethod
    def get_all_examples() -> List[str]:
        """Get all example texts"""
        return (ExampleData.POSITIVE_EXAMPLES + 
                ExampleData.NEGATIVE_EXAMPLES + 
                ExampleData.NEUTRAL_EXAMPLES)
    
    @staticmethod
    def get_summary() -> Dict[str, List[str]]:
        """Get organized example summary"""
        return {
            'positive': ExampleData.POSITIVE_EXAMPLES,
            'negative': ExampleData.NEGATIVE_EXAMPLES,
            'neutral': ExampleData.NEUTRAL_EXAMPLES
        }


# ──────────────────────────────────────────────────────────────────────────────
# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

class Config:
    """Application configuration"""
    
    # Model paths
    MODEL_DIR = "ml_results_full_tfidf_svm_rbf"
    VECTORIZER_DIR = "representations_full"
    
    # API configuration
    API_HOST = "0.0.0.0"
    API_PORT = 5000
    API_DEBUG = True
    
    # Streamlit configuration
    STREAMLIT_THEME = "auto"
    
    # Text processing
    MIN_TEXT_LENGTH = 3
    MAX_TEXT_LENGTH = 10000
    
    # Batch processing
    MAX_BATCH_SIZE = 100
    BATCH_PROCESS_DELAY = 0.5  # seconds
    
    # Model metadata
    MODEL_NAME = "SVM with RBF Kernel + TF-IDF"
    MODEL_ACCURACY = 0.80
    MODEL_F1_SCORE = 0.7705
    MODEL_PRECISION = 0.7458
    MODEL_RECALL = 0.80
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'model_dir': cls.MODEL_DIR,
            'vectorizer_dir': cls.VECTORIZER_DIR,
            'api_host': cls.API_HOST,
            'api_port': cls.API_PORT,
            'min_text_length': cls.MIN_TEXT_LENGTH,
            'max_text_length': cls.MAX_TEXT_LENGTH,
            'max_batch_size': cls.MAX_BATCH_SIZE,
            'model_accuracy': cls.MODEL_ACCURACY,
            'model_f1_score': cls.MODEL_F1_SCORE
        }
