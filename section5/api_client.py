"""
API Client for testing the Flask Sentiment Analysis API
Useful for integration testing and batch processing
"""

import requests
import json
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime
import time


class SentimentAPIClient:
    """Client for interacting with the Sentiment Analysis API"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        """
        Initialize API client
        
        Args:
            base_url: Base URL of the Flask API
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request and return JSON response"""
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"✗ Request failed: {str(e)}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """Check if API is healthy"""
        try:
            return self._make_request('GET', '/health')
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and performance metrics"""
        return self._make_request('GET', '/model/info')
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict sentiment for a single text
        
        Args:
            text: Text to classify
            
        Returns:
            Prediction result dictionary
        """
        response = self._make_request('POST', '/predict', 
                                     json={'text': text})
        
        if response.get('status') == 'success':
            return response['data']
        else:
            raise ValueError(f"Prediction failed: {response.get('error')}")
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict sentiment for multiple texts
        
        Args:
            texts: List of texts to classify
            
        Returns:
            List of prediction results
        """
        response = self._make_request('POST', '/predict/batch',
                                     json={'texts': texts})
        
        if response.get('status') == 'success':
            return response['data']
        else:
            raise ValueError(f"Batch prediction failed: {response.get('error')}")
    
    def predict_csv(self, filepath: str, text_column: str = 'text',
                    output_filepath: str = None) -> pd.DataFrame:
        """
        Predict sentiment for all texts in a CSV file
        
        Args:
            filepath: Path to input CSV file
            text_column: Name of column containing texts
            output_filepath: Optional path to save results
            
        Returns:
            DataFrame with predictions
        """
        df = pd.read_csv(filepath)
        
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in CSV")
        
        texts = df[text_column].astype(str).tolist()
        print(f"[*] Predicting sentiment for {len(texts)} texts...")
        
        # Process in batches to avoid overwhelming the API
        batch_size = 50
        all_results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            print(f"  Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}...")
            
            try:
                results = self.predict_batch(batch)
                all_results.extend(results)
                time.sleep(0.5)  # Small delay between batches
            except Exception as e:
                print(f"  ✗ Error in batch: {str(e)}")
        
        # Create results DataFrame
        results_df = pd.DataFrame([
            {
                'text': r['text'],
                'sentiment': r['sentiment'],
                'confidence': r['confidence'],
                **{f'score_{k}': v for k, v in r['class_scores'].items()}
            }
            for r in all_results
        ])
        
        if output_filepath:
            results_df.to_csv(output_filepath, index=False)
            print(f"[+] Results saved to {output_filepath}")
        
        return results_df


# ──────────────────────────────────────────────────────────────────────────────
# ── USAGE EXAMPLES ────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*70)
    print("Sentiment Analysis API Client - Usage Examples")
    print("="*70 + "\n")
    
    # Initialize client
    client = SentimentAPIClient()
    
    # Example 1: Health check
    print("[1] Health Check")
    health = client.health_check()
    if health.get('status') == 'healthy':
        print("✓ API is healthy\n")
    else:
        print("✗ API is not responding. Make sure Flask app is running:")
        print("   python flask_app.py\n")
        exit(1)
    
    # Example 2: Model info
    print("[2] Model Information")
    info = client.get_model_info()
    print(f"Model Type: {info['model_type']}")
    print(f"Classes: {', '.join(info['classes'])}")
    print(f"Accuracy: {info['accuracy']*100}%")
    print(f"F1-Score: {info['f1_score']}\n")
    
    # Example 3: Single prediction
    print("[3] Single Text Prediction")
    test_texts = [
        "This product is amazing! I love it so much!",
        "Terrible experience, would not recommend",
        "It's okay, nothing special"
    ]
    
    for text in test_texts:
        try:
            result = client.predict(text)
            print(f"Text: {text}")
            print(f"Sentiment: {result['sentiment']}")
            print(f"Confidence: {result['confidence']*100:.2f}%")
            print(f"Scores: {result['class_scores']}\n")
        except Exception as e:
            print(f"Error: {e}\n")
    
    # Example 4: Batch prediction
    print("[4] Batch Prediction")
    try:
        results = client.predict_batch(test_texts)
        print(f"Processed {len(results)} texts")
        sentiments = [r['sentiment'] for r in results]
        print(f"Sentiments: {sentiments}\n")
    except Exception as e:
        print(f"Error: {e}\n")
    
    # Example 5: CSV file processing (if file exists)
    print("[5] CSV File Processing")
    try:
        # This would process a CSV file if available
        print("Uncomment the line below to process a CSV file:")
        print("# results_df = client.predict_csv('input.csv', output_filepath='output.csv')")
    except Exception as e:
        print(f"Error: {e}\n")
    
    print("="*70)
    print("API Client Examples Complete")
    print("="*70 + "\n")
