"""
Performance Comparison Tool
Compare Flask vs Streamlit performance, validate results, and benchmark the API
"""

import time
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

try:
    import requests
    from streamlit.web import cli as stcli
except ImportError:
    requests = None

from model_loader import get_model


# ──────────────────────────────────────────────────────────────────────────────
# ── PERFORMANCE BENCHMARK ─────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

class PerformanceBenchmark:
    """Benchmark and test model performance"""
    
    def __init__(self):
        self.model = get_model()
        self.results = []
    
    def benchmark_single_prediction(self, text: str, iterations: int = 100) -> Dict[str, float]:
        """
        Benchmark single text prediction
        
        Args:
            text: Text to predict
            iterations: Number of times to run
            
        Returns:
            Dictionary with timing statistics
        """
        print(f"\n[*] Benchmarking single prediction ({iterations} iterations)...")
        times = []
        
        for _ in range(iterations):
            start = time.time()
            self.model.predict(text)
            end = time.time()
            times.append((end - start) * 1000)  # Convert to ms
        
        return {
            'iterations': iterations,
            'avg_time_ms': np.mean(times),
            'min_time_ms': np.min(times),
            'max_time_ms': np.max(times),
            'std_time_ms': np.std(times),
            'total_time_ms': np.sum(times)
        }
    
    def benchmark_batch_prediction(self, texts: List[str], iterations: int = 10) -> Dict[str, float]:
        """
        Benchmark batch prediction
        
        Args:
            texts: List of texts to predict
            iterations: Number of times to run
            
        Returns:
            Dictionary with timing statistics
        """
        print(f"\n[*] Benchmarking batch prediction ({iterations} iterations)...")
        times = []
        
        for _ in range(iterations):
            start = time.time()
            self.model.predict_batch(texts)
            end = time.time()
            times.append((end - start) * 1000)  # Convert to ms
        
        batch_size = len(texts)
        
        return {
            'batch_size': batch_size,
            'iterations': iterations,
            'avg_time_ms': np.mean(times),
            'avg_time_per_text_ms': np.mean(times) / batch_size,
            'min_time_ms': np.min(times),
            'max_time_ms': np.max(times),
            'std_time_ms': np.std(times),
            'throughput_texts_per_sec': (batch_size / (np.mean(times) / 1000))
        }
    
    def test_consistency(self, text: str, iterations: int = 50) -> Dict[str, any]:
        """Test if predictions are consistent across multiple runs"""
        print(f"\n[*] Testing prediction consistency ({iterations} runs)...")
        
        results = []
        for _ in range(iterations):
            result = self.model.predict(text)
            results.append(result['sentiment'])
        
        # Check if all predictions are the same
        unique_predictions = set(results)
        consistency = len(unique_predictions) == 1
        
        return {
            'text': text,
            'iterations': iterations,
            'consistent': consistency,
            'unique_predictions': len(unique_predictions),
            'predictions': results,
            'mode_sentiment': pd.Series(results).mode()[0]
        }
    
    def test_edge_cases(self) -> Dict[str, any]:
        """Test model behavior on edge cases"""
        print("\n[*] Testing edge cases...")
        
        edge_cases = {
            'empty_like': '',
            'single_char': 'a',
            'numbers_only': '12345',
            'special_chars': '!@#$%^&*()',
            'very_long': 'a' * 5000,
            'unicode': '你好世界😀🎉',
            'urls': 'http://example.com and https://test.org',
            'emojis_only': '😀😁😂😃😄😅😆😇',
            'repeated_words': 'good good good good good',
            'mixed_case': 'ThIs Is A mIxEd CaSe TeXt'
        }
        
        results = {}
        for case_name, text in edge_cases.items():
            try:
                if text:  # Skip empty string
                    result = self.model.predict(text)
                    results[case_name] = {
                        'success': True,
                        'sentiment': result['sentiment'],
                        'confidence': result['confidence']
                    }
                else:
                    results[case_name] = {'success': False, 'error': 'Empty text'}
            except Exception as e:
                results[case_name] = {'success': False, 'error': str(e)}
        
        return results


# ──────────────────────────────────────────────────────────────────────────────
# ── API CLIENT BENCHMARK ──────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

class APIBenchmark:
    """Benchmark Flask API performance"""
    
    def __init__(self, api_url: str = "http://localhost:5000"):
        self.api_url = api_url
        self.is_available = self._check_api()
    
    def _check_api(self) -> bool:
        """Check if API is available"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def benchmark_api_single(self, text: str, iterations: int = 50) -> Dict[str, float]:
        """Benchmark API single prediction"""
        if not self.is_available:
            print("⚠ API not available at", self.api_url)
            return {}
        
        print(f"\n[*] Benchmarking API single prediction ({iterations} iterations)...")
        times = []
        
        for _ in range(iterations):
            start = time.time()
            response = requests.post(
                f"{self.api_url}/predict",
                json={'text': text},
                timeout=10
            )
            response.raise_for_status()
            end = time.time()
            times.append((end - start) * 1000)
        
        return {
            'iterations': iterations,
            'avg_time_ms': np.mean(times),
            'min_time_ms': np.min(times),
            'max_time_ms': np.max(times),
            'std_time_ms': np.std(times),
        }
    
    def benchmark_api_batch(self, texts: List[str], iterations: int = 10) -> Dict[str, float]:
        """Benchmark API batch prediction"""
        if not self.is_available:
            print("⚠ API not available at", self.api_url)
            return {}
        
        print(f"\n[*] Benchmarking API batch prediction ({iterations} iterations)...")
        times = []
        
        for _ in range(iterations):
            start = time.time()
            response = requests.post(
                f"{self.api_url}/predict/batch",
                json={'texts': texts},
                timeout=30
            )
            response.raise_for_status()
            end = time.time()
            times.append((end - start) * 1000)
        
        return {
            'batch_size': len(texts),
            'iterations': iterations,
            'avg_time_ms': np.mean(times),
            'avg_time_per_text_ms': np.mean(times) / len(texts),
            'throughput_texts_per_sec': (len(texts) / (np.mean(times) / 1000))
        }


# ──────────────────────────────────────────────────────────────────────────────
# ── COMPARISON & REPORTING ────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

class PerformanceReport:
    """Generate performance comparison reports"""
    
    @staticmethod
    def print_benchmark_results(result: Dict[str, float], title: str = "Benchmark Results"):
        """Print formatted benchmark results"""
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}")
        
        for key, value in result.items():
            if isinstance(value, float):
                if 'ms' in key:
                    print(f"  {key:.<40} {value:>12.2f} ms")
                elif 'sec' in key:
                    print(f"  {key:.<40} {value:>12.2f} texts/sec")
                else:
                    print(f"  {key:.<40} {value:>12.2f}")
            else:
                print(f"  {key:.<40} {value:>12}")
        
        print(f"{'='*70}\n")
    
    @staticmethod
    def save_report(results: Dict[str, any], filepath: str = "performance_report.json"):
        """Save report to JSON file"""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'results': results
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ Report saved to {filepath}")


# ──────────────────────────────────────────────────────────────────────────────
# ── MAIN BENCHMARK EXECUTION ──────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*70)
    print("Sentiment Analysis - Performance Benchmark Suite")
    print("="*70)
    
    # Sample texts for benchmarking
    test_texts = [
        "This product is absolutely amazing! I love it!",
        "Terrible experience, would not recommend",
        "It's okay, nothing special",
        "Best purchase ever, highly recommend!",
        "Worst product I've ever bought"
    ]
    
    benchmark = PerformanceBenchmark()
    
    # Benchmark 1: Single prediction
    result_single = benchmark.benchmark_single_prediction(test_texts[0], iterations=100)
    PerformanceReport.print_benchmark_results(result_single, "Single Prediction Benchmark")
    
    # Benchmark 2: Batch prediction
    result_batch = benchmark.benchmark_batch_prediction(test_texts, iterations=20)
    PerformanceReport.print_benchmark_results(result_batch, "Batch Prediction Benchmark")
    
    # Benchmark 3: Consistency test
    consistency_result = benchmark.test_consistency(test_texts[0], iterations=50)
    print(f"\n{'='*70}")
    print("  Prediction Consistency Test")
    print(f"{'='*70}")
    print(f"  Text: {consistency_result['text']}")
    print(f"  Consistent: {'✓ Yes' if consistency_result['consistent'] else '✗ No'}")
    print(f"  Mode Sentiment: {consistency_result['mode_sentiment']}")
    print(f"  Unique Predictions: {consistency_result['unique_predictions']}")
    print(f"{'='*70}\n")
    
    # Benchmark 4: Edge cases
    edge_cases_result = benchmark.test_edge_cases()
    print(f"\n{'='*70}")
    print("  Edge Cases Test Results")
    print(f"{'='*70}")
    for case_name, result in edge_cases_result.items():
        status = "✓" if result.get('success') else "✗"
        if result.get('success'):
            print(f"  {status} {case_name:.<40} {result['sentiment']} ({result['confidence']:.2%})")
        else:
            print(f"  {status} {case_name:.<40} {result.get('error', 'Error')}")
    print(f"{'='*70}\n")
    
    # Try to benchmark API if available
    print("[*] Checking Flask API availability...")
    api_benchmark = APIBenchmark()
    
    if api_benchmark.is_available:
        print("✓ Flask API is running!\n")
        
        api_result_single = api_benchmark.benchmark_api_single(test_texts[0], iterations=30)
        PerformanceReport.print_benchmark_results(api_result_single, "API Single Prediction")
        
        api_result_batch = api_benchmark.benchmark_api_batch(test_texts, iterations=10)
        PerformanceReport.print_benchmark_results(api_result_batch, "API Batch Prediction")
    else:
        print("⚠ Flask API is not running")
        print("  Start it with: python flask_app.py\n")
    
    # Save report
    all_results = {
        'direct_single': result_single,
        'direct_batch': result_batch,
        'consistency': consistency_result,
        'edge_cases': edge_cases_result
    }
    
    if api_benchmark.is_available:
        all_results['api_single'] = api_result_single
        all_results['api_batch'] = api_result_batch
    
    PerformanceReport.save_report(all_results, "benchmark_report.json")
    
    print("="*70)
    print("Benchmark Complete!")
    print("="*70 + "\n")
