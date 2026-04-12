"""
Error Analysis & Failed Case Investigation
Identify and analyze incorrectly classified samples to understand model failures
"""

import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


class ErrorAnalyzer:
    """Analyze model errors and failed cases"""
    
    def __init__(self, section4_dir: str = "../section4"):
        """
        Initialize error analyzer
        
        Args:
            section4_dir: Path to section4 where models are stored
        """
        self.section4_dir = Path(section4_dir)
        self.failed_cases = {}
        self.analysis_results = {}
    
    def load_best_model(self) -> Dict[str, Any]:
        """
        Load the best performing model (SVM RBF - Full preprocessing)
        
        Returns:
            Dictionary with model, label encoder, features, and labels
        """
        print("Loading best model (SVM RBF - Full Preprocessing)...")
        
        model_dir = self.section4_dir / "ml_results_full_tfidf_svm_rbf"
        features_dir = self.section4_dir / "representations_full"
        
        # Load model
        with open(model_dir / "svm_model.pkl", 'rb') as f:
            model = pickle.load(f)
        
        with open(model_dir / "label_encoder.pkl", 'rb') as f:
            label_encoder = pickle.load(f)
        
        # Load features and labels from same file
        features_df = pd.read_csv(features_dir / "representations_combined.csv")
        
        # Extract only tfidf_* columns (skip text/metadata columns)
        tfidf_cols = [c for c in features_df.columns if c.startswith('tfidf_')]
        X = features_df[tfidf_cols].values
        
        return {
            'model': model,
            'label_encoder': label_encoder,
            'X': X,
            'y': features_df['final_label'].values,
            'texts': features_df['content'].values,
            'label_encoder_classes': label_encoder.classes_
        }
    
    def identify_failed_cases(self, model_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identify all incorrectly classified samples
        
        Returns:
            List of failed case dictionaries
        """
        print("\nIdentifying failed cases...")
        
        model = model_data['model']
        X = model_data['X']
        y = model_data['y']  # String labels
        texts = model_data['texts']
        label_encoder = model_data['label_encoder']
        
        # Encode labels (model was trained on encoded integers)
        y_encoded = label_encoder.transform(y)
        
        # Get predictions (returns encoded integers)
        y_pred = model.predict(X)
        
        # Get decision scores
        decision_scores = model.decision_function(X)
        
        # Find misclassified samples
        misclassified_mask = y_pred != y_encoded
        failed_indices = np.where(misclassified_mask)[0]
        
        failed_cases = []
        for idx in failed_indices:
            true_label = label_encoder.inverse_transform([y_encoded[idx]])[0]
            pred_label = label_encoder.inverse_transform([y_pred[idx]])[0]
            
            # Calculate confidence
            scores = decision_scores[idx]
            max_score = scores.max()
            min_score = scores.min()
            confidence = 1 / (1 + np.exp(-max_score))  # Sigmoid for confidence
            
            failed_cases.append({
                'index': int(idx),
                'text': texts[idx],
                'true_label': true_label,
                'predicted_label': pred_label,
                'confidence': float(confidence),
                'decision_scores': {
                    label: float(score)
                    for label, score in zip(label_encoder.classes_, scores)
                }
            })
        
        self.failed_cases = failed_cases
        print(f"✓ Found {len(failed_cases)} misclassified cases out of {len(y)} total")
        
        return failed_cases
    
    def analyze_error_patterns(self, failed_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze patterns in failed cases
        
        Returns:
            Dictionary with analysis results
        """
        print("\nAnalyzing error patterns...")
        
        df = pd.DataFrame(failed_cases)
        
        analysis = {
            'total_errors': len(failed_cases),
            'error_rate': f"{len(failed_cases) / (len(self.model_data['y'] if hasattr(self, 'model_data') else []) or 1) * 100:.2f}%",
            
            # Confusion patterns
            'confusion_matrix': self._get_confusion_from_cases(failed_cases),
            
            # Why failures happen
            'error_categories': self._categorize_errors(failed_cases),
            
            # Confidence analysis
            'avg_confidence_failed': float(df['confidence'].mean()),
            'min_confidence_failed': float(df['confidence'].min()),
            'max_confidence_failed': float(df['confidence'].max()),
            
            # Text characteristics of failed cases
            'text_analysis': self._analyze_failed_texts(failed_cases)
        }
        
        self.analysis_results = analysis
        return analysis
    
    def _get_confusion_from_cases(self, failed_cases: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
        """Get confusion patterns from failed cases"""
        confusion = {}
        for case in failed_cases:
            true_label = case['true_label']
            pred_label = case['predicted_label']
            
            if true_label not in confusion:
                confusion[true_label] = {}
            
            if pred_label not in confusion[true_label]:
                confusion[true_label][pred_label] = 0
            
            confusion[true_label][pred_label] += 1
        
        return confusion
    
    def _categorize_errors(self, failed_cases: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Categorize errors by type"""
        categories = {
            'neutral_confused_with_positive': [],
            'neutral_confused_with_negative': [],
            'positive_confused_with_negative': [],
            'negative_confused_with_positive': [],
            'other': []
        }
        
        for case in failed_cases:
            true_label = case['true_label'].lower()
            pred_label = case['predicted_label'].lower()
            
            if true_label == 'neutral' and pred_label == 'positive':
                categories['neutral_confused_with_positive'].append(case['text'])
            elif true_label == 'neutral' and pred_label == 'negative':
                categories['neutral_confused_with_negative'].append(case['text'])
            elif true_label == 'positive' and pred_label == 'negative':
                categories['positive_confused_with_negative'].append(case['text'])
            elif true_label == 'negative' and pred_label == 'positive':
                categories['negative_confused_with_positive'].append(case['text'])
            else:
                categories['other'].append(case['text'])
        
        # Keep only non-empty categories
        return {k: v for k, v in categories.items() if v}
    
    def _analyze_failed_texts(self, failed_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze characteristics of failed text samples"""
        texts = []
        for case in failed_cases:
            text = case['text']
            # Ensure text is a string
            if isinstance(text, str):
                texts.append(text)
            else:
                texts.append(str(text) if text is not None else "")
        
        if not texts:
            return {
                'avg_words': 0,
                'min_words': 0,
                'max_words': 0,
                'avg_chars': 0,
                'sample_texts': []
            }
        
        lengths = [len(text.split()) if text else 0 for text in texts]
        
        return {
            'avg_words': float(np.mean(lengths)) if lengths else 0,
            'min_words': int(np.min(lengths)) if lengths else 0,
            'max_words': int(np.max(lengths)) if lengths else 0,
            'avg_chars': float(np.mean([len(t) for t in texts])) if texts else 0,
            'sample_texts': texts[:5]  # First 5 failed samples
        }
    
    def print_analysis_report(self):
        """Print detailed error analysis report"""
        if not self.analysis_results:
            print("No analysis results. Run analyze_error_patterns first.")
            return
        
        try:
            analysis = self.analysis_results
            
            print("\n" + "="*80)
            print("ERROR ANALYSIS REPORT")
            print("="*80 + "\n")
            
            print(f"Total Errors: {analysis['total_errors']}")
            print(f"Error Rate: {analysis['error_rate']}")
            print()
            
            print("CONFUSION PATTERNS:")
            print("-" * 80)
            confusion_matrix = analysis.get('confusion_matrix', {})
            if isinstance(confusion_matrix, dict):
                for true_label, predictions in confusion_matrix.items():
                    if isinstance(predictions, dict):
                        print(f"\n{true_label.upper()} samples misclassified as:")
                        for pred_label, count in predictions.items():
                            print(f"  → {pred_label}: {count} cases")
            
            print("\n" + "-" * 80)
            print("ERROR CATEGORIES & THEORIES:")
            print("-" * 80)
            
            for category, samples in analysis.get('error_categories', {}).items():
                if isinstance(samples, (list, set)):
                    samples_list = list(samples) if isinstance(samples, set) else samples
                    print(f"\n{category.upper()} ({len(samples_list)} cases):")
                    print(f"  Theory: Model struggles when {category.replace('_', ' ')}")
                    print(f"  Examples:")
                    for sample in samples_list[:3]:
                        if isinstance(sample, str):
                            print(f"    - \"{sample[:60]}...\"")
            
            print("\n" + "-" * 80)
            print("TEXT CHARACTERISTICS OF FAILURES:")
            print("-" * 80)
            text_analysis = analysis.get('text_analysis', {})
            print(f"  Average words: {text_analysis.get('avg_words', 0):.1f}")
            print(f"  Min words: {text_analysis.get('min_words', 0)}")
            print(f"  Max words: {text_analysis.get('max_words', 0)}")
            print(f"  Average characters: {text_analysis.get('avg_chars', 0):.1f}")
            
            print("\n" + "-" * 80)
            print("CONFIDENCE ANALYSIS:")
            print("-" * 80)
            print(f"  Avg confidence on failed cases: {analysis.get('avg_confidence_failed', 0):.4f}")
            print(f"  Min confidence: {analysis.get('min_confidence_failed', 0):.4f}")
            print(f"  Max confidence: {analysis.get('max_confidence_failed', 0):.4f}")
            
            print("\n" + "="*80)
        except Exception as e:
            print(f"Warning: Could not print full report: {str(e)}")
    
    def save_report(self, filepath: str = "error_analysis_report.json"):
        """Save error analysis to JSON"""
        try:
            output = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'failed_cases': self.failed_cases,
                'analysis_summary': {
                    k: v for k, v in self.analysis_results.items()
                    if k not in ['confusion_matrix']  # Confusion matrix handling
                },
                'confusion_patterns': self.analysis_results.get('confusion_matrix', {})
            }
            
            # Convert sets to lists for JSON serialization
            if 'error_categories' in output['analysis_summary']:
                categories = output['analysis_summary']['error_categories']
                output['analysis_summary']['error_categories'] = {
                    k: list(v) if isinstance(v, (list, set)) else str(v)
                    for k, v in categories.items()
                }
            
            with open(filepath, 'w') as f:
                json.dump(output, f, indent=2, default=str)
            
            print(f"✓ Report saved to {filepath}")
        except Exception as e:
            print(f"Warning: Could not save report: {str(e)}")
    
    def plot_error_analysis(self, filepath: str = "error_analysis_visualization.png"):
        """Create visualization of error patterns"""
        if not self.failed_cases:
            print("No failed cases to visualize.")
            return
        
        try:
            # Create a simplified DataFrame without problematic columns
            simple_cases = []
            for case in self.failed_cases:
                simple_cases.append({
                    'true_label': case['true_label'],
                    'predicted_label': case['predicted_label'],
                    'confidence': float(case.get('confidence', 0))
                })
            
            df = pd.DataFrame(simple_cases)
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Error Analysis Visualization', fontsize=16, fontweight='bold')
            
            # 1. Confusion heatmap
            confusion_data = {}
            for case in self.failed_cases:
                key = f"{case['true_label']} → {case['predicted_label']}"
                confusion_data[key] = confusion_data.get(key, 0) + 1
            
            errors_df = pd.DataFrame(list(confusion_data.items()), 
                                     columns=['Error Type', 'Count'])
            errors_df = errors_df.sort_values('Count', ascending=True)
            
            axes[0, 0].barh(errors_df['Error Type'], errors_df['Count'], color='coral')
            axes[0, 0].set_xlabel('Number of Cases')
            axes[0, 0].set_title('Confusion Patterns')
            
            # 2. Confidence distribution
            if len(df) > 0:
                axes[0, 1].hist(df['confidence'], bins=20, color='skyblue', edgecolor='black')
                axes[0, 1].set_xlabel('Confidence Score')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].set_title('Confidence Distribution of Failed Cases')
                axes[0, 1].axvline(df['confidence'].mean(), color='red', linestyle='--', 
                                  label=f"Mean: {df['confidence'].mean():.3f}")
                axes[0, 1].legend()
            
            # 3. True label distribution
            true_label_counts = df['true_label'].value_counts()
            axes[1, 0].bar(true_label_counts.index, true_label_counts.values, color='lightgreen')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].set_title('Failed Cases by True Label')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # 4. Predicted label distribution
            pred_label_counts = df['predicted_label'].value_counts()
            axes[1, 1].bar(pred_label_counts.index, pred_label_counts.values, color='lightcoral')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title('Failed Cases by Predicted Label')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Visualization saved to {filepath}")
            plt.close()
        except Exception as e:
            print(f"Warning: Could not create visualization: {str(e)}")
    
    def run_full_analysis(self):
        """Run complete error analysis"""
        print("\n" + "="*80)
        print("STARTING COMPREHENSIVE ERROR ANALYSIS")
        print("="*80)
        
        # Load model
        model_data = self.load_best_model()
        self.model_data = model_data
        
        # Identify failures
        failed_cases = self.identify_failed_cases(model_data)
        
        # Analyze patterns
        analysis = self.analyze_error_patterns(failed_cases)
        
        # Print report
        self.print_analysis_report()
        
        # Save and visualize
        self.save_report()
        self.plot_error_analysis()
        
        print("\n✓ Error analysis complete!")


if __name__ == "__main__":
    analyzer = ErrorAnalyzer()
    analyzer.run_full_analysis()
