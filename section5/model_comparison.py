"""
Model Comparison - Before & After Optimization
Compare performance before and after hyperparameter optimization
"""

import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns


class ModelComparison:
    """Compare models before and after optimization"""
    
    def __init__(self, section4_dir: str = "../section4"):
        """
        Initialize comparison
        
        Args:
            section4_dir: Path to section4 where baseline models are stored
        """
        self.section4_dir = Path(section4_dir)
        self.baseline_metrics = {}
        self.optimized_metrics = {}
        self.comparison_results = []
    
    def load_baseline_model(self) -> Dict[str, Any]:
        """Load the baseline SVM RBF model"""
        print("Loading baseline model (SVM RBF - Full)...")
        
        model_dir = self.section4_dir / "ml_results_full_tfidf_svm_rbf"
        
        with open(model_dir / "svm_model.pkl", 'rb') as f:
            model = pickle.load(f)
        
        with open(model_dir / "label_encoder.pkl", 'rb') as f:
            label_encoder = pickle.load(f)
        
        with open(model_dir / "training_config.json", 'r') as f:
            config = json.load(f)
        
        return {
            'model': model,
            'label_encoder': label_encoder,
            'config': config
        }
    
    def load_optimized_model(self, filepath: str = "optimized_svm_model.pkl") -> Any:
        """Load the optimized model"""
        print(f"Loading optimized model from {filepath}...")
        
        try:
            import joblib
            model = joblib.load(filepath)
            return model
        except FileNotFoundError:
            print(f"✗ Optimized model not found at {filepath}")
            return None
    
    def load_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load test data"""
        features_path = self.section4_dir / "representations_full" / "representations_combined.csv"
        
        features_df = pd.read_csv(features_path)
        
        # Extract only tfidf_* columns (skip text/metadata columns)
        tfidf_cols = [c for c in features_df.columns if c.startswith('tfidf_')]
        X = features_df[tfidf_cols].values
        y = features_df['final_label'].values
        
        # Use last 20% as test set (simulating holdout test)
        test_size = int(len(X) * 0.2)
        X_test = X[-test_size:]
        y_test = y[-test_size:]
        
        return X_test, y_test
    
    def evaluate_model(self, model, X_test: np.ndarray, 
                      y_test: np.ndarray, label_encoder=None) -> Dict[str, float]:
        """Evaluate model on test set"""
        # Encode labels if they're strings
        if label_encoder is not None and len(y_test) > 0 and isinstance(y_test[0], str):
            y_test_encoded = label_encoder.transform(y_test)
        else:
            y_test_encoded = y_test
        
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test_encoded, y_pred),
            'precision': precision_score(y_test_encoded, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test_encoded, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
        }
        
        return metrics
    
    def compare_models(self) -> pd.DataFrame:
        """Compare baseline vs optimized models"""
        print("\n" + "="*80)
        print("MODEL COMPARISON: BASELINE vs OPTIMIZED")
        print("="*80 + "\n")
        
        # Load data
        X_test, y_test = self.load_test_data()
        
        # Baseline model
        baseline_obj = self.load_baseline_model()
        baseline_model = baseline_obj['model']
        baseline_label_encoder = baseline_obj['label_encoder']
        baseline_metrics = self.evaluate_model(baseline_model, X_test, y_test, baseline_label_encoder)
        baseline_metrics['model'] = 'Baseline SVM RBF'
        
        print("✓ Baseline SVM RBF")
        print(f"  Accuracy:  {baseline_metrics['accuracy']:.4f}")
        print(f"  Precision: {baseline_metrics['precision']:.4f}")
        print(f"  Recall:    {baseline_metrics['recall']:.4f}")
        print(f"  F1-Score:  {baseline_metrics['f1_score']:.4f}\n")
        
        # Optimized model
        optimized_model = self.load_optimized_model()
        
        if optimized_model is None:
            print("✗ Could not load optimized model. Skipping comparison.")
            return None
        
        optimized_metrics = self.evaluate_model(optimized_model, X_test, y_test, baseline_label_encoder)
        optimized_metrics['model'] = 'Optimized SVM RBF'
        
        print("✓ Optimized SVM RBF")
        print(f"  Accuracy:  {optimized_metrics['accuracy']:.4f}")
        print(f"  Precision: {optimized_metrics['precision']:.4f}")
        print(f"  Recall:    {optimized_metrics['recall']:.4f}")
        print(f"  F1-Score:  {optimized_metrics['f1_score']:.4f}\n")
        
        # Calculate improvements
        improvements = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            baseline_val = baseline_metrics[metric]
            optimized_val = optimized_metrics[metric]
            improvement = optimized_val - baseline_val
            improvement_pct = (improvement / baseline_val * 100) if baseline_val != 0 else 0
            
            improvements[metric] = {
                'baseline': baseline_val,
                'optimized': optimized_val,
                'improvement': improvement,
                'improvement_pct': improvement_pct
            }
        
        # Display improvements
        print("="*80)
        print("IMPROVEMENTS")
        print("="*80 + "\n")
        
        for metric, vals in improvements.items():
            print(f"{metric.upper()}")
            print(f"  Baseline:       {vals['baseline']:.4f}")
            print(f"  Optimized:      {vals['optimized']:.4f}")
            print(f"  Improvement:    {vals['improvement']:+.4f} ({vals['improvement_pct']:+.2f}%)\n")
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame([baseline_metrics, optimized_metrics])
        self.comparison_results = comparison_df
        self.improvements = improvements
        
        return comparison_df
    
    def check_random_baseline(self, num_classes: int = 3) -> float:
        """
        Check random chance baseline
        
        For balanced classes, random chance = 1/num_classes
        """
        random_f1 = 1.0 / num_classes
        improvement_target = random_f1 * 1.2  # 20% above random
        
        return random_f1, improvement_target
    
    def print_optimization_check(self):
        """Check if optimization meets requirements (20% above random)"""
        if not self.comparison_results.any(axis=None):
            print("No comparison results. Run compare_models first.")
            return
        
        random_f1, improvement_target = self.check_random_baseline()
        optimized_f1 = self.comparison_results.loc[1, 'f1_score']
        
        print("\n" + "="*80)
        print("OPTIMIZATION REQUIREMENT CHECK")
        print("="*80 + "\n")
        
        print(f"Random Chance F1-Score (3 classes):     {random_f1:.4f}")
        print(f"Target (20% above random):              {improvement_target:.4f}")
        print(f"Optimized Model F1-Score:               {optimized_f1:.4f}")
        print()
        
        if optimized_f1 >= improvement_target:
            print(f"✓ PASSED: Model exceeds 20% improvement threshold!")
            print(f"  Performance: {(optimized_f1 / random_f1 - 1) * 100:.2f}% above random chance")
        else:
            print(f"✗ FAILED: Model does not meet 20% improvement threshold")
            print(f"  Current: {(optimized_f1 / random_f1 - 1) * 100:.2f}% above random chance")
    
    def save_comparison_report(self, filepath: str = "model_comparison_report.json"):
        """Save comparison report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'baseline': {
                'model': 'SVM RBF - Full Preprocessing',
                'metrics': self.comparison_results.iloc[0].to_dict()
            },
            'optimized': {
                'model': 'Optimized SVM RBF',
                'metrics': self.comparison_results.iloc[1].to_dict() if len(self.comparison_results) > 1 else {}
            },
            'improvements': {
                k: {
                    'baseline': v['baseline'],
                    'optimized': v['optimized'],
                    'improvement': v['improvement'],
                    'improvement_pct': v['improvement_pct']
                }
                for k, v in self.improvements.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ Report saved to {filepath}")
    
    def plot_comparison(self, filepath: str = "model_comparison.png"):
        """Create visualization comparing baseline and optimized models"""
        if not self.comparison_results.any(axis=None):
            print("No results to plot.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Model Performance: Baseline vs Optimized', fontsize=14, fontweight='bold')
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        baseline_values = self.comparison_results.iloc[0][metrics].values
        optimized_values = self.comparison_results.iloc[1][metrics].values if len(self.comparison_results) > 1 else baseline_values
        
        x = np.arange(len(metrics))
        width = 0.35
        
        # Bar chart
        axes[0].bar(x - width/2, baseline_values, width, label='Baseline', color='skyblue')
        if len(self.comparison_results) > 1:
            axes[0].bar(x + width/2, optimized_values, width, label='Optimized', color='lightgreen')
        
        axes[0].set_ylabel('Score')
        axes[0].set_title('Metrics Comparison')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([m.replace('_', '\n') for m in metrics], fontsize=9)
        axes[0].legend()
        axes[0].set_ylim([0, 1])
        axes[0].grid(axis='y', alpha=0.3)
        
        # Improvement percentages
        if len(self.comparison_results) > 1:
            improvements = [
                (self.comparison_results.iloc[1][m] - self.comparison_results.iloc[0][m]) / 
                self.comparison_results.iloc[0][m] * 100
                for m in metrics
            ]
            
            colors = ['green' if x > 0 else 'red' for x in improvements]
            axes[1].bar(metrics, improvements, color=colors, alpha=0.7)
            axes[1].set_ylabel('Improvement (%)')
            axes[1].set_title('Improvement Over Baseline')
            axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            axes[1].grid(axis='y', alpha=0.3)
            axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison plot saved to {filepath}")
        plt.close()
    
    def run_comparison(self):
        """Run complete comparison"""
        print("\n" + "="*80)
        print("STARTING MODEL COMPARISON")
        print("="*80)
        
        # Compare models
        comparison_df = self.compare_models()
        
        if comparison_df is not None:
            # Check optimization requirements
            self.print_optimization_check()
            
            # Save and visualize
            self.save_comparison_report()
            self.plot_comparison()
        
        print("\n✓ Comparison complete!")


if __name__ == "__main__":
    comparator = ModelComparison()
    comparator.run_comparison()
