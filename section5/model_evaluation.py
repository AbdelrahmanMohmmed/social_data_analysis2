"""
Comprehensive Model Evaluation & Benchmarking
Evaluate 18 models across multiple metrics to identify the best models/preprocessing schemes
"""

import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, auc, matthews_corrcoef
)
import matplotlib.pyplot as plt
import seaborn as sns


class ComprehensiveModelEvaluator:
    """Evaluate and benchmark all trained models"""
    
    def __init__(self, section4_dir: str = "../section4"):
        """
        Initialize evaluator
        
        Args:
            section4_dir: Path to section4 where models are stored
        """
        self.section4_dir = Path(section4_dir)
        self.models = {}
        self.results = {}
        self.best_models = {}
        
        # Define all model combinations
        self.model_configs = self._get_model_configs()
    
    def _get_model_configs(self) -> List[Dict[str, str]]:
        """Get all 12 model configurations"""
        datasets = ['minimal', 'stopwords', 'lemmatization', 'full']
        kernels = ['linear', 'rbf']
        
        configs = []
        
        # SVM models (8)
        for dataset in datasets:
            for kernel in kernels:
                configs.append({
                    'dataset': dataset,
                    'model_type': 'svm',
                    'kernel': kernel,
                    'name': f'SVM ({kernel.upper()}) - {dataset}'
                })
        
        # Logistic Regression models (4)
        for dataset in datasets:
            configs.append({
                'dataset': dataset,
                'model_type': 'logistic',
                'name': f'Logistic - {dataset}'
            })
        
        return configs
    
    def _get_model_dir(self, config: Dict[str, str]) -> Path:
        """Get model directory path for a configuration"""
        if config['model_type'] == 'svm':
            suffix = f"svm_{config['kernel']}"
        else:
            suffix = "logistic"
        
        dataset_prefix = config['dataset']
        dir_name = f"ml_results_{dataset_prefix}_tfidf_{suffix}"
        return self.section4_dir / dir_name
    
    def _get_features_dir(self, config: Dict[str, str]) -> Path:
        """Get features/representations directory"""
        dataset_map = {
            'minimal': 'representations_minimal',
            'stopwords': 'representations_stopwords',
            'lemmatization': 'representations_lemmatization',
            'full': 'representations_full'
        }
        return self.section4_dir / dataset_map[config['dataset']]
    
    def load_model(self, config: Dict[str, str]) -> Dict[str, Any]:
        """Load model and related components"""
        model_dir = self._get_model_dir(config)
        
        if not model_dir.exists():
            print(f"✗ Model directory not found: {model_dir}")
            return None
        
        try:
            # Load model
            model_path = model_dir / (
                "svm_model.pkl" if config['model_type'] == 'svm' 
                else "logistic_regression_model.pkl"
            )
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Load label encoder
            encoder_path = model_dir / "label_encoder.pkl"
            with open(encoder_path, 'rb') as f:
                label_encoder = pickle.load(f)
            
            # Load config
            config_path = model_dir / "training_config.json"
            with open(config_path, 'r') as f:
                train_config = json.load(f)
            
            return {
                'model': model,
                'label_encoder': label_encoder,
                'config': train_config,
                'model_dir': model_dir
            }
        except Exception as e:
            print(f"✗ Error loading model {config['name']}: {str(e)}")
            return None
    
    def load_features(self, config: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray]:
        """Load features and labels for a configuration"""
        features_dir = self._get_features_dir(config)
        
        try:
            # Load features and labels from same file
            features_path = features_dir / "representations_combined.csv"
            features_df = pd.read_csv(features_path)
            
            # Extract only tfidf_* columns (skip text/metadata columns)
            tfidf_cols = [c for c in features_df.columns if c.startswith('tfidf_')]
            X = features_df[tfidf_cols].values
            
            # Extract labels from same file
            y = features_df['final_label'].values
            
            return X, y
        except Exception as e:
            print(f"✗ Error loading features for {config['name']}: {str(e)}")
            return None, None
    
    def evaluate_model(self, model_obj: Dict[str, Any], X: np.ndarray, 
                      y: np.ndarray, config: Dict[str, str]) -> Dict[str, float]:
        """
        Evaluate a model on test data
        
        Args:
            model_obj: Loaded model object with model, label_encoder, config
            X: Features
            y: Labels (string format)
            config: Model configuration
            
        Returns:
            Dictionary with all evaluation metrics
        """
        if model_obj is None or X is None or y is None:
            return None
        
        model = model_obj['model']
        label_encoder = model_obj['label_encoder']
        
        # Encode string labels to integers (model expects encoded labels)
        y_encoded = label_encoder.transform(y)
        
        # Get predictions (returns encoded integers)
        y_pred = model.predict(X)
        
        # Calculate metrics
        metrics = {
            'model_name': config['name'],
            'dataset': config['dataset'],
            'model_type': config['model_type'],
            'accuracy': float(accuracy_score(y_encoded, y_pred)),
            'precision_weighted': float(precision_score(y_encoded, y_pred, average='weighted', zero_division=0)),
            'recall_weighted': float(recall_score(y_encoded, y_pred, average='weighted', zero_division=0)),
            'f1_weighted': float(f1_score(y_encoded, y_pred, average='weighted', zero_division=0)),
            'precision_macro': float(precision_score(y_encoded, y_pred, average='macro', zero_division=0)),
            'recall_macro': float(recall_score(y_encoded, y_pred, average='macro', zero_division=0)),
            'f1_macro': float(f1_score(y_encoded, y_pred, average='macro', zero_division=0)),
            'mcc': float(matthews_corrcoef(y_encoded, y_pred)),  # Matthews Correlation Coefficient
            'true_labels': y_encoded,
            'predictions': y_pred
        }
        
        # Try ROC-AUC (for multiclass)
        try:
            y_pred_proba = model.decision_function(X) if hasattr(model, 'decision_function') else None
            if y_pred_proba is not None:
                # For multiclass, use OvR approach
                from sklearn.preprocessing import label_binarize
                y_bin = label_binarize(y_encoded, classes=range(len(label_encoder.classes_)))
                
                if hasattr(model, 'decision_function'):
                    y_score = model.decision_function(X)
                    if y_score.ndim == 1:
                        y_score = y_score.reshape(-1, 1)
                    
                    try:
                        metrics['roc_auc_ovr'] = float(roc_auc_score(y_bin, y_score, multi_class='ovr', zero_division=0))
                    except:
                        metrics['roc_auc_ovr'] = None
        except:
            metrics['roc_auc_ovr'] = None
        
        return metrics
    
    def evaluate_all_models(self) -> pd.DataFrame:
        """Evaluate all 12 models and return results"""
        print("\n" + "="*80)
        print("EVALUATING ALL 12 MODELS")
        print("="*80 + "\n")
        
        all_results = []
        
        for i, config in enumerate(self.model_configs, 1):
            print(f"[{i}/12] Evaluating {config['name']}...", end=" ")
            
            # Load model
            model_obj = self.load_model(config)
            if model_obj is None:
                print("✗ Failed to load model")
                continue
            
            # Load features
            X, y = self.load_features(config)
            if X is None:
                print("✗ Failed to load features")
                continue
            
            # Evaluate
            metrics = self.evaluate_model(model_obj, X, y, config)
            if metrics is not None:
                all_results.append(metrics)
                print(f"✓ Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_weighted']:.4f}")
        
        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)
        self.results_df = results_df
        
        return results_df
    
    def get_confusion_matrix(self, metrics: Dict[str, Any]) -> np.ndarray:
        """Get confusion matrix from evaluation metrics"""
        return confusion_matrix(metrics['true_labels'], metrics['predictions'])
    
    def print_summary(self):
        """Print evaluation summary"""
        if not hasattr(self, 'results_df') or self.results_df.empty:
            print("No results to display. Run evaluate_all_models() first.")
            return
        
        df = self.results_df.copy()
        
        print("\n" + "="*100)
        print("MODEL EVALUATION SUMMARY")
        print("="*100 + "\n")
        
        # Sort by F1 score
        df_sorted = df.sort_values('f1_weighted', ascending=False)
        
        print(df_sorted[[
            'model_name', 'accuracy', 'precision_weighted', 
            'recall_weighted', 'f1_weighted', 'mcc'
        ]].to_string(index=False))
        
        print("\n" + "="*100)
        print("TOP 3 MODELS")
        print("="*100 + "\n")
        
        for idx, (_, row) in enumerate(df_sorted.head(3).iterrows(), 1):
            print(f"{idx}. {row['model_name']}")
            print(f"   Accuracy:  {row['accuracy']:.4f}")
            print(f"   Precision: {row['precision_weighted']:.4f}")
            print(f"   Recall:    {row['recall_weighted']:.4f}")
            print(f"   F1-Score:  {row['f1_weighted']:.4f}")
            print(f"   MCC:       {row['mcc']:.4f}")
            print()
    
    def save_results(self, filepath: str = "model_evaluation_results.json"):
        """Save detailed results to JSON"""
        if not hasattr(self, 'results_df'):
            print("No results to save.")
            return
        
        results_dict = self.results_df.drop(
            columns=['true_labels', 'predictions'], errors='ignore'
        ).to_dict('records')
        
        output = {
            'timestamp': datetime.now().isoformat(),
            'total_models': len(results_dict),
            'models': results_dict
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"✓ Results saved to {filepath}")
    
    def plot_comparison(self, filepath: str = "model_comparison.png"):
        """Plot model comparison charts"""
        if not hasattr(self, 'results_df') or self.results_df.empty:
            print("No results to plot.")
            return
        
        df = self.results_df.sort_values('f1_weighted', ascending=False)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Evaluation Comparison', fontsize=16, fontweight='bold')
        
        # Accuracy
        axes[0, 0].barh(df['model_name'], df['accuracy'], color='skyblue')
        axes[0, 0].set_xlabel('Accuracy')
        axes[0, 0].set_title('Accuracy Comparison')
        axes[0, 0].set_xlim([0, 1])
        
        # Precision
        axes[0, 1].barh(df['model_name'], df['precision_weighted'], color='lightgreen')
        axes[0, 1].set_xlabel('Precision')
        axes[0, 1].set_title('Precision Comparison')
        axes[0, 1].set_xlim([0, 1])
        
        # Recall
        axes[1, 0].barh(df['model_name'], df['recall_weighted'], color='lightcoral')
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_title('Recall Comparison')
        axes[1, 0].set_xlim([0, 1])
        
        # F1-Score
        axes[1, 1].barh(df['model_name'], df['f1_weighted'], color='orange')
        axes[1, 1].set_xlabel('F1-Score')
        axes[1, 1].set_title('F1-Score Comparison')
        axes[1, 1].set_xlim([0, 1])
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison chart saved to {filepath}")
        plt.close()


if __name__ == "__main__":
    evaluator = ComprehensiveModelEvaluator()
    
    # Evaluate all models
    results = evaluator.evaluate_all_models()
    
    # Print summary
    evaluator.print_summary()
    
    # Save results
    evaluator.save_results()
    
    # Plot comparison
    evaluator.plot_comparison()
    
    print("\n✓ Model evaluation complete!")
