"""
Model Validation & Overfitting Detection Framework
Detects overfitting, validates sentiment predictions, catches failing test cases
"""

import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

# Known failing test cases and expected sentiments
FAILING_TEST_CASES = {
    "not good": "Negative",
    "not bad": "Positive", 
    "not great": "Negative",
    "pretty bad": "Negative",
    "really good": "Positive",
    "very bad": "Negative",
    "extremely good": "Positive",
    "terribly wrong": "Negative",
    "absolutely amazing": "Positive",
    "so bad": "Negative",
    "kinda good": "Positive",
    "somewhat okay": "Neutral",
    "just okay": "Neutral",
    "nothing special": "Neutral",
    "not okay": "Negative"
}

class OverfittingDetector:
    """Detects ovitting in models"""
    
    def __init__(self, model, vectorizer, label_encoder, X_train, X_test, y_train, y_test):
        self.model = model
        self.vectorizer = vectorizer
        self.label_encoder = label_encoder
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    
    def check_train_test_gap(self):
        """Check if there's a big gap between train and test accuracy (sign of overfitting)"""
        train_pred = self.model.predict(self.X_train)
        test_pred = self.model.predict(self.X_test)
        
        train_acc = accuracy_score(self.y_train, train_pred)
        test_acc = accuracy_score(self.y_test, test_pred)
        
        gap = train_acc - test_acc
        
        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'gap': gap,
            'is_overfitting': gap > 0.1,  # 10% gap is suspicious
            'severity': 'SEVERE' if gap > 0.2 else 'MODERATE' if gap > 0.1 else 'MINIMAL'
        }
    
    def cross_validation_check(self, cv_folds=5):
        """Use cross-validation to detect overfitting"""
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Get scores for each fold
        cv_scores = []
        fold_diffs = []
        
        for train_idx, val_idx in skf.split(self.X_train, self.y_train):
            X_fold_train = self.X_train[train_idx]
            X_fold_val = self.X_train[val_idx]
            y_fold_train = self.y_train[train_idx]
            y_fold_val = self.y_train[val_idx]
            
            # Train on fold
            self.model.fit(X_fold_train, y_fold_train)
            
            # Evaluate
            fold_train_acc = accuracy_score(y_fold_train, self.model.predict(X_fold_train))
            fold_val_acc = accuracy_score(y_fold_val, self.model.predict(X_fold_val))
            
            cv_scores.append(fold_val_acc)
            fold_diffs.append(fold_train_acc - fold_val_acc)
        
        return {
            'cv_scores': cv_scores,
            'mean_cv_score': np.mean(cv_scores),
            'std_cv_score': np.std(cv_scores),
            'mean_fold_diff': np.mean(fold_diffs),
            'is_overfitting': np.mean(fold_diffs) > 0.1,
            'score_variance': np.std(cv_scores)
        }
    
    def check_class_imbalance(self):
        """Detect if model struggles with minority classes (can indicate poor generalization)"""
        train_dist = np.bincount(self.y_train)
        test_dist = np.bincount(self.y_test)
        
        # Get per-class metrics
        test_pred = self.model.predict(self.X_test)
        per_class_f1 = f1_score(self.y_test, test_pred, average=None)
        
        return {
            'train_class_distribution': train_dist.tolist(),
            'test_class_distribution': test_dist.tolist(),
            'per_class_f1': per_class_f1.tolist(),
            'f1_variance': float(np.var(per_class_f1)),
            'has_class_imbalance': np.var(per_class_f1) > 0.05
        }
    
    def detect_overfitting(self):
        """Comprehensive overfitting detection"""
        print("\n" + "="*80)
        print("OVERFITTING DETECTION REPORT")
        print("="*80)
        
        # Check 1: Train-test gap
        gap_check = self.check_train_test_gap()
        print(f"\n[1] Train-Test Accuracy Gap:")
        print(f"    Train Accuracy: {gap_check['train_accuracy']:.4f}")
        print(f"    Test Accuracy:  {gap_check['test_accuracy']:.4f}")
        print(f"    Gap: {gap_check['gap']:.4f} ({gap_check['severity']})")
        print(f"    Status: {'⚠ OVERFITTING DETECTED' if gap_check['is_overfitting'] else '✓ OK'}")
        
        # Check 2: Class imbalance
        class_check = self.check_class_imbalance()
        print(f"\n[2] Class Balance Check:")
        print(f"    Train distribution: {class_check['train_class_distribution']}")
        print(f"    Test distribution: {class_check['test_class_distribution']}")
        print(f"    Per-class F1 scores: {[f'{x:.4f}' for x in class_check['per_class_f1']]}")
        print(f"    F1 Variance: {class_check['f1_variance']:.4f}")
        print(f"    Status: {'⚠ IMBALANCED' if class_check['has_class_imbalance'] else '✓ BALANCED'}")
        
        return {
            'gap_check': gap_check,
            'class_check': class_check
        }

class SentimentValidator:
    """Validates model sentiment predictions"""
    
    def __init__(self, model, vectorizer, label_encoder):
        self.model = model
        self.vectorizer = vectorizer
        self.label_encoder = label_encoder
    
    def test_sentiment(self, text):
        """Test a single text prediction"""
        try:
            # Vectorize
            text_vec = self.vectorizer.transform([text])
            # Convert to dense if sparse (fix for sparse matrix issue)
            if hasattr(text_vec, 'toarray'):
                text_vec = text_vec.toarray()
            # Predict
            pred_encoded = self.model.predict(text_vec)[0]
            pred_label = self.label_encoder.inverse_transform([pred_encoded])[0]
            
            # Get confidence
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(text_vec)[0]
                confidence = float(np.max(proba))
            elif hasattr(self.model, 'decision_function'):
                scores = self.model.decision_function(text_vec)[0]
                confidence = float(np.max(np.abs(scores)))
            else:
                confidence = 0.0
            
            return pred_label, confidence
        except Exception as e:
            return f"ERROR: {str(e)}", 0.0
    
    def validate_test_cases(self):
        """Validate on known failing cases"""
        print("\n" + "="*80)
        print("SENTIMENT TEST VALIDATION")
        print("="*80)
        print(f"Testing {len(FAILING_TEST_CASES)} known cases...")
        
        results = []
        passed = 0
        failed = 0
        
        for text, expected in FAILING_TEST_CASES.items():
            predicted, confidence = self.test_sentiment(text)
            is_correct = predicted == expected
            status = "✓" if is_correct else "✗"
            
            results.append({
                'text': text,
                'expected': expected,
                'predicted': predicted,
                'correct': is_correct,
                'confidence': confidence
            })
            
            if is_correct:
                passed += 1
            else:
                failed += 1
            
            print(f"{status} '{text}'")
            print(f"    Expected: {expected:10s} | Predicted: {predicted:10s} | Conf: {confidence:.3f}")
        
        accuracy = passed / (passed + failed) if (passed + failed) > 0 else 0
        
        print(f"\n{'='*80}")
        print(f"Sentiment Validation Results:")
        print(f"  Passed: {passed}/{passed + failed} ({accuracy*100:.1f}%)")
        print(f"  Failed: {failed}/{passed + failed}")
        print(f"  Status: {'✓ PASS' if accuracy >= 0.8 else '⚠ FAIL - Model not ready'}")
        print(f"{'='*80}")
        
        return results, accuracy

class ComprehensiveModelValidator:
    """Full model validation pipeline"""
    
    def __init__(self, model_dir, section4_dir="../section4"):
        self.model_dir = Path(model_dir)
        self.section4_dir = Path(section4_dir)
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
    
    def load_model(self):
        """Load trained model and components"""
        try:
            with open(self.model_dir / "svm_model.pkl", 'rb') as f:
                self.model = pickle.load(f)
            
            with open(self.model_dir / "label_encoder.pkl", 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            # Try to load vectorizer
            try:
                import joblib
                self.vectorizer = joblib.load(self.model_dir / "tfidf_vectorizer.pkl")
            except:
                print("Vectorizer not found")
                self.vectorizer = None
            
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def run_full_validation(self):
        """Run comprehensive validation"""
        if not self.load_model():
            return
        
        # 1. Sentiment validation
        validator = SentimentValidator(self.model, self.vectorizer, self.label_encoder)
        test_results, accuracy = validator.validate_test_cases()
        
        # Save results
        output_file = self.model_dir / "sentiment_validation.json"
        with open(output_file, 'w') as f:
            json.dump({
                'test_cases': test_results,
                'overall_accuracy': accuracy,
                'passed': sum(1 for r in test_results if r['correct']),
                'total': len(test_results)
            }, f, indent=2)
        
        print(f"\n✓ Validation results saved to: {output_file}")

# Test the improved preprocessing
if __name__ == "__main__":
    # Example usage
    model_dir = Path("/media/abdo/Games/social_data_analysis/section4/ml_results_full_tfidf_svm_rbf")
    
    validator = ComprehensiveModelValidator(model_dir)
    validator.run_full_validation()
