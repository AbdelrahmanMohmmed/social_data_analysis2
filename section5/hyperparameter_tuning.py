"""
Hyperparameter Tuning & Optimization
Optimize models using Grid Search, Random Search, and Genetic Algorithm
"""

import pickle
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import f1_score, accuracy_score
import joblib


class HyperparameterOptimizer:
    """Optimize model hyperparameters using multiple strategies"""
    
    def __init__(self, section4_dir: str = "../section4"):
        """
        Initialize optimizer
        
        Args:
            section4_dir: Path to section4 where data is stored
        """
        self.section4_dir = Path(section4_dir)
        self.optimization_results = {}
        self.best_models = {}
    
    def load_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load features and labels"""
        features_path = self.section4_dir / "representations_full" / "representations_combined.csv"
        
        features_df = pd.read_csv(features_path)
        
        # Extract only tfidf_* columns (skip text/metadata columns)
        tfidf_cols = [c for c in features_df.columns if c.startswith('tfidf_')]
        X = features_df[tfidf_cols].values
        y_raw = features_df['final_label'].values
        
        # Encode labels to match trained model format
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw)
        
        return X, y
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                  test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train and test"""
        return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    # ──────────────────────────────────────────────────────────────────────────
    # ── GRID SEARCH ────────────────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────────────────
    
    def grid_search_svm_rbf(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Grid search for SVM with RBF kernel
        
        Tuning: C, gamma
        """
        print("\n[1/3] Grid Search: SVM RBF")
        print("-" * 60)
        
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'kernel': ['rbf']
        }
        
        svm = SVC(random_state=42)
        grid_search = GridSearchCV(
            svm, param_grid, cv=5, scoring='f1_weighted', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        result = {
            'method': 'Grid Search - SVM RBF',
            'best_params': grid_search.best_params_,
            'best_score': float(grid_search.best_score_),
            'best_model': grid_search.best_estimator_,
            'cv_results': grid_search.cv_results_
        }
        
        print(f"✓ Best params: {result['best_params']}")
        print(f"✓ Best CV F1-Score: {result['best_score']:.4f}")
        
        return result
    
    def grid_search_svm_linear(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Grid search for SVM with Linear kernel"""
        print("\n[1/3] Grid Search: SVM Linear")
        print("-" * 60)
        
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'kernel': ['linear']
        }
        
        svm = SVC(random_state=42)
        grid_search = GridSearchCV(
            svm, param_grid, cv=5, scoring='f1_weighted', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        result = {
            'method': 'Grid Search - SVM Linear',
            'best_params': grid_search.best_params_,
            'best_score': float(grid_search.best_score_),
            'best_model': grid_search.best_estimator_,
            'cv_results': grid_search.cv_results_
        }
        
        print(f"✓ Best params: {result['best_params']}")
        print(f"✓ Best CV F1-Score: {result['best_score']:.4f}")
        
        return result
    
    def grid_search_logistic(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Grid search for Logistic Regression"""
        print("\n[1/3] Grid Search: Logistic Regression")
        print("-" * 60)
        
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10],
            'solver': ['lbfgs', 'saga'],
            'max_iter': [200, 400]
        }
        
        lr = LogisticRegression(random_state=42, multi_class='multinomial')
        grid_search = GridSearchCV(
            lr, param_grid, cv=5, scoring='f1_weighted', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        result = {
            'method': 'Grid Search - Logistic Regression',
            'best_params': grid_search.best_params_,
            'best_score': float(grid_search.best_score_),
            'best_model': grid_search.best_estimator_,
            'cv_results': grid_search.cv_results_
        }
        
        print(f"✓ Best params: {result['best_params']}")
        print(f"✓ Best CV F1-Score: {result['best_score']:.4f}")
        
        return result
    
    # ──────────────────────────────────────────────────────────────────────────
    # ── RANDOM SEARCH ──────────────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────────────────
    
    def random_search_svm_rbf(self, X_train: np.ndarray, y_train: np.ndarray, 
                             n_iter: int = 20) -> Dict[str, Any]:
        """Random search for SVM RBF"""
        print("\n[2/3] Random Search: SVM RBF")
        print("-" * 60)
        
        param_dist = {
            'C': np.logspace(-2, 3, 100),
            'gamma': np.logspace(-4, 1, 100),
        }
        
        svm = SVC(kernel='rbf', random_state=42)
        random_search = RandomizedSearchCV(
            svm, param_dist, n_iter=n_iter, cv=5, 
            scoring='f1_weighted', n_jobs=-1, verbose=1, random_state=42
        )
        
        random_search.fit(X_train, y_train)
        
        result = {
            'method': 'Random Search - SVM RBF',
            'best_params': random_search.best_params_,
            'best_score': float(random_search.best_score_),
            'best_model': random_search.best_estimator_,
            'n_iter': n_iter
        }
        
        print(f"✓ Best params: {result['best_params']}")
        print(f"✓ Best CV F1-Score: {result['best_score']:.4f}")
        
        return result
    
    # ──────────────────────────────────────────────────────────────────────────
    # ── GENETIC ALGORITHM (using deap) ───────────────────────────────────────
    # ──────────────────────────────────────────────────────────────────────────
    
    def genetic_algorithm_optimization(self, X_train: np.ndarray, 
                                       y_train: np.ndarray) -> Dict[str, Any]:
        """
        Genetic Algorithm for hyperparameter optimization
        (if deap is installed)
        """
        print("\n[3/3] Genetic Algorithm: SVM RBF")
        print("-" * 60)
        
        try:
            from deap import base, creator, tools, algorithms
            
            # Define fitness and individual
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMax)
            
            toolbox = base.Toolbox()
            
            # Attributes
            toolbox.register("C_attr", np.random.uniform, -2, 3)  # log scale
            toolbox.register("gamma_attr", np.random.uniform, -4, 1)  # log scale
            
            # Individuals and populations
            toolbox.register("individual", tools.initCycle, creator.Individual,
                           (toolbox.C_attr, toolbox.gamma_attr), n=1)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            
            # Evaluation function
            def evaluate_params(individual):
                C = 10 ** individual[0]
                gamma = 10 ** individual[1]
                
                svm = SVC(C=C, gamma=gamma, kernel='rbf', random_state=42)
                svm.fit(X_train, y_train)
                score = f1_score(y_train, svm.predict(X_train), average='weighted')
                
                return (score,)
            
            toolbox.register("evaluate", evaluate_params)
            toolbox.register("mate", tools.cxBlend, alpha=0.5)
            toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.2)
            toolbox.register("select", tools.selTournament, tournsize=3)
            
            # Bounds
            toolbox.decorate("mate", tools.DeltaPenality(
                [-2, -4], [3, 1], [0, 0]
            ))
            toolbox.decorate("mutate", tools.DeltaPenality(
                [-2, -4], [3, 1], [0, 0]
            ))
            
            # Run GA
            pop = toolbox.population(n=20)
            hof = tools.HallOfFame(1)
            
            pop, logbook = algorithms.eaSimple(
                pop, toolbox, cxpb=0.7, mutpb=0.3, ngen=10, 
                halloffame=hof, verbose=1
            )
            
            best_ind = hof[0]
            best_params = {
                'C': 10 ** best_ind[0],
                'gamma': 10 ** best_ind[1],
                'kernel': 'rbf'
            }
            
            # Train final model
            svm = SVC(**best_params, random_state=42)
            svm.fit(X_train, y_train)
            best_score = f1_score(y_train, svm.predict(X_train), average='weighted')
            
            result = {
                'method': 'Genetic Algorithm - SVM RBF',
                'best_params': best_params,
                'best_score': float(best_score),
                'best_model': svm,
                'generations': 10
            }
            
            print(f"✓ Best params: {result['best_params']}")
            print(f"✓ Best F1-Score: {result['best_score']:.4f}")
            
            return result
            
        except ImportError:
            print("⚠ DEAP not installed. Skipping Genetic Algorithm.")
            return None
    
    # ──────────────────────────────────────────────────────────────────────────
    # ── EVALUATION & COMPARISON ────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────────────────
    
    def evaluate_optimized_models(self, models_results: List[Dict[str, Any]], 
                                  X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """Evaluate optimized models on test set"""
        print("\n" + "="*60)
        print("EVALUATING OPTIMIZED MODELS")
        print("="*60 + "\n")
        
        eval_results = []
        
        for result in models_results:
            if result is None:
                continue
            
            model = result['best_model']
            y_pred = model.predict(X_test)
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            eval_results.append({
                'method': result['method'],
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                'model': model
            })
            
            print(f"✓ {result['method']}")
            print(f"  Accuracy:  {eval_results[-1]['accuracy']:.4f}")
            print(f"  Precision: {eval_results[-1]['precision']:.4f}")
            print(f"  Recall:    {eval_results[-1]['recall']:.4f}")
            print(f"  F1-Score:  {eval_results[-1]['f1_score']:.4f}\n")
        
        return pd.DataFrame(eval_results)
    
    def save_best_model(self, model, filepath: str = "optimized_svm_model.pkl"):
        """Save best optimized model"""
        joblib.dump(model, filepath)
        print(f"✓ Model saved to {filepath}")
    
    def run_optimization(self):
        """Run complete hyperparameter optimization"""
        print("\n" + "="*80)
        print("HYPERPARAMETER OPTIMIZATION")
        print("="*80)
        
        # Load data
        print("\nLoading data...")
        X, y = self.load_training_data()
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        print(f"✓ Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
        
        results = []
        
        # Grid Search
        try:
            results.append(self.grid_search_svm_rbf(X_train, y_train))
            results.append(self.grid_search_svm_linear(X_train, y_train))
            results.append(self.grid_search_logistic(X_train, y_train))
        except Exception as e:
            print(f"Grid Search error: {e}")
        
        # Random Search
        try:
            results.append(self.random_search_svm_rbf(X_train, y_train, n_iter=20))
        except Exception as e:
            print(f"Random Search error: {e}")
        
        # Genetic Algorithm
        try:
            ga_result = self.genetic_algorithm_optimization(X_train, y_train)
            if ga_result:
                results.append(ga_result)
        except Exception as e:
            print(f"Genetic Algorithm error: {e}")
        
        # Evaluate
        eval_df = self.evaluate_optimized_models(results, X_test, y_test)
        
        # Display summary
        print("\n" + "="*80)
        print("OPTIMIZATION SUMMARY")
        print("="*80 + "\n")
        print(eval_df.to_string(index=False))
        
        # Save best model
        best_idx = eval_df['f1_score'].idxmax()
        best_model = eval_df.loc[best_idx, 'model']
        self.save_best_model(best_model)
        
        # Save results
        eval_df_save = eval_df.drop('model', axis=1)
        eval_df_save.to_csv('optimization_results.csv', index=False)
        print(f"\n✓ Results saved to optimization_results.csv")
        
        return eval_df


if __name__ == "__main__":
    optimizer = HyperparameterOptimizer()
    results = optimizer.run_optimization()
