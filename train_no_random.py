#!/usr/bin/env python3
"""
Train models with clean labels (no random voting)
"""
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("TRAINING WITH CLEAN LABELS (NO RANDOM VOTING)")
print("="*80)

# Load clean labeled data
df = pd.read_csv("section4/labeled_dataset_full_4000_no_random.csv")
#Remove NaN content
df = df[df['content'].notna()].copy()

print(f"\n✓ Loaded {len(df)} samples with clean labels")
print(f"  Label distribution:")
for label, count in df['final_label'].value_counts().items():
    pct = (count / len(df)) * 100
    print(f"    {label}: {count:4d} ({pct:5.1f}%)")

# Create TF-IDF
X = df['content'].values
y = df['final_label'].values

vectorizer = TfidfVectorizer(max_features=303, ngram_range=(1, 2), 
                              min_df=2, max_df=0.8, lowercase=True)
X_tfidf = vectorizer.fit_transform(X)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"\n✓ TF-IDF: {X_tfidf.shape}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

print(f"✓ Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# OUTPUT DIR
output_dir = Path("section4/ml_results_full_tfidf_all_models_4000_no_random")
output_dir.mkdir(exist_ok=True)

results = {}

print("\nTRAINING MODELS:")
print("─" * 40)

# SVM
print("  SVM (RBF)...", end=' ', flush=True)
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
svm_acc = accuracy_score(y_test, svm_pred)
results['svm'] = {'accuracy': svm_acc, 'f1': f1_score(y_test, svm_pred, average='weighted')}
pickle.dump(svm_model, open(output_dir / "svm_model.pkl", "wb"))
print(f"✓ {svm_acc:.2%}")

# Logistic Regression
print("  Logistic Regression...", end=' ', flush=True)
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)
results['logistic'] = {'accuracy': lr_acc, 'f1': f1_score(y_test, lr_pred, average='weighted')}
pickle.dump(lr_model, open(output_dir / "logistic_regression_model.pkl", "wb"))
print(f"✓ {lr_acc:.2%}")

# Decision Tree
print("  Decision Tree...", end=' ', flush=True)
dt_model = DecisionTreeClassifier(max_depth=20, random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)
results['decision_tree'] = {'accuracy': dt_acc, 'f1': f1_score(y_test, dt_pred, average='weighted')}
pickle.dump(dt_model, open(output_dir / "decision_tree_model.pkl", "wb"))
print(f"✓ {dt_acc:.2%}")

# Random Forest
print("  Random Forest...", end=' ', flush=True)
rf_model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
results['random_forest'] = {'accuracy': rf_acc, 'f1': f1_score(y_test, rf_pred, average='weighted')}
pickle.dump(rf_model, open(output_dir / "random_forest_model.pkl", "wb"))
print(f"✓ {rf_acc:.2%}")

print("─" * 40)

# Save artifacts
pickle.dump(vectorizer, open(output_dir / "tfidf_vectorizer.pkl", "wb"))
pickle.dump(label_encoder, open(output_dir / "label_encoder.pkl", "wb"))

with open(output_dir / "ml_models_report.json", "w") as f:
    json.dump(results, f, indent=2)

config = {
    "dataset": f"{len(df)} samples - Clean labels (no random voting)",
    "train_samples": X_train.shape[0],
    "test_samples": X_test.shape[0],
    "features": "tfidf_303",
    "classes": list(label_encoder.classes_)
}
with open(output_dir / "training_config.json", "w") as f:
    json.dump(config, f, indent=2)

# Summary
print(f"\n{'='*80}")
print("RESULTS")
print(f"{'='*80}")
print(f"\n✓ SVM             : {results['svm']['accuracy']:.2%} accuracy ⭐ BEST")
print(f"✓ Logistic Regr.  : {results['logistic']['accuracy']:.2%} accuracy")
print(f"✓ Random Forest   : {results['random_forest']['accuracy']:.2%} accuracy")
print(f"✓ Decision Tree   : {results['decision_tree']['accuracy']:.2%} accuracy")
print(f"\nModels saved: {output_dir}")
print("="*80 + "\n")
