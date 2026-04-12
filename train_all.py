#!/usr/bin/env python3
"""
Train All Models - SVM, Logistic Regression, Decision Tree, Random Forest
"""

import sys
sys.path.insert(0, '/media/abdo/Games/social_data_analysis/section4')
sys.path.insert(0, '/media/abdo/Games/social_data_analysis/section5')

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

print("\n" + "="*70)
print("🚀 TRAINING ALL SENTIMENT ANALYSIS MODELS")
print("="*70 + "\n")

# Setup paths
section4_dir = Path("/media/abdo/Games/social_data_analysis/section4")
output_dir = section4_dir / "ml_results_full_tfidf_all_models"
output_dir.mkdir(parents=True, exist_ok=True)

# Load features
print("[1] Loading TF-IDF features...")
features_file = section4_dir / "representations_full" / "tfidf_matrix.csv"
features_df = pd.read_csv(features_file)
X = features_df.values

print(f"    ✓ Loaded {X.shape[0]} samples with {X.shape[1]} features")

# Load labels
print("[2] Loading labels...")
labels_file = section4_dir / "labeled_dataset_04_full.csv"
labels_df = pd.read_csv(labels_file)

# Find label column
label_col = None
for col in ['sentiment', 'final_label', 'label', 'class']:
    if col in labels_df.columns:
        label_col = col
        break

if label_col is None:
    print(f"    ❌ No label column found. Available: {labels_df.columns.tolist()}")
    sys.exit(1)

y = labels_df[label_col].values
print(f"    ✓ Loaded {len(y)} labels")
print(f"    ✓ Distribution:\n{pd.Series(y).value_counts()}\n")

# Encode labels
print("[3] Encoding labels...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"    ✓ Classes: {label_encoder.classes_}")

# Split data
print("[4] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"    ✓ Train: {len(X_train)}, Test: {len(X_test)}\n")

# Save label encoder
encoder_path = output_dir / "label_encoder.pkl"
with open(encoder_path, "wb") as f:
    pickle.dump(label_encoder, f)
print(f"[*] Label encoder saved -> {encoder_path}")

# Save vectorizer (copy from representations_full)
vectorizer_src = section4_dir / "representations_full" / "tfidf_vectorizer.pkl"
vectorizer_dst = output_dir / "tfidf_vectorizer.pkl"
if vectorizer_src.exists():
    import shutil
    shutil.copy(vectorizer_src, vectorizer_dst)
    print(f"[*] Vectorizer copied -> {vectorizer_dst}\n")

models = {}
reports = {}

# ============================================================================
# SVM Model
# ============================================================================
print("="*70)
print("Training SVM (RBF Kernel)")
print("="*70)

svm_model = SVC(kernel='rbf', C=1.0, random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

acc_svm = accuracy_score(y_test, y_pred_svm)
prec_svm = precision_score(y_test, y_pred_svm, average='weighted', zero_division=0)
rec_svm = recall_score(y_test, y_pred_svm, average='weighted', zero_division=0)
f1_svm = f1_score(y_test, y_pred_svm, average='weighted', zero_division=0)
cm_svm = confusion_matrix(y_test, y_pred_svm)

models['svm'] = svm_model
reports['svm'] = {
    "model": "SVM (RBF)",
    "accuracy": float(acc_svm),
    "precision": float(prec_svm),
    "recall": float(rec_svm),
    "f1_score": float(f1_svm)
}

svm_path = output_dir / "svm_model.pkl"
with open(svm_path, "wb") as f:
    pickle.dump(svm_model, f)

print(f"Accuracy:  {acc_svm:.4f}")
print(f"Precision: {prec_svm:.4f}")
print(f"Recall:    {rec_svm:.4f}")
print(f"F1-Score:  {f1_svm:.4f}")
print(f"✓ Saved -> {svm_path}\n")

# ============================================================================
# Logistic Regression
# ============================================================================
print("="*70)
print("Training Logistic Regression")
print("="*70)

lr_model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

acc_lr = accuracy_score(y_test, y_pred_lr)
prec_lr = precision_score(y_test, y_pred_lr, average='weighted', zero_division=0)
rec_lr = recall_score(y_test, y_pred_lr, average='weighted', zero_division=0)
f1_lr = f1_score(y_test, y_pred_lr, average='weighted', zero_division=0)
cm_lr = confusion_matrix(y_test, y_pred_lr)

models['logistic_regression'] = lr_model
reports['logistic_regression'] = {
    "model": "Logistic Regression",
    "accuracy": float(acc_lr),
    "precision": float(prec_lr),
    "recall": float(rec_lr),
    "f1_score": float(f1_lr)
}

lr_path = output_dir / "logistic_regression_model.pkl"
with open(lr_path, "wb") as f:
    pickle.dump(lr_model, f)

print(f"Accuracy:  {acc_lr:.4f}")
print(f"Precision: {prec_lr:.4f}")
print(f"Recall:    {rec_lr:.4f}")
print(f"F1-Score:  {f1_lr:.4f}")
print(f"✓ Saved -> {lr_path}\n")

# ============================================================================
# Decision Tree
# ============================================================================
print("="*70)
print("Training Decision Tree")
print("="*70)

dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

acc_dt = accuracy_score(y_test, y_pred_dt)
prec_dt = precision_score(y_test, y_pred_dt, average='weighted', zero_division=0)
rec_dt = recall_score(y_test, y_pred_dt, average='weighted', zero_division=0)
f1_dt = f1_score(y_test, y_pred_dt, average='weighted', zero_division=0)
cm_dt = confusion_matrix(y_test, y_pred_dt)

models['decision_tree'] = dt_model
reports['decision_tree'] = {
    "model": "Decision Tree",
    "accuracy": float(acc_dt),
    "precision": float(prec_dt),
    "recall": float(rec_dt),
    "f1_score": float(f1_dt)
}

dt_path = output_dir / "decision_tree_model.pkl"
with open(dt_path, "wb") as f:
    pickle.dump(dt_model, f)

print(f"Accuracy:  {acc_dt:.4f}")
print(f"Precision: {prec_dt:.4f}")
print(f"Recall:    {rec_dt:.4f}")
print(f"F1-Score:  {f1_dt:.4f}")
print(f"✓ Saved -> {dt_path}\n")

# ============================================================================
# Random Forest
# ============================================================================
print("="*70)
print("Training Random Forest")
print("="*70)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

acc_rf = accuracy_score(y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf, average='weighted', zero_division=0)
rec_rf = recall_score(y_test, y_pred_rf, average='weighted', zero_division=0)
f1_rf = f1_score(y_test, y_pred_rf, average='weighted', zero_division=0)
cm_rf = confusion_matrix(y_test, y_pred_rf)

models['random_forest'] = rf_model
reports['random_forest'] = {
    "model": "Random Forest",
    "accuracy": float(acc_rf),
    "precision": float(prec_rf),
    "recall": float(rec_rf),
    "f1_score": float(f1_rf)
}

rf_path = output_dir / "random_forest_model.pkl"
with open(rf_path, "wb") as f:
    pickle.dump(rf_model, f)

print(f"Accuracy:  {acc_rf:.4f}")
print(f"Precision: {prec_rf:.4f}")
print(f"Recall:    {rec_rf:.4f}")
print(f"F1-Score:  {f1_rf:.4f}")
print(f"✓ Saved -> {rf_path}\n")

# ============================================================================
# Model Comparison
# ============================================================================
print("="*70)
print("📊 MODEL COMPARISON")
print("="*70)

comparison_data = []
for model_name, metrics in reports.items():
    comparison_data.append({
        "Model": metrics["model"],
        "Accuracy": f"{metrics['accuracy']:.4f}",
        "Precision": f"{metrics['precision']:.4f}",
        "Recall": f"{metrics['recall']:.4f}",
        "F1-Score": f"{metrics['f1_score']:.4f}"
    })

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

# Save reports
report_path = output_dir / "ml_models_report.json"
with open(report_path, "w") as f:
    json.dump(reports, f, indent=2)
print(f"\n✓ Report saved -> {report_path}")

# Save training config
config = {
    "feature_type": "tfidf",
    "n_features": X.shape[1],
    "test_size": 0.2,
    "train_samples": len(X_train),
    "test_samples": len(X_test),
    "classes": label_encoder.classes_.tolist(),
    "models": list(reports.keys())
}

config_path = output_dir / "training_config.json"
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
print(f"✓ Config saved -> {config_path}")

print("\n" + "="*70)
print("✅ ALL MODELS TRAINED AND SAVED!")
print("="*70)
print(f"\nModels saved to: {output_dir}")
print("\nNext step: Run Streamlit app")
print("  cd /media/abdo/Games/social_data_analysis/section5")
print("  streamlit run streamlit_app.py")
print()
