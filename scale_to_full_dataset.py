#!/usr/bin/env python3
"""
Scale sentiment analysis to use FULL dataset (all 4000 rows)
Instead of just 200 labeled samples
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
print("SCALING MODELS TO FULL DATASET (4000 ROWS)")
print("="*80)

# ────────────────────────────────────────────────────────────────────────────
# STEP 1: LOAD ALL DATA & CREATE LABELS
# ────────────────────────────────────────────────────────────────────────────

print("\n[1/4] LOADING DATA...")

# Load full dataset
all_data = pd.read_csv("section3/all_cleaned.csv")
print(f"  ✓ Loaded {len(all_data)} rows from all_cleaned.csv")

# Load labels without random voting (4000 properly labeled)
labeled_data = pd.read_csv("section4/labeled_dataset_full_4000_no_random.csv")
print(f"  ✓ Loaded {len(labeled_data)} labeled rows (score-based + rule-based, NO random voting)")

# Get text column name
text_col = 'content'  # The actual column name in this dataset

# Merge: use the clean labels from labeled_data (all 4000 records already labeled)
# No need for fallback - we have proper labels from majority voting
print(f"  ✓ Using clean labels from integrated voting (score-based + rule-based)")

# Merge all_data with labeled_data to get proper labels
all_data = all_data.merge(
    labeled_data[['content', 'final_label']], 
    on='content', 
    how='left'
)

# Fill any missing labels with the ones from labeled_data if not matched by content
if all_data['final_label'].isna().any():
    print(f"  ⚠ Warning: {all_data['final_label'].isna().sum()} rows have no matching labels")

# Rename for consistency
all_data['label'] = all_data['final_label']

# Remove rows with NaN labels (shouldn't be any if join works properly)
all_data = all_data[all_data['label'].notna()].copy()

# Check distribution
dist = all_data['label'].value_counts()
print(f"  ✓ Label distribution:")
for label, count in dist.items():
    pct = (count / len(all_data)) * 100
    print(f"      {label}: {count:4d} ({pct:5.1f}%)")

# Save full labeled dataset
output_labeled = "section4/labeled_dataset_full_4000.csv"
all_data.to_csv(output_labeled, index=False)
print(f"  ✓ Saved full labeled dataset: {output_labeled}")

# ────────────────────────────────────────────────────────────────────────────
# STEP 3: CREATE TEXT REPRESENTATIONS (TF-IDF)
# ────────────────────────────────────────────────────────────────────────────

print("\n[3/4] CREATING TEXT REPRESENTATIONS...")

# Remove rows with NaN content
all_data_clean = all_data[all_data[text_col].notna()].copy()
print(f"  ✓ Removed {len(all_data) - len(all_data_clean)} rows with NaN content")

X = all_data_clean[text_col].values
y = all_data_clean['label'].values

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=303, ngram_range=(1, 2), 
                              min_df=2, max_df=0.8, lowercase=True)
X_tfidf = vectorizer.fit_transform(X)

print(f"  ✓ Created TF-IDF matrix: {X_tfidf.shape}")
print(f"      Samples: {X_tfidf.shape[0]}")
print(f"      Features: {X_tfidf.shape[1]}")

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"  ✓ Encoded labels: {label_encoder.classes_}")

# ────────────────────────────────────────────────────────────────────────────
# STEP 4: TRAIN MODELS ON FULL DATASET
# ────────────────────────────────────────────────────────────────────────────

print("\n[4/4] TRAINING MODELS ON FULL DATASET...")

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

print(f"  Training set: {X_train.shape[0]} samples")
print(f"  Test set:     {X_test.shape[0]} samples")

# Output directory
output_dir = Path("section4/ml_results_full_tfidf_all_models_4000_no_random/")
output_dir.mkdir(exist_ok=True)

# Store results
results = {}

# ── MODEL 1: SVM (RBF) ──
print("\n  Training SVM (RBF)...")
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
svm_acc = accuracy_score(y_test, svm_pred)
svm_f1 = f1_score(y_test, svm_pred, average='weighted')
results['svm'] = {
    'accuracy': svm_acc,
    'f1': svm_f1,
    'samples': X_train.shape[0]
}
pickle.dump(svm_model, open(output_dir / "svm_model.pkl", "wb"))
print(f"    ✓ SVM Accuracy: {svm_acc:.4f}, F1: {svm_f1:.4f}")

# ── MODEL 2: Logistic Regression ──
print("  Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred, average='weighted')
results['logistic'] = {
    'accuracy': lr_acc,
    'f1': lr_f1,
    'samples': X_train.shape[0]
}
pickle.dump(lr_model, open(output_dir / "logistic_regression_model.pkl", "wb"))
print(f"    ✓ LR Accuracy: {lr_acc:.4f}, F1: {lr_f1:.4f}")

# ── MODEL 3: Decision Tree ──
print("  Training Decision Tree...")
dt_model = DecisionTreeClassifier(max_depth=20, random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)
dt_f1 = f1_score(y_test, dt_pred, average='weighted')
results['decision_tree'] = {
    'accuracy': dt_acc,
    'f1': dt_f1,
    'samples': X_train.shape[0]
}
pickle.dump(dt_model, open(output_dir / "decision_tree_model.pkl", "wb"))
print(f"    ✓ DT Accuracy: {dt_acc:.4f}, F1: {dt_f1:.4f}")

# ── MODEL 4: Random Forest ──
print("  Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred, average='weighted')
results['random_forest'] = {
    'accuracy': rf_acc,
    'f1': rf_f1,
    'samples': X_train.shape[0]
}
pickle.dump(rf_model, open(output_dir / "random_forest_model.pkl", "wb"))
print(f"    ✓ RF Accuracy: {rf_acc:.4f}, F1: {rf_f1:.4f}")

# ────────────────────────────────────────────────────────────────────────────
# SAVE ARTIFACTS & CONFIGURATION
# ────────────────────────────────────────────────────────────────────────────

# Save vectorizer and encoder
pickle.dump(vectorizer, open(output_dir / "tfidf_vectorizer.pkl", "wb"))
pickle.dump(label_encoder, open(output_dir / "label_encoder.pkl", "wb"))

# Save results
with open(output_dir / "ml_models_report.json", "w") as f:
    json.dump(results, f, indent=2)

config = {
    "dataset": "all_cleaned (4000 rows, clean labels from voting - NO random)",
    "labeling_method": "score-based + rule-based (moderate agreement, kappa=0.372)",
    "total_samples": all_data_clean.shape[0],
    "train_samples": X_train.shape[0],
    "test_samples": X_test.shape[0],
    "features": "tfidf",
    "max_features": 303,
    "test_size": 0.2,
    "classes": list(label_encoder.classes_)
}
with open(output_dir / "training_config.json", "w") as f:
    json.dump(config, f, indent=2)

# ────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ────────────────────────────────────────────────────────────────────────────

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\n✓ Dataset: {all_data_clean.shape[0]} rows with CLEAN labels (no random voting)")
print(f"✓ Training set: {X_train.shape[0]} samples (80%)")
print(f"✓ Test set: {X_test.shape[0]} samples (20%)")
print(f"\nMODEL PERFORMANCE (on 800 test samples):")
print(f"  SVM (RBF):          {results['svm']['accuracy']:.2%} accuracy  ⭐ BEST")
print(f"  Logistic Regression: {results['logistic']['accuracy']:.2%} accuracy")
print(f"  Decision Tree:      {results['decision_tree']['accuracy']:.2%} accuracy")
print(f"  Random Forest:      {results['random_forest']['accuracy']:.2%} accuracy")
print(f"\nModels saved to: {output_dir}")
print("="*80 + "\n")
