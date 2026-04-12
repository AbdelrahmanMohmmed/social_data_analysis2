import argparse
import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)

# ──────────────────────────────────────────────────────────────────────────────
# ── ARGUMENT PARSING ──────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="ML-Based Sentiment Models: SVM and Logistic Regression")
parser.add_argument("--features",        type=str, required=True, 
                    help="CSV file with text representations (from text_representation.py)")
parser.add_argument("--labels",          type=str, default="labeled_reviews.csv",
                    help="CSV file with ground truth labels")
parser.add_argument("--output-dir",      type=str, default=".",
                    help="Output directory for models and results")
parser.add_argument("--feature-type",    choices=["tfidf", "glove", "both"], default="tfidf",
                    help="Which features to use: tfidf, glove, or both (default: tfidf)")
parser.add_argument("--svm",             action="store_true", help="Train SVM classifier")
parser.add_argument("--logistic",        action="store_true", help="Train Logistic Regression classifier")
parser.add_argument("--decision-tree",   action="store_true", help="Train Decision Tree classifier")
parser.add_argument("--random-forest",   action="store_true", help="Train Random Forest classifier")
parser.add_argument("--test-size",       type=float, default=0.2,
                    help="Test set size (default: 0.2)")
parser.add_argument("--svm-kernel",      choices=["linear", "rbf", "poly"], default="rbf",
                    help="SVM kernel type (default: rbf)")
parser.add_argument("--svm-c",           type=float, default=1.0,
                    help="SVM C parameter (default: 1.0)")
parser.add_argument("--dt-depth",        type=int, default=10,
                    help="Decision Tree max depth (default: 10)")
parser.add_argument("--rf-trees",        type=int, default=100,
                    help="Random Forest number of trees (default: 100)")

args = parser.parse_args()

# Require at least 1 model
if not args.svm and not args.logistic and not args.decision_tree and not args.random_forest:
    parser.error("You must enable at least 1 model (--svm, --logistic, --decision-tree, and/or --random-forest)")

# ──────────────────────────────────────────────────────────────────────────────
# ── LOAD DATA ─────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("ML-Based Sentiment Classification")
print("=" * 70 + "\n")

# Load features
features_df = pd.read_csv(args.features)
print(f"[+] Loaded features from {args.features}")
print(f"    Shape: {features_df.shape}")

# Load labels
labels_df = pd.read_csv(args.labels)
if "final_label" not in labels_df.columns:
    raise ValueError("Labels CSV must contain 'final_label' column")

print(f"[+] Loaded labels from {args.labels}")
print(f"  Label distribution:\n{labels_df['final_label'].value_counts()}\n")

# Select feature columns based on feature_type
feature_cols = []
if args.feature_type in ["tfidf", "both"]:
    tfidf_cols = [c for c in features_df.columns if c.startswith("tfidf_")]
    feature_cols.extend(tfidf_cols)
    print(f"[+] Using TF-IDF features: {len(tfidf_cols)} features")

if args.feature_type in ["glove", "both"]:
    glove_cols = [c for c in features_df.columns if c.startswith("glove_")]
    feature_cols.extend(glove_cols)
    print(f"[+] Using GloVe features: {len(glove_cols)} features")

if not feature_cols:
    raise ValueError(f"No features found for type '{args.feature_type}' in features file")

X = features_df[feature_cols].values
y = labels_df["final_label"].values

print(f"[+] Total features: {len(feature_cols)}\n")

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=args.test_size, random_state=42, stratify=y_encoded
)

print(f"Train set: {len(X_train)} samples")
print(f"Test set:  {len(X_test)} samples\n")

# Create output directory
Path(args.output_dir).mkdir(parents=True, exist_ok=True)

reports = {}
results_data = {
    "X_test_shape": X_test.shape,
    "y_test": y_test.tolist(),
    "class_labels": label_encoder.classes_.tolist()
}

# ──────────────────────────────────────────────────────────────────────────────
# ── SUPPORT VECTOR MACHINE (SVM) ──────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

if args.svm:
    print("=" * 70)
    print("Training SVM Classifier")
    print("=" * 70)
    print(f"Kernel: {args.svm_kernel}, C: {args.svm_c}\n")
    
    svm_model = SVC(kernel=args.svm_kernel, C=args.svm_c, random_state=42)
    svm_model.fit(X_train, y_train)
    
    y_pred_svm = svm_model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred_svm)
    precision = precision_score(y_test, y_pred_svm, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred_svm, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred_svm, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred_svm)
    
    reports["svm"] = {
        "model": f"SVM ({args.svm_kernel})",
        "parameters": {"kernel": args.svm_kernel, "C": args.svm_c},
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
        "test_samples": len(X_test)
    }
    
    results_data["svm_predictions"] = y_pred_svm.tolist()
    
    print(f"[+] Accuracy:  {accuracy:.4f}")
    print(f"[+] Precision: {precision:.4f}")
    print(f"[+] Recall:    {recall:.4f}")
    print(f"[+] F1-Score:  {f1:.4f}")
    print(f"\nConfusion Matrix:\n{cm}\n")
    
    # Save model
    svm_output = Path(args.output_dir) / "svm_model.pkl"
    with open(svm_output, "wb") as f:
        pickle.dump(svm_model, f)
    print(f"[+] Model saved -> {svm_output}\n")

# ──────────────────────────────────────────────────────────────────────────────
# ── LOGISTIC REGRESSION ────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

if args.logistic:
    print("=" * 70)
    print("Training Logistic Regression Classifier")
    print("=" * 70 + "\n")
    
    lr_model = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
    lr_model.fit(X_train, y_train)
    
    y_pred_lr = lr_model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred_lr)
    precision = precision_score(y_test, y_pred_lr, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred_lr, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred_lr, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred_lr)
    
    reports["logistic_regression"] = {
        "model": "Logistic Regression",
        "parameters": {"max_iter": 1000, "multi_class": "multinomial"},
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
        "test_samples": len(X_test)
    }
    
    results_data["logistic_predictions"] = y_pred_lr.tolist()
    
    print(f"[+] Accuracy:  {accuracy:.4f}")
    print(f"[+] Precision: {precision:.4f}")
    print(f"[+] Recall:    {recall:.4f}")
    print(f"[+] F1-Score:  {f1:.4f}")
    print(f"\nConfusion Matrix:\n{cm}\n")
    
    # Save model
    lr_output = Path(args.output_dir) / "logistic_regression_model.pkl"
    with open(lr_output, "wb") as f:
        pickle.dump(lr_model, f)
    print(f"[+] Model saved -> {lr_output}\n")

# ──────────────────────────────────────────────────────────────────────────────
# ── DECISION TREE ──────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

if args.decision_tree:
    print("=" * 70)
    print("Training Decision Tree Classifier")
    print("=" * 70)
    print(f"Max Depth: {args.dt_depth}\n")
    
    dt_model = DecisionTreeClassifier(max_depth=args.dt_depth, random_state=42)
    dt_model.fit(X_train, y_train)
    
    y_pred_dt = dt_model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred_dt)
    precision = precision_score(y_test, y_pred_dt, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred_dt, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred_dt, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred_dt)
    
    reports["decision_tree"] = {
        "model": "Decision Tree",
        "parameters": {"max_depth": args.dt_depth},
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
        "test_samples": len(X_test)
    }
    
    results_data["decision_tree_predictions"] = y_pred_dt.tolist()
    
    print(f"[+] Accuracy:  {accuracy:.4f}")
    print(f"[+] Precision: {precision:.4f}")
    print(f"[+] Recall:    {recall:.4f}")
    print(f"[+] F1-Score:  {f1:.4f}")
    print(f"\nConfusion Matrix:\n{cm}\n")
    
    # Save model
    dt_output = Path(args.output_dir) / "decision_tree_model.pkl"
    with open(dt_output, "wb") as f:
        pickle.dump(dt_model, f)
    print(f"[+] Model saved -> {dt_output}\n")

# ──────────────────────────────────────────────────────────────────────────────
# ── RANDOM FOREST ──────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

if args.random_forest:
    print("=" * 70)
    print("Training Random Forest Classifier")
    print("=" * 70)
    print(f"Number of Trees: {args.rf_trees}\n")
    
    rf_model = RandomForestClassifier(n_estimators=args.rf_trees, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    y_pred_rf = rf_model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred_rf)
    precision = precision_score(y_test, y_pred_rf, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred_rf, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred_rf, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred_rf)
    
    reports["random_forest"] = {
        "model": "Random Forest",
        "parameters": {"n_estimators": args.rf_trees},
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
        "test_samples": len(X_test)
    }
    
    results_data["random_forest_predictions"] = y_pred_rf.tolist()
    
    print(f"[+] Accuracy:  {accuracy:.4f}")
    print(f"[+] Precision: {precision:.4f}")
    print(f"[+] Recall:    {recall:.4f}")
    print(f"[+] F1-Score:  {f1:.4f}")
    print(f"\nConfusion Matrix:\n{cm}\n")
    
    # Save model
    rf_output = Path(args.output_dir) / "random_forest_model.pkl"
    with open(rf_output, "wb") as f:
        pickle.dump(rf_model, f)
    print(f"[+] Model saved -> {rf_output}\n")

# ──────────────────────────────────────────────────────────────────────────────
# ── MODEL COMPARISON & SUMMARY ────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("Model Comparison")
print("=" * 70)

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
print()

# ──────────────────────────────────────────────────────────────────────────────
# ── SAVE RESULTS ──────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

# Save label encoder for later use
encoder_output = Path(args.output_dir) / "label_encoder.pkl"
with open(encoder_output, "wb") as f:
    pickle.dump(label_encoder, f)
print(f"[+] Label encoder saved -> {encoder_output}")

# Save detailed report
report_output = Path(args.output_dir) / "ml_models_report.json"
with open(report_output, "w") as f:
    json.dump(reports, f, indent=2)
print(f"[+] Detailed report saved -> {report_output}")

# Save training configuration
config_output = Path(args.output_dir) / "training_config.json"
config = {
    "feature_file": args.features,
    "feature_type": args.feature_type,
    "n_features": len(feature_cols),
    "test_size": args.test_size,
    "train_samples": len(X_train),
    "test_samples": len(X_test),
    "classes": label_encoder.classes_.tolist()
}
with open(config_output, "w") as f:
    json.dump(config, f, indent=2)
print(f"[+] Training config saved -> {config_output}")

print("=" * 70)
print("All models trained and saved successfully!")
print("=" * 70)
