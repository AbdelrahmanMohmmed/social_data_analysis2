"""Hyperparameter optimization and fit diagnostics for ML sentiment models.

This script tunes the three ML models used in the project:
- SVM
- Logistic Regression
- Random Forest

It also reports overfitting/underfitting signals and compares tuned model performance.
"""

import argparse
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


def parse_args():
	parser = argparse.ArgumentParser(
		description="Tune SVM, Logistic Regression, and Random Forest with fit diagnostics"
	)
	parser.add_argument(
		"--features",
		type=str,
		default="../section4/model/labeled_clean_2_glove_embeddings.csv",
		help="Path to feature CSV from text_representation outputs",
	)
	parser.add_argument(
		"--labels",
		type=str,
		default="../section4/data/labeled_clean_2.csv",
		help="Path to labeled CSV with final_label",
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default="benchmark/optimization",
		help="Directory to save tuned models and reports",
	)
	parser.add_argument(
		"--feature-type",
		choices=["tfidf", "glove", "auto"],
		default="auto",
		help="Feature family to use. auto tries tfidf_ then glove_ prefixes",
	)
	parser.add_argument("--test-size", type=float, default=0.2, help="Test split size")
	parser.add_argument("--cv", type=int, default=5, help="CV folds for tuning")
	parser.add_argument(
		"--search-iter",
		type=int,
		default=30,
		help="RandomizedSearch iterations per model",
	)
	parser.add_argument(
		"--scoring",
		type=str,
		default="f1_weighted",
		help="Scoring metric for RandomizedSearchCV",
	)
	parser.add_argument("--random-state", type=int, default=42, help="Random seed")
	parser.add_argument(
		"--overfit-gap-threshold",
		type=float,
		default=0.10,
		help="If train_f1 - test_f1 > threshold, mark as overfitting",
	)
	parser.add_argument(
		"--underfit-f1-threshold",
		type=float,
		default=0.55,
		help="If both train/test f1 are below threshold, mark as underfitting",
	)
	return parser.parse_args()


def select_features(features_df: pd.DataFrame, feature_type: str) -> pd.DataFrame:
	if feature_type == "tfidf":
		cols = [c for c in features_df.columns if c.startswith("tfidf_")]
	elif feature_type == "glove":
		cols = [c for c in features_df.columns if c.startswith("glove_")]
	else:
		tfidf_cols = [c for c in features_df.columns if c.startswith("tfidf_")]
		glove_cols = [c for c in features_df.columns if c.startswith("glove_")]
		cols = tfidf_cols if tfidf_cols else glove_cols

	if not cols:
		raise ValueError(
			"No valid feature columns found. Expected columns prefixed with tfidf_ or glove_."
		)

	return features_df[cols]


def fit_status(train_f1: float, test_f1: float, overfit_gap: float, underfit_threshold: float) -> str:
	gap = train_f1 - test_f1
	if gap > overfit_gap:
		return "overfitting"
	if train_f1 < underfit_threshold and test_f1 < underfit_threshold:
		return "underfitting"
	return "balanced"


def evaluate_estimator(name, estimator, X_train, X_test, y_train, y_test):
	y_pred_train = estimator.predict(X_train)
	y_pred_test = estimator.predict(X_test)

	train_acc = accuracy_score(y_train, y_pred_train)
	test_acc = accuracy_score(y_test, y_pred_test)
	train_f1 = f1_score(y_train, y_pred_train, average="weighted", zero_division=0)
	test_f1 = f1_score(y_test, y_pred_test, average="weighted", zero_division=0)

	return {
		"model": name,
		"train_accuracy": float(train_acc),
		"test_accuracy": float(test_acc),
		"train_f1_weighted": float(train_f1),
		"test_f1_weighted": float(test_f1),
		"generalization_gap_f1": float(train_f1 - test_f1),
		"confusion_matrix": confusion_matrix(y_test, y_pred_test).tolist(),
		"classification_report": classification_report(y_test, y_pred_test, output_dict=True, zero_division=0),
	}


def main():
	args = parse_args()

	features_path = Path(args.features)
	labels_path = Path(args.labels)
	out_dir = Path(args.output_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	features_df = pd.read_csv(features_path)
	labels_df = pd.read_csv(labels_path)

	if "final_label" not in labels_df.columns:
		raise ValueError("Labels file must contain final_label column")
	if len(features_df) != len(labels_df):
		raise ValueError(
			f"Mismatched rows: features={len(features_df)} labels={len(labels_df)}"
		)

	X_df = select_features(features_df, args.feature_type)
	y_raw = labels_df["final_label"].astype(str).values

	encoder = LabelEncoder()
	y = encoder.fit_transform(y_raw)

	X_train, X_test, y_train, y_test = train_test_split(
		X_df.values,
		y,
		test_size=args.test_size,
		random_state=args.random_state,
		stratify=y,
	)

	print("=" * 80)
	print("Optimizing ML Models")
	print("=" * 80)
	print(f"Features: {features_path}")
	print(f"Labels: {labels_path}")
	print(f"Selected feature columns: {X_df.shape[1]}")
	print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

	model_spaces = {
		"svm": {
			"estimator": Pipeline(
				[
					("scaler", StandardScaler()),
					("clf", SVC(random_state=args.random_state)),
				]
			),
			"params": {
				"clf__kernel": ["linear", "rbf", "poly"],
				"clf__C": [0.01, 0.1, 1, 10, 100],
				"clf__gamma": ["scale", "auto"],
				"clf__class_weight": [None, "balanced"],
			},
		},
		"logistic": {
			"estimator": Pipeline(
				[
					("scaler", StandardScaler()),
					(
						"clf",
						LogisticRegression(
							max_iter=5000,
							random_state=args.random_state,
						),
					),
				]
			),
			"params": {
				"clf__C": [0.01, 0.1, 1, 10, 100],
				"clf__solver": ["lbfgs", "saga"],
				"clf__class_weight": [None, "balanced"],
			},
		},
		"random_forest": {
			"estimator": RandomForestClassifier(random_state=args.random_state, n_jobs=-1),
			"params": {
				"n_estimators": [100, 200, 300, 500],
				"max_depth": [None, 8, 12, 16, 24],
				"min_samples_split": [2, 5, 10],
				"min_samples_leaf": [1, 2, 4],
				"max_features": ["sqrt", "log2", None],
				"class_weight": [None, "balanced", "balanced_subsample"],
			},
		},
	}

	tuned_summary = []

	for model_name, spec in model_spaces.items():
		print("\n" + "-" * 80)
		print(f"Tuning {model_name}...")
		print("-" * 80)

		search = RandomizedSearchCV(
			estimator=spec["estimator"],
			param_distributions=spec["params"],
			n_iter=args.search_iter,
			scoring=args.scoring,
			cv=args.cv,
			random_state=args.random_state,
			n_jobs=-1,
			refit=True,
			verbose=0,
		)

		search.fit(X_train, y_train)
		best_model = search.best_estimator_

		metrics = evaluate_estimator(model_name, best_model, X_train, X_test, y_train, y_test)
		metrics["best_cv_score"] = float(search.best_score_)
		metrics["best_params"] = search.best_params_
		metrics["fit_status"] = fit_status(
			metrics["train_f1_weighted"],
			metrics["test_f1_weighted"],
			args.overfit_gap_threshold,
			args.underfit_f1_threshold,
		)

		tuned_summary.append(metrics)

		# Save per-model artifacts
		with (out_dir / f"{model_name}_best_params.json").open("w", encoding="utf-8") as f:
			json.dump(search.best_params_, f, indent=2)

		with (out_dir / f"{model_name}_metrics.json").open("w", encoding="utf-8") as f:
			json.dump(metrics, f, indent=2)

		with (out_dir / f"{model_name}_best_model.pkl").open("wb") as f:
			pickle.dump(best_model, f)

		print(
			f"Best CV ({args.scoring}): {metrics['best_cv_score']:.4f} | "
			f"Test F1: {metrics['test_f1_weighted']:.4f} | "
			f"Fit status: {metrics['fit_status']}"
		)

	# Save shared artifacts
	with (out_dir / "label_encoder.pkl").open("wb") as f:
		pickle.dump(encoder, f)

	comparison_df = pd.DataFrame(tuned_summary).sort_values("test_f1_weighted", ascending=False)
	comparison_df.to_csv(out_dir / "tuned_models_comparison.csv", index=False)

	best = comparison_df.iloc[0].to_dict()
	summary = {
		"features_file": str(features_path),
		"labels_file": str(labels_path),
		"feature_columns": int(X_df.shape[1]),
		"train_samples": int(len(X_train)),
		"test_samples": int(len(X_test)),
		"search_iterations": int(args.search_iter),
		"cv_folds": int(args.cv),
		"best_model": {
			"name": best["model"],
			"test_f1_weighted": float(best["test_f1_weighted"]),
			"test_accuracy": float(best["test_accuracy"]),
			"fit_status": best["fit_status"],
		},
		"classes": encoder.classes_.tolist(),
	}
	with (out_dir / "optimization_summary.json").open("w", encoding="utf-8") as f:
		json.dump(summary, f, indent=2)

	# Visualization: train vs test F1 and generalization gap
	plt.figure(figsize=(10, 6))
	x = np.arange(len(comparison_df))
	width = 0.35
	plt.bar(x - width / 2, comparison_df["train_f1_weighted"], width, label="Train F1")
	plt.bar(x + width / 2, comparison_df["test_f1_weighted"], width, label="Test F1")
	plt.xticks(x, comparison_df["model"].tolist())
	plt.ylim(0, 1)
	plt.ylabel("F1 (weighted)")
	plt.title("Train vs Test F1 after Hyperparameter Tuning")
	plt.legend()
	plt.tight_layout()
	plt.savefig(out_dir / "tuned_train_test_f1.png", dpi=200)
	plt.close()

	plt.figure(figsize=(8, 5))
	plt.bar(comparison_df["model"], comparison_df["generalization_gap_f1"], color="#B56576")
	plt.axhline(args.overfit_gap_threshold, color="red", linestyle="--", label="Overfit threshold")
	plt.ylabel("Train F1 - Test F1")
	plt.title("Generalization Gap by Model")
	plt.legend()
	plt.tight_layout()
	plt.savefig(out_dir / "generalization_gap.png", dpi=200)
	plt.close()

	print("\n" + "=" * 80)
	print("Optimization complete")
	print("=" * 80)
	print(f"Best model: {best['model']}")
	print(f"Best test F1: {best['test_f1_weighted']:.4f}")
	print(f"Best fit status: {best['fit_status']}")
	print(f"Saved outputs in: {out_dir}")


if __name__ == "__main__":
	main()
