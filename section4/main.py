import subprocess
import sys
from pathlib import Path


def run_step(command, title):
	print("\n" + "=" * 80)
	print(title)
	print("=" * 80)
	print("Command:", " ".join(command))
	subprocess.run(command, check=True)


def main():
	section4_dir = Path(__file__).resolve().parent
	root_dir = section4_dir.parent
	py = sys.executable

	# Inputs from section3 pipeline
	section3_data_dir = root_dir / "section3" / "data"

	# Output locations in section4
	data_dir = section4_dir / "data"
	model_dir = section4_dir / "model"
	lex_model_dir = section4_dir / "lex_model"
	models_dir = section4_dir / "models"

	data_dir.mkdir(parents=True, exist_ok=True)
	model_dir.mkdir(parents=True, exist_ok=True)
	lex_model_dir.mkdir(parents=True, exist_ok=True)
	models_dir.mkdir(parents=True, exist_ok=True)

	label_script = section4_dir / "label_data.py"
	repr_script = section4_dir / "text_representation.py"
	lexical_script = section4_dir / "lexical_based_models.py"
	ml_script = section4_dir / "ml_based_models.py"

	# 1) Label clean_1..clean_4 -> section4/data/labeled_clean_*.csv
	for i in range(1, 5):
		input_file = section3_data_dir / f"labeled_reviews_raw_clean_{i}.csv"
		output_file = data_dir / f"labeled_clean_{i}.csv"

		run_step(
			[
				py,
				str(label_script),
				"--score-based",
				"--rule-based",
				"--size",
				"200",
				"--input",
				str(input_file),
				"--output",
				str(output_file),
			],
			f"Step 1.{i} - Label data clean_{i}",
		)

	# 2) Build TF-IDF + GloVe representations for all labeled files -> section4/model
	run_step(
		[
			py,
			str(repr_script),
			"--input",
			str(data_dir),
			"--output-dir",
			str(model_dir),
			"--tfidf",
			"--glove",
		],
		"Step 2 - Create text representations (TF-IDF + GloVe)",
	)

	# 3) Lexical models for clean_1..clean_4 -> section4/lex_model
	for i in range(1, 5):
		labeled_file = data_dir / f"labeled_clean_{i}.csv"
		lexical_out = lex_model_dir / f"lexical_clean_{i}.csv"
		lexical_report = lex_model_dir / f"lexical_clean_{i}_report.json"

		run_step(
			[
				py,
				str(lexical_script),
				"--input",
				str(labeled_file),
				"--output",
				str(lexical_out),
				"--output-report",
				str(lexical_report),
				"--vader",
				"--afinn",
				"--afinn-negation",
			],
			f"Step 3.{i} - Run lexical models for clean_{i}",
		)

	# 4) ML models (SVM + Logistic + RF) for TF-IDF and GloVe -> section4/models
	for i in range(1, 5):
		labels_file = data_dir / f"labeled_clean_{i}.csv"

		tfidf_features = model_dir / f"labeled_clean_{i}_tfidf_matrix.csv"
		tfidf_out_dir = models_dir / f"clean_{i}_tfidf"
		run_step(
			[
				py,
				str(ml_script),
				"--features",
				str(tfidf_features),
				"--labels",
				str(labels_file),
				"--output-dir",
				str(tfidf_out_dir),
				"--feature-type",
				"tfidf",
				"--svm",
				"--logistic",
				"--random-forest",
			],
			f"Step 4.{i}.1 - Train ML models for clean_{i} TF-IDF",
		)

		glove_features = model_dir / f"labeled_clean_{i}_glove_embeddings.csv"
		glove_out_dir = models_dir / f"clean_{i}_glove"
		run_step(
			[
				py,
				str(ml_script),
				"--features",
				str(glove_features),
				"--labels",
				str(labels_file),
				"--output-dir",
				str(glove_out_dir),
				"--feature-type",
				"glove",
				"--svm",
				"--logistic",
				"--random-forest",
			],
			f"Step 4.{i}.2 - Train ML models for clean_{i} GloVe",
		)

	print("\nPipeline completed successfully.")
	print(f"Labeled data: {data_dir}")
	print(f"Representations: {model_dir}")
	print(f"Lexical results: {lex_model_dir}")
	print(f"ML models/results: {models_dir}")


if __name__ == "__main__":
	main()
