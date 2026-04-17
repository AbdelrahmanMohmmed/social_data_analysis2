import subprocess
import sys
from pathlib import Path


def build_commands(base_dir: Path):
	script_path = base_dir / "text_preprocessing_v2.py"
	input_csv = base_dir / "labeled_reviews_raw.csv"
	output_dir = base_dir / "data"

	commands = [
		[
			sys.executable,
			str(script_path),
			"--input",
			str(input_csv),
			"--output",
			str(output_dir / "labeled_reviews_raw_clean_1.csv"),
			"--remove_urls",
			"--remove_emojis",
			"--remove_punctuation",
		],
		[
			sys.executable,
			str(script_path),
			"--input",
			str(input_csv),
			"--output",
			str(output_dir / "labeled_reviews_raw_clean_2.csv"),
			"--remove_urls",
			"--remove_emojis",
			"--remove_punctuation",
			"--remove_stopwords",
		],
		[
			sys.executable,
			str(script_path),
			"--input",
			str(input_csv),
			"--output",
			str(output_dir / "labeled_reviews_raw_clean_3.csv"),
			"--remove_urls",
			"--remove_emojis",
			"--remove_punctuation",
			"--lemmatize",
		],
		[
			sys.executable,
			str(script_path),
			"--input",
			str(input_csv),
			"--output",
			str(output_dir / "labeled_reviews_raw_clean_4.csv"),
			"--remove_urls",
			"--remove_emojis",
			"--remove_punctuation",
			"--remove_stopwords",
			"--lemmatize",
			"--fix_spelling",
		],
	]
	return commands, output_dir


def main():
	base_dir = Path(__file__).resolve().parent
	commands, output_dir = build_commands(base_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	for i, cmd in enumerate(commands, start=1):
		print(f"Running clean_{i}...")
		subprocess.run(cmd, check=True)

	print("All clean files were generated in section3/data")


if __name__ == "__main__":
	main()
