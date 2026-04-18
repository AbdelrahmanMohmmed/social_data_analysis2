"""Compare ML and lexical model results and pick the best-performing model."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_ml_reports(models_dir: Path):
    rows = []
    for report_path in sorted(models_dir.glob("*/ml_models_report.json")):
        run_id = report_path.parent.name  # e.g. clean_1_tfidf
        with report_path.open("r", encoding="utf-8") as f:
            report = json.load(f)

        for model_key, metrics in report.items():
            rows.append(
                {
                    "family": "ml",
                    "run_id": run_id,
                    "feature_type": "tfidf" if run_id.endswith("_tfidf") else "glove",
                    "preprocessing": run_id.replace("_tfidf", "").replace("_glove", ""),
                    "model_key": model_key,
                    "model_name": metrics.get("model", model_key),
                    "accuracy": float(metrics.get("accuracy", 0.0)),
                    "precision": float(metrics.get("precision", 0.0)),
                    "recall": float(metrics.get("recall", 0.0)),
                    "f1_score": float(metrics.get("f1_score", 0.0)),
                    "confusion_matrix": metrics.get("confusion_matrix"),
                    "source_report": str(report_path),
                }
            )
    return rows


def load_lex_reports(lex_dir: Path):
    rows = []
    for report_path in sorted(lex_dir.glob("lexical_clean_*_report.json")):
        run_id = report_path.stem.replace("_report", "")  # lexical_clean_1
        with report_path.open("r", encoding="utf-8") as f:
            report = json.load(f)

        for model_key, metrics in report.items():
            rows.append(
                {
                    "family": "lex",
                    "run_id": run_id,
                    "feature_type": "lexical",
                    "preprocessing": run_id.replace("lexical_", ""),
                    "model_key": model_key,
                    "model_name": metrics.get("model", model_key),
                    "accuracy": float(metrics.get("accuracy", 0.0)),
                    "precision": float(metrics.get("precision", 0.0)),
                    "recall": float(metrics.get("recall", 0.0)),
                    "f1_score": float(metrics.get("f1_score", 0.0)),
                    "confusion_matrix": metrics.get("confusion_matrix"),
                    "source_report": str(report_path),
                }
            )
    return rows


def save_visualizations(df: pd.DataFrame, out_dir: Path):
    # Top models by F1 score
    top = df.sort_values(["f1_score", "accuracy"], ascending=False).head(12)
    plt.figure(figsize=(12, 6))
    labels = [f"{r['model_name']}\n({r['run_id']})" for _, r in top.iterrows()]
    plt.bar(range(len(top)), top["f1_score"], color="#2E86AB")
    plt.xticks(range(len(top)), labels, rotation=45, ha="right")
    plt.ylabel("F1 Score")
    plt.title("Top Model Configurations by F1 Score")
    plt.tight_layout()
    plt.savefig(out_dir / "top_models_f1.png", dpi=200)
    plt.close()

    # Family average comparison (ML vs Lex)
    family_avg = df.groupby("family")[["accuracy", "precision", "recall", "f1_score"]].mean()
    family_avg.plot(kind="bar", figsize=(10, 6))
    plt.title("Average Metrics by Model Family")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_dir / "family_average_metrics.png", dpi=200)
    plt.close()

    # Best model confusion matrix
    best = df.sort_values(["f1_score", "accuracy"], ascending=False).iloc[0]
    cm = best["confusion_matrix"]
    if isinstance(cm, list) and cm:
        cm_arr = np.array(cm)
        plt.figure(figsize=(6, 5))
        plt.imshow(cm_arr, cmap="Blues")
        plt.title(f"Confusion Matrix - Best Model\n{best['model_name']} ({best['run_id']})")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        for i in range(cm_arr.shape[0]):
            for j in range(cm_arr.shape[1]):
                plt.text(j, i, str(cm_arr[i, j]), ha="center", va="center", color="black")
        plt.tight_layout()
        plt.savefig(out_dir / "best_model_confusion_matrix.png", dpi=200)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Benchmark ML vs lexical models and select best run")
    parser.add_argument("--section4-dir", default="../section4", help="Path to section4 directory")
    parser.add_argument("--output-dir", default="benchmark", help="Path to save benchmark outputs")
    args = parser.parse_args()

    section4_dir = Path(args.section4_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    models_dir = section4_dir / "models"
    lex_dir = section4_dir / "lex_model"

    ml_rows = load_ml_reports(models_dir)
    lex_rows = load_lex_reports(lex_dir)
    all_rows = ml_rows + lex_rows

    if not all_rows:
        raise FileNotFoundError(
            "No benchmarkable reports found. Expected ML reports in section4/models and "
            "lexical reports in section4/lex_model."
        )

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["f1_score", "accuracy"], ascending=False).reset_index(drop=True)

    best_overall = df.iloc[0].to_dict()
    best_ml = df[df["family"] == "ml"].iloc[0].to_dict() if (df["family"] == "ml").any() else None
    best_lex = df[df["family"] == "lex"].iloc[0].to_dict() if (df["family"] == "lex").any() else None

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_runs": int(len(df)),
        "total_ml_runs": int((df["family"] == "ml").sum()),
        "total_lex_runs": int((df["family"] == "lex").sum()),
        "best_overall": {
            "model_name": best_overall["model_name"],
            "family": best_overall["family"],
            "run_id": best_overall["run_id"],
            "accuracy": float(best_overall["accuracy"]),
            "precision": float(best_overall["precision"]),
            "recall": float(best_overall["recall"]),
            "f1_score": float(best_overall["f1_score"]),
            "source_report": best_overall["source_report"],
        },
        "best_ml": best_ml,
        "best_lex": best_lex,
    }

    # Save tabular results + summary
    df.to_csv(output_dir / "model_benchmark_table.csv", index=False)
    with (output_dir / "benchmark_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Also write a compact markdown summary for easy reading
    with (output_dir / "benchmark_summary.md").open("w", encoding="utf-8") as f:
        f.write("# Benchmark Summary\n\n")
        f.write(f"Total runs: {len(df)}  \\n")
        f.write(f"ML runs: {(df['family'] == 'ml').sum()}  \\n")
        f.write(f"Lexical runs: {(df['family'] == 'lex').sum()}\n\n")
        f.write("## Best Overall\n")
        f.write(f"- Model: {best_overall['model_name']}\\n")
        f.write(f"- Family: {best_overall['family']}\\n")
        f.write(f"- Run: {best_overall['run_id']}\\n")
        f.write(f"- F1: {best_overall['f1_score']:.4f}\\n")
        f.write(f"- Accuracy: {best_overall['accuracy']:.4f}\\n\n")
        f.write("## Top 10 by F1\n\n")
        for i, (_, row) in enumerate(df.head(10).iterrows(), 1):
            f.write(
                f"{i}. {row['model_name']} ({row['run_id']}, {row['family']}) - "
                f"F1={row['f1_score']:.4f}, Acc={row['accuracy']:.4f}\n"
            )

    save_visualizations(df, output_dir)

    print("\n" + "=" * 72)
    print("Benchmark complete")
    print("=" * 72)
    print(f"Best overall: {best_overall['model_name']} ({best_overall['run_id']}, {best_overall['family']})")
    print(f"F1: {best_overall['f1_score']:.4f} | Accuracy: {best_overall['accuracy']:.4f}")
    print(f"Outputs saved in: {output_dir}")


if __name__ == "__main__":
    main()
