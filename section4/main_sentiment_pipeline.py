#!/usr/bin/env python3
"""
SENTIMENT ANALYSIS MAIN PIPELINE
Orchestrates the complete workflow:
  1. Creates 4 preprocessed datasets with different strategies
  2. Labels each dataset
  3. Generates text representations (TF-IDF + GloVe)
  4. Trains ML models with mixed pipeline combinations
  5. Compares all results and generates final report
"""

import argparse
import subprocess
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import os

# ──────────────────────────────────────────────────────────────────────────────
# ── CONFIGURATION ──────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

class PipelineConfig:
    def __init__(self):
        self.raw_data = "all_cleaned.csv"  # Should exist in section3/
        self.section3_dir = Path("../section3")
        self.section4_dir = Path(".")
        self.output_dir = Path("pipeline_results")
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Dataset configurations
        self.datasets = {
            "minimal": {
                "name": "Minimal (Emojis Only)",
                "flags": ["--remove_emojis"],
                "output": "dataset_01_minimal_emojis.csv"
            },
            "stopwords": {
                "name": "Stop Words Removed",
                "flags": ["--lowercase", "--remove_stopwords"],
                "output": "dataset_02_stopwords.csv"
            },
            "lemmatization": {
                "name": "Lemmatization Only",
                "flags": ["--lowercase", "--remove_punctuation", "--lemmatize"],
                "output": "dataset_03_lemmatization.csv"
            },
            "full": {
                "name": "Full Preprocessing",
                "flags": ["--lowercase", "--remove_urls", "--remove_emojis", 
                         "--remove_punctuation", "--remove_stopwords", "--lemmatize"],
                "output": "dataset_04_full.csv"
            }
        }
        
        # ML pipeline combinations (TF-IDF only for reliability)
        self.ml_combinations = [
            {"features": "tfidf", "model": "svm", "kernel": "rbf"},
            {"features": "tfidf", "model": "svm", "kernel": "linear"},
            {"features": "tfidf", "model": "logistic"},
        ]

# ──────────────────────────────────────────────────────────────────────────────
# ── UTILITY FUNCTIONS ──────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

def run_command(cmd, description, shell=False):
    """Execute a command and return success status"""
    print(f"\n{'='*70}")
    print(f"▶ {description}")
    print(f"{'='*70}")
    try:
        # Replace 'python' with the current interpreter
        if cmd[0] == 'python':
            cmd[0] = sys.executable
        
        result = subprocess.run(cmd, shell=shell, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: {e}")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        return False

def log(message, level="INFO"):
    """Print formatted log messages"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    symbols = {"INFO": "[*]", "SUCCESS": "[+]", "ERROR": "[!]", "WARN": "[?]"}
    print(f"[{timestamp}] {symbols.get(level, '->')} {message}")

# ──────────────────────────────────────────────────────────────────────────────
# ── STEP 1: CREATE PREPROCESSED DATASETS ────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

def create_datasets(config):
    """Create 4 datasets with different preprocessing strategies"""
    log("STEP 1: Creating 4 preprocessed datasets", "INFO")
    
    datasets_created = {}
    
    for key, dataset_config in config.datasets.items():
        dataset_output = config.section4_dir / dataset_config["output"]
        
        # Skip if already exists
        if dataset_output.exists():
            log(f"Dataset '{dataset_config['name']}' already exists, skipping", "WARN")
            datasets_created[key] = dataset_output
            continue
        
        # Build command
        cmd = [
            "python", "../section3/text_preprocessing_v2.py",
            "--input", str(config.section3_dir / config.raw_data),
            "--output", str(dataset_output)
        ]
        cmd.extend(dataset_config["flags"])
        
        if run_command(cmd, f"Creating: {dataset_config['name']}"):
            datasets_created[key] = dataset_output
            log(f"Created: {dataset_config['name']}", "SUCCESS")
        else:
            log(f"✗ Failed to create: {dataset_config['name']}", "ERROR")
            return None
    
    return datasets_created

# ──────────────────────────────────────────────────────────────────────────────
# ── STEP 2: LABEL DATASETS ─────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

def label_datasets(config, datasets_created):
    """Label each dataset"""
    log("STEP 2: Labeling all datasets with multiple annotators", "INFO")
    
    labeled_datasets = {}
    
    for key, dataset_path in datasets_created.items():
        labeled_output = dataset_path.parent / f"labeled_{dataset_path.stem}.csv"
        
        # Skip if already exists
        if labeled_output.exists():
            log(f"Labeled dataset for '{key}' already exists, skipping", "WARN")
            labeled_datasets[key] = labeled_output
            continue
        
        cmd = [
            "python", "label_data.py",
            "--input", str(dataset_path),
            "--output", str(labeled_output),
            "--size", "200",
            "--score-based",
            "--rule-based"
        ]
        
        if run_command(cmd, f"Labeling: {config.datasets[key]['name']}"):
            labeled_datasets[key] = labeled_output
            log(f"Labeled: {key}", "SUCCESS")
        else:
            log(f"Failed to label: {key}", "ERROR")
            return None
    
    return labeled_datasets

# ──────────────────────────────────────────────────────────────────────────────
# ── STEP 3: CREATE TEXT REPRESENTATIONS ────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

def create_representations(config, labeled_datasets):
    """Generate TF-IDF representations for each dataset"""
    log("STEP 3: Creating text representations (TF-IDF Vectorization)", "INFO")
    
    representations = {}
    
    for key, labeled_path in labeled_datasets.items():
        rep_dir = config.section4_dir / f"representations_{key}"
        
        # Skip if already exists
        combined_file = rep_dir / "representations_combined.csv"
        if combined_file.exists():
            log(f"Representations for '{key}' already exist, skipping", "WARN")
            representations[key] = rep_dir
            continue
        
        rep_dir.mkdir(exist_ok=True)
        
        # Use TF-IDF only (more reliable, faster)
        cmd = [
            "python", "text_representation.py",
            "--input", str(labeled_path),
            "--output-dir", str(rep_dir),
            "--tfidf",
            "--max-features", "5000"
        ]
        
        if run_command(cmd, f"Representations for: {config.datasets[key]['name']}"):
            representations[key] = rep_dir
            log(f"Created representations for: {key}", "SUCCESS")
        else:
            log(f"Failed to create representations for: {key}", "ERROR")
            return None
    
    return representations

# ──────────────────────────────────────────────────────────────────────────────
# ── STEP 4: TRAIN ML MODELS WITH MIXED PIPELINES ───────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

def train_ml_models(config, labeled_datasets, representations):
    """Train ML models with different feature + model combinations"""
    log("STEP 4: Training ML models with mixed pipelines", "INFO")
    
    all_results = {}
    
    for dataset_key, labeled_path in labeled_datasets.items():
        log(f"\n--- Processing dataset: {dataset_key} ---", "INFO")
        
        all_results[dataset_key] = {}
        rep_dir = representations[dataset_key]
        features_file = rep_dir / "representations_combined.csv"
        
        for combo_idx, combo in enumerate(config.ml_combinations, 1):
            features = combo["features"]
            model = combo["model"]
            kernel = combo.get("kernel", "")
            
            # Generate combination name
            combo_name = f"{features}_{model}"
            if kernel:
                combo_name += f"_{kernel}"
            
            output_subdir = config.section4_dir / f"ml_results_{dataset_key}_{combo_name}"
            output_subdir.mkdir(parents=True, exist_ok=True)
            
            # Build ML command
            cmd = [
                "python", "ml_based_models.py",
                "--features", str(features_file),
                "--labels", str(labeled_path),
                "--output-dir", str(output_subdir),
                "--feature-type", features,
                f"--{model}"
            ]
            
            if kernel:
                cmd.extend(["--svm-kernel", kernel])
            
            description = f"ML Model ({combo_idx}/{len(config.ml_combinations)}): " \
                         f"{dataset_key} + {combo_name}"
            
            if run_command(cmd, description):
                # Read results
                report_file = output_subdir / "ml_models_report.json"
                if report_file.exists():
                    with open(report_file, "r") as f:
                        results = json.load(f)
                    all_results[dataset_key][combo_name] = {
                        "output_dir": str(output_subdir),
                        "results": results
                    }
                    log(f"{combo_name}: Trained successfully", "SUCCESS")
                else:
                    log(f"Could not find report for {combo_name}", "ERROR")
            else:
                log(f"Failed to train: {combo_name}", "ERROR")
    
    return all_results

# ──────────────────────────────────────────────────────────────────────────────
# ── STEP 5: GENERATE COMPARISON REPORT ──────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

def generate_comparison_report(config, all_results):
    """Create comprehensive comparison report"""
    log("STEP 5: Generating comprehensive comparison report", "INFO")
    
    config.output_dir.mkdir(exist_ok=True)
    
    # Flatten results for easier analysis
    comparison_data = []
    
    for dataset_key, combinations in all_results.items():
        dataset_name = config.datasets[dataset_key]["name"]
        
        for combo_name, combo_results in combinations.items():
            results = combo_results["results"]
            
            # Extract model name and metrics
            for model_name, metrics in results.items():
                row = {
                    "Dataset": dataset_name,
                    "Combination": combo_name,
                    "Model": metrics.get("model", model_name),
                    "Accuracy": metrics.get("accuracy", 0),
                    "Precision": metrics.get("precision", 0),
                    "Recall": metrics.get("recall", 0),
                    "F1-Score": metrics.get("f1_score", 0),
                }
                comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save detailed CSV
    csv_output = config.output_dir / f"comparison_all_results_{config.timestamp}.csv"
    comparison_df.to_csv(csv_output, index=False)
    log(f"Saved detailed results: {csv_output.name}", "SUCCESS")
    
    # Find best combinations
    print("\n" + "="*70)
    print("TOP 10 BEST PERFORMING COMBINATIONS")
    print("="*70)
    top_10 = comparison_df.nlargest(10, "F1-Score")[["Dataset", "Combination", "F1-Score", "Accuracy"]]
    print(top_10.to_string(index=False))
    
    # Best by dataset
    print("\n" + "="*70)
    print("BEST COMBINATION PER DATASET")
    print("="*70)
    best_by_dataset = comparison_df.loc[comparison_df.groupby("Dataset")["F1-Score"].idxmax()]
    print(best_by_dataset[["Dataset", "Combination", "F1-Score", "Accuracy"]].to_string(index=False))
    
    # Best by model type
    print("\n" + "="*70)
    print("BEST COMBINATION PER MODEL TYPE")
    print("="*70)
    best_by_model = comparison_df.loc[comparison_df.groupby("Model")["F1-Score"].idxmax()]
    print(best_by_model[["Model", "Dataset", "Combination", "F1-Score"]].to_string(index=False))
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(comparison_df[["Accuracy", "Precision", "Recall", "F1-Score"]].describe().to_string())
    
    # Save summary report
    summary = {
        "timestamp": config.timestamp,
        "total_combinations_trained": len(comparison_data),
        "datasets": list(config.datasets.keys()),
        "best_overall": {
            "dataset": best_by_dataset.iloc[0]["Dataset"],
            "combination": best_by_dataset.iloc[0]["Combination"],
            "f1_score": float(best_by_dataset.iloc[0]["F1-Score"])
        },
        "statistics": {
            "mean_f1": float(comparison_df["F1-Score"].mean()),
            "max_f1": float(comparison_df["F1-Score"].max()),
            "min_f1": float(comparison_df["F1-Score"].min()),
        }
    }
    
    summary_output = config.output_dir / f"summary_report_{config.timestamp}.json"
    with open(summary_output, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"Saved summary report: {summary_output.name}", "SUCCESS")
    
    return comparison_df, summary

# ──────────────────────────────────────────────────────────────────────────────
# ── MAIN EXECUTION ─────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sentiment Analysis Complete Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_sentiment_pipeline.py                    # Run full pipeline
  python main_sentiment_pipeline.py --step 2           # Run only step 2 onwards
        """
    )
    parser.add_argument("--step", type=int, default=1, choices=[1,2,3,4,5],
                       help="Start from step N (default: 1)")
    parser.add_argument("--skip-ml", action="store_true",
                       help="Skip ML training (for quick testing)")
    
    args = parser.parse_args()
    config = PipelineConfig()
    
    print("\n" + "="*70)
    print("SENTIMENT ANALYSIS MASTER PIPELINE")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results directory: {config.output_dir}")
    print("="*70)
    
    try:
        # Step 1: Create datasets
        if args.step <= 1:
            datasets_created = create_datasets(config)
            if not datasets_created:
                raise Exception("Failed to create datasets")
        
        # Step 2: Label datasets
        if args.step <= 2:
            labeled_datasets = label_datasets(config, datasets_created)
            if not labeled_datasets:
                raise Exception("Failed to label datasets")
        
        # Step 3: Create representations
        if args.step <= 3:
            representations = create_representations(config, labeled_datasets)
            if not representations:
                raise Exception("Failed to create representations")
        
        # Step 4: Train ML models
        if args.step <= 4 and not args.skip_ml:
            all_results = train_ml_models(config, labeled_datasets, representations)
            if not all_results:
                raise Exception("Failed to train ML models")
        
        # Step 5: Generate report
        if args.step <= 5 and not args.skip_ml:
            comparison_df, summary = generate_comparison_report(config, all_results)
        
        print("\n" + "="*70)
        print("[+] PIPELINE COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        log(f"Pipeline failed: {e}", "ERROR")
        sys.exit(1)

if __name__ == "__main__":
    main()
