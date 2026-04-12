#!/usr/bin/env python3
"""
Complete Training Pipeline - Trains all models (SVM, Logistic Regression, Decision Tree, Random Forest)
"""

import subprocess
import sys
from pathlib import Path
import json

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*70}")
    print(f"🔧 {description}")
    print(f"{'='*70}\n")
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent.parent / "section4", check=True, capture_output=False)
        print(f"\n✅ {description} - SUCCESS\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} - FAILED")
        print(f"Error: {e}\n")
        return False

def main():
    print("\n" + "="*70)
    print("🚀 SENTIMENT ANALYSIS - COMPLETE MODEL TRAINING PIPELINE")
    print("="*70 + "\n")
    
    print("📋 This pipeline will train:")
    print("   1. SVM (Support Vector Machine) - RBF Kernel")
    print("   2. Logistic Regression - MultiClass")
    print("   3. Decision Tree - Max Depth 10")
    print("   4. Random Forest - 100 Trees")
    print()
    
    # Check if data exists
    features_file = Path(__file__).parent.parent / "section4" / "representations_full" / "features.csv"
    labels_file = Path(__file__).parent.parent / "section4" / "labeled_reviews.csv"
    
    if not features_file.exists():
        print(f"❌ Features file not found: {features_file}")
        print("Please run text_representation.py first")
        return False
    
    if not labels_file.exists():
        print(f"❌ Labels file not found: {labels_file}")
        print("Please run label_data.py first")
        return False
    
    print(f"✅ Found features: {features_file}")
    print(f"✅ Found labels: {labels_file}\n")
    
    # Train all models in one command
    cmd = [
        sys.executable, 
        "ml_based_models.py",
        "--features", str(features_file),
        "--labels", str(labels_file),
        "--output-dir", "ml_results_full_tfidf_all_models",
        "--feature-type", "tfidf",
        "--svm",
        "--logistic",
        "--decision-tree",
        "--random-forest",
        "--svm-kernel", "rbf",
        "--svm-c", "1.0",
        "--dt-depth", "10",
        "--rf-trees", "100"
    ]
    
    if not run_command(cmd, "Training All Models"):
        return False
    
    print("\n" + "="*70)
    print("✅ ALL MODELS TRAINED SUCCESSFULLY!")
    print("="*70)
    print("\nModels trained and saved to: ml_results_full_tfidf_all_models/")
    print("\nNext steps:")
    print("  1. Run the Streamlit app: streamlit run streamlit_app.py")
    print("  2. or update model_loader.py to use these new models")
    print()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
