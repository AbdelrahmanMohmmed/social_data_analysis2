"""
Task 4 - Complete Orchestrator
Runs Model Evaluation, Optimization, Error Analysis, and Deployment
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# Import all modules
from model_evaluation import ComprehensiveModelEvaluator
from error_analysis import ErrorAnalyzer
from hyperparameter_tuning import HyperparameterOptimizer
from model_comparison import ModelComparison


class Task4Orchestrator:
    """Orchestrate complete Task 4: Model Optimization and Deployment"""
    
    def __init__(self, section4_dir: str = "../section4", output_dir: str = "task4_results"):
        """
        Initialize orchestrator
        
        Args:
            section4_dir: Path to section4 with trained models
            output_dir: Output directory for results
        """
        self.section4_dir = section4_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.start_time = datetime.now()
        self.results = {}
    
    def print_header(self, title: str):
        """Print formatted header"""
        print("\n" + "="*80)
        print(f"  {title}")
        print("="*80 + "\n")
    
    # ──────────────────────────────────────────────────────────────────────────
    # ── STEP 1: MODEL EVALUATION ───────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────────────────
    
    def step1_evaluate_models(self):
        """
        Step 1: Evaluate and benchmark all 12 models
        
        Outputs:
        - Model evaluation results (JSON + CSV)
        - Comparison visualization
        - Identifies best models
        """
        self.print_header("STEP 1: MODEL EVALUATION & BENCHMARKING")
        
        try:
            evaluator = ComprehensiveModelEvaluator(self.section4_dir)
            
            # Evaluate all models
            results_df = evaluator.evaluate_all_models()
            
            # Print summary
            evaluator.print_summary()
            
            # Save results
            results_csv = self.output_dir / "model_evaluation_results.csv"
            results_df.to_csv(results_csv, index=False)
            
            results_json = self.output_dir / "model_evaluation_results.json"
            evaluator.save_results(str(results_json))
            
            # Plot
            plot_file = self.output_dir / "model_comparison.png"
            evaluator.plot_comparison(str(plot_file))
            
            self.results['evaluation'] = {
                'status': 'completed',
                'best_models': results_df.nlargest(3, 'f1_weighted').to_dict('records'),
                'output_files': [str(results_csv), str(results_json), str(plot_file)]
            }
            
            print(f"\n✓ Step 1 complete!")
            print(f"  Results saved to: {self.output_dir}")
            
            return True
            
        except Exception as e:
            print(f"\n✗ Step 1 failed: {str(e)}")
            self.results['evaluation'] = {'status': 'failed', 'error': str(e)}
            return False
    
    # ──────────────────────────────────────────────────────────────────────────
    # ── STEP 2: ERROR ANALYSIS ─────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────────────────
    
    def step2_analyze_errors(self):
        """
        Step 2: Analyze model errors and failed cases
        
        Outputs:
        - Error analysis report (JSON)
        - Failed cases analysis
        - Error visualization
        - Conclusions on model failure patterns
        """
        self.print_header("STEP 2: ERROR ANALYSIS")
        
        try:
            analyzer = ErrorAnalyzer(self.section4_dir)
            
            # Load model and identify failures
            model_data = analyzer.load_best_model()
            failed_cases = analyzer.identify_failed_cases(model_data)
            
            # Analyze patterns
            analysis = analyzer.analyze_error_patterns(failed_cases)
            
            # Print report
            analyzer.print_analysis_report()
            
            # Save results
            report_file = self.output_dir / "error_analysis_report.json"
            analyzer.save_report(str(report_file))
            
            # Visualization
            viz_file = self.output_dir / "error_analysis_visualization.png"
            analyzer.plot_error_analysis(str(viz_file))
            
            self.results['error_analysis'] = {
                'status': 'completed',
                'total_errors': len(failed_cases),
                'error_rate': analysis.get('error_rate'),
                'main_issues': list(analysis.get('error_categories', {}).keys()),
                'output_files': [str(report_file), str(viz_file)]
            }
            
            print(f"\n✓ Step 2 complete!")
            
            return True
            
        except Exception as e:
            print(f"\n✗ Step 2 failed: {str(e)}")
            self.results['error_analysis'] = {'status': 'failed', 'error': str(e)}
            return False
    
    # ──────────────────────────────────────────────────────────────────────────
    # ── STEP 3: HYPERPARAMETER OPTIMIZATION ────────────────────────────────────
    # ──────────────────────────────────────────────────────────────────────────
    
    def step3_optimize_models(self):
        """
        Step 3: Optimize models using Grid Search, Random Search, Genetic Algorithm
        
        Outputs:
        - Optimized model (pickle file)
        - Optimization results (CSV)
        - Comparison of different optimization methods
        """
        self.print_header("STEP 3: HYPERPARAMETER OPTIMIZATION")
        
        try:
            optimizer = HyperparameterOptimizer(self.section4_dir)
            
            # Run optimization
            eval_df = optimizer.run_optimization()
            
            # Save results
            opt_results_file = self.output_dir / "optimization_results.csv"
            eval_df.drop('model', axis=1).to_csv(opt_results_file, index=False)
            
            self.results['optimization'] = {
                'status': 'completed',
                'best_method': eval_df.iloc[eval_df['f1_score'].idxmax()]['method'],
                'best_f1_score': float(eval_df['f1_score'].max()),
                'output_files': [str(opt_results_file), "optimized_svm_model.pkl"]
            }
            
            print(f"\n✓ Step 3 complete!")
            
            return True
            
        except Exception as e:
            print(f"\n✗ Step 3 failed: {str(e)}")
            self.results['optimization'] = {'status': 'failed', 'error': str(e)}
            return False
    
    # ──────────────────────────────────────────────────────────────────────────
    # ── STEP 4: MODEL COMPARISON ───────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────────────────
    
    def step4_compare_models(self):
        """
        Step 4: Compare baseline vs optimized models
        
        Outputs:
        - Comparison report (JSON)
        - Before/after metrics
        - Visualization
        - Check if 20% improvement threshold met
        """
        self.print_header("STEP 4: MODEL COMPARISON (BEFORE vs AFTER)")
        
        try:
            comparator = ModelComparison(self.section4_dir)
            
            # Compare models
            comparison_df = comparator.compare_models()
            
            if comparison_df is not None:
                # Check optimization requirement
                comparator.print_optimization_check()
                
                # Save results
                report_file = self.output_dir / "model_comparison_report.json"
                comparator.save_comparison_report(str(report_file))
                
                # visualization
                viz_file = self.output_dir / "model_comparison.png"
                comparator.plot_comparison(str(viz_file))
                
                improvements = comparator.improvements
                random_f1, improvement_target = comparator.check_random_baseline()
                optimized_f1 = comparison_df.iloc[1]['f1_score'] if len(comparison_df) > 1 else 0
                
                self.results['comparison'] = {
                    'status': 'completed',
                    'baseline_f1': float(comparison_df.iloc[0]['f1_score']),
                    'optimized_f1': float(optimized_f1),
                    'f1_improvement': float(improvements.get('f1_score', {}).get('improvement', 0)),
                    'random_baseline': float(random_f1),
                    'improvement_threshold_met': optimized_f1 >= improvement_target,
                    'output_files': [str(report_file), str(viz_file)]
                }
                
                print(f"\n✓ Step 4 complete!")
                
                return True
            else:
                return False
            
        except Exception as e:
            print(f"\n✗ Step 4 failed: {str(e)}")
            self.results['comparison'] = {'status': 'failed', 'error': str(e)}
            return False
    
    # ──────────────────────────────────────────────────────────────────────────
    # ── DEPLOYMENT INFO ────────────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────────────────
    
    def step5_deployment_ready(self):
        """
        Step 5: Confirm deployment readiness
        
        Deployment already implemented in:
        - flask_app.py (REST API)
        - streamlit_app.py (Interactive UI)
        """
        self.print_header("STEP 5: DEPLOYMENT READY")
        
        print("✓ Deployment Infrastructure Already Built:\n")
        print("  1. Flask REST API (flask_app.py)")
        print("     - Endpoint: POST /predict")
        print("     - Request:  {'text': 'sample text'}")
        print("     - Response: {'sentiment': '...', 'confidence': ...}\n")
        
        print("  2. Streamlit Dashboard (streamlit_app.py)")
        print("     - Interactive UI for predictions")
        print("     - Batch processing capability")
        print("     - Model information display\n")
        
        print("  3. API Client (api_client.py)")
        print("     - Python wrapper for API integration")
        print("     - CSV file processing support")
        print("     - Batch prediction capability\n")
        
        print("To deploy optimized model:")
        print("  1. Copy optimized_svm_model.pkl to appropriate directory")
        print("  2. Update model_loader.py to use optimized model")
        print("  3. Run: python flask_app.py (for API)")
        print("  4. Run: streamlit run streamlit_app.py (for UI)\n")
        
        self.results['deployment'] = {
            'status': 'ready',
            'components': ['Flask API', 'Streamlit UI', 'API Client']
        }
        
        return True
    
    # ──────────────────────────────────────────────────────────────────────────
    # ── ORCHESTRATION ──────────────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────────────────
    
    def run_all_steps(self, skip_steps: List[int] = None):
        """
        Run all steps in sequence
        
        Args:
            skip_steps: List of step numbers to skip (1-5)
        """
        if skip_steps is None:
            skip_steps = []
        
        print("\n" + "="*80)
        print("TASK 4: MODEL OPTIMIZATION AND DEPLOYMENT")
        print("="*80)
        print(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        steps = [
            (1, "Model Evaluation & Benchmarking", self.step1_evaluate_models),
            (2, "Error Analysis", self.step2_analyze_errors),
            (3, "Hyperparameter Optimization", self.step3_optimize_models),
            (4, "Model Comparison", self.step4_compare_models),
            (5, "Deployment Ready", self.step5_deployment_ready),
        ]
        
        completed = 0
        for step_num, step_name, step_func in steps:
            if step_num in skip_steps:
                print(f"\n[SKIPPED] Step {step_num}: {step_name}")
                continue
            
            success = step_func()
            if success:
                completed += 1
            else:
                print(f"\n⚠ Step {step_num} encountered issues but continuing...")
        
        # Summary
        self._print_summary(completed, len(steps) - len(skip_steps))
    
    def _print_summary(self, completed: int, total: int):
        """Print execution summary"""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        print("\n" + "="*80)
        print("TASK 4 SUMMARY")
        print("="*80 + "\n")
        
        print(f"Execution Time: {duration.total_seconds():.2f} seconds")
        print(f"Steps Completed: {completed}/{total}")
        print(f"Output Directory: {self.output_dir}\n")
        
        print("Results:")
        for step_name, step_results in self.results.items():
            status = "✓" if step_results.get('status') == 'completed' else "✗"
            print(f"  {status} {step_name.replace('_', ' ').title()}: {step_results.get('status')}")
            
            if 'output_files' in step_results:
                for f in step_results.get('output_files', []):
                    print(f"     → {f}")
        
        print("\n" + "="*80)
        print("✓ Task 4 Complete!")
        print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Task 4: Model Optimization and Deployment")
    parser.add_argument('--section4', default='../section4', 
                       help='Path to section4 directory')
    parser.add_argument('--output', default='task4_results',
                       help='Output directory for results')
    parser.add_argument('--skip', nargs='+', type=int, default=[],
                       help='Steps to skip (1-5)')
    parser.add_argument('--step', type=int, default=None,
                       help='Run only specific step (1-5)')
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = Task4Orchestrator(args.section4, args.output)
    
    # Run
    if args.step:
        if args.step == 1:
            orchestrator.step1_evaluate_models()
        elif args.step == 2:
            orchestrator.step2_analyze_errors()
        elif args.step == 3:
            orchestrator.step3_optimize_models()
        elif args.step == 4:
            orchestrator.step4_compare_models()
        elif args.step == 5:
            orchestrator.step5_deployment_ready()
    else:
        orchestrator.run_all_steps(args.skip)


if __name__ == "__main__":
    main()
