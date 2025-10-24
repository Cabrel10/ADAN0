#!/usr/bin/env python3
"""
Advanced Optuna Results Analyzer - Per Worker Optimization Results

This script analyzes Optuna study results and extracts the best hyperparameters
for each worker individually, providing detailed performance analysis.
"""

import optuna
import logging
import sys
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class WorkerOptimizationAnalyzer:
    """Analyzes Optuna results to find best hyperparameters per worker."""

    def __init__(self, storage_name: str, study_name: str):
        self.storage_name = storage_name
        self.study_name = study_name
        self.study = None
        self.worker_ids = ["w1", "w2", "w3", "w4"]
        self.worker_names = {
            "w1": "Conservative Worker",
            "w2": "Moderate Worker",
            "w3": "Aggressive Worker",
            "w4": "Adaptive Worker",
        }

    def load_study(self) -> bool:
        """Load the Optuna study from database."""
        try:
            self.study = optuna.load_study(
                study_name=self.study_name, storage=self.storage_name
            )
            logger.info(f"✅ Successfully loaded study: '{self.study_name}'")
            return True
        except (KeyError, ValueError) as e:
            logger.error(
                f"❌ Study '{self.study_name}' not found in '{self.storage_name}'"
            )
            logger.info("Please ensure you have run 'optimize_hyperparams.py' first.")
            return False

    def analyze_trials(self) -> Dict[str, Any]:
        """Analyze all trials and categorize them."""
        if not self.study:
            return {}

        all_trials = self.study.trials
        completed_trials = [
            t for t in all_trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        pruned_trials = [
            t for t in all_trials if t.state == optuna.trial.TrialState.PRUNED
        ]
        failed_trials = [
            t for t in all_trials if t.state == optuna.trial.TrialState.FAIL
        ]

        return {
            "total": len(all_trials),
            "completed": completed_trials,
            "pruned": len(pruned_trials),
            "failed": len(failed_trials),
        }

    def find_best_worker_trials(self, completed_trials: List) -> Dict[str, Dict]:
        """Find the best trial for each worker individually."""
        best_workers = {}

        for worker_id in self.worker_ids:
            best_score = -float("inf")
            best_trial = None

            for trial in completed_trials:
                # Try to get individual worker score
                worker_score_key = f"{worker_id}_score"
                worker_score = trial.user_attrs.get(worker_score_key, None)

                if worker_score is not None and worker_score > best_score:
                    best_score = worker_score
                    best_trial = trial

            if best_trial:
                best_workers[worker_id] = {
                    "trial": best_trial,
                    "score": best_score,
                    "trial_number": best_trial.number,
                }
            else:
                logger.warning(f"⚠️  No valid trials found for {worker_id}")

        return best_workers

    def extract_worker_params(self, trial, worker_id: str) -> Dict[str, Any]:
        """Extract hyperparameters for a specific worker from a trial."""
        params = {}

        # Try to get saved worker-specific params first
        worker_params_key = f"{worker_id}_params"
        if worker_params_key in trial.user_attrs:
            return trial.user_attrs[worker_params_key]

        # Fallback: extract from trial.params using prefixes
        prefix = f"{worker_id}_"
        for param_name, value in trial.params.items():
            if param_name.startswith(prefix):
                clean_name = param_name[len(prefix) :]  # Remove prefix
                params[clean_name] = value

        return params

    def get_worker_behavior(self, trial, worker_id: str) -> Dict[str, Any]:
        """Extract behavior analysis for a specific worker."""
        worker_behaviors = trial.user_attrs.get("worker_behaviors", {})
        return worker_behaviors.get(worker_id, {})

    def print_study_summary(self, analysis: Dict[str, Any]):
        """Print overall study summary."""
        print("\n" + "=" * 80)
        print("🎯 OPTUNA STUDY ANALYSIS - BEST HYPERPARAMETERS PER WORKER")
        print("=" * 80)
        print(f"📊 Study: {self.study_name}")
        print(f"💾 Storage: {self.storage_name}")
        print(f"📈 Total trials: {analysis['total']}")
        print(f"   ✅ Completed: {len(analysis['completed'])}")
        print(f"   ✂️  Pruned: {analysis['pruned']}")
        print(f"   ❌ Failed: {analysis['failed']}")

    def print_worker_results(self, best_workers: Dict[str, Dict]):
        """Print detailed results for each worker."""

        for worker_id in self.worker_ids:
            print(f"\n{'=' * 60}")
            print(f"🤖 {self.worker_names[worker_id].upper()} ({worker_id.upper()})")
            print("=" * 60)

            if worker_id not in best_workers:
                print("❌ No successful trials found for this worker")
                continue

            worker_data = best_workers[worker_id]
            trial = worker_data["trial"]
            score = worker_data["score"]

            print(f"🏆 Best Trial: #{trial.number}")
            print(f"📊 Individual Score: {score:.4f}")
            print(f"🌐 Global Trial Score: {trial.value:.4f}")

            # Duration if available
            if "duration_minutes" in trial.user_attrs:
                duration = trial.user_attrs["duration_minutes"]
                print(f"⏱️  Duration: {duration:.1f} minutes")

            # Worker-specific parameters
            worker_params = self.extract_worker_params(trial, worker_id)
            if worker_params:
                print(f"\n🔧 OPTIMAL HYPERPARAMETERS:")
                for param_name, value in worker_params.items():
                    if isinstance(value, float):
                        print(f"   • {param_name}: {value:.6f}")
                    else:
                        print(f"   • {param_name}: {value}")

            # Behavior analysis if available
            behavior = self.get_worker_behavior(trial, worker_id)
            if behavior:
                print(f"\n📈 PERFORMANCE METRICS:")

                if "total_trades" in behavior:
                    print(f"   • Total Trades: {behavior['total_trades']}")

                if "win_rate" in behavior:
                    print(f"   • Win Rate: {behavior['win_rate']:.1%}")

                if "total_pnl" in behavior:
                    print(f"   • Total PnL: ${behavior['total_pnl']:.2f}")

                if "portfolio_growth" in behavior:
                    print(f"   • Portfolio Growth: {behavior['portfolio_growth']:.2%}")

                if "timeframe_trades" in behavior:
                    tf_trades = behavior["timeframe_trades"]
                    print(f"   • Timeframe Distribution:")
                    for tf, count in tf_trades.items():
                        print(f"     - {tf}: {count} trades")

                if "tf_diversity" in behavior:
                    print(
                        f"   • Timeframe Diversity: {behavior['tf_diversity']}/3 used"
                    )

    def print_global_best(self, completed_trials: List):
        """Print the best global trial for comparison."""
        if not completed_trials:
            return

        best_global = max(completed_trials, key=lambda t: t.value)

        print(f"\n{'=' * 60}")
        print("🌟 BEST GLOBAL TRIAL (FOR REFERENCE)")
        print("=" * 60)
        print(f"🏆 Trial: #{best_global.number}")
        print(f"📊 Global Score: {best_global.value:.4f}")

        # Show individual worker scores from this trial
        print(f"\n📊 Individual Worker Scores in Best Global Trial:")
        for worker_id in self.worker_ids:
            worker_score_key = f"{worker_id}_score"
            worker_score = best_global.user_attrs.get(worker_score_key, "N/A")
            if worker_score != "N/A":
                print(f"   • {worker_id.upper()}: {worker_score:.4f}")
            else:
                print(f"   • {worker_id.upper()}: N/A")

    def generate_summary_table(self, best_workers: Dict[str, Dict]):
        """Generate a summary table of all workers."""
        print(f"\n{'=' * 80}")
        print("📋 SUMMARY TABLE - BEST SCORES PER WORKER")
        print("=" * 80)

        print(f"{'Worker':<15} {'Trial#':<8} {'Score':<10} {'Params Preview':<45}")
        print("-" * 80)

        for worker_id in self.worker_ids:
            if worker_id in best_workers:
                data = best_workers[worker_id]
                trial = data["trial"]
                score = data["score"]

                # Get a few key parameters for preview
                params = self.extract_worker_params(trial, worker_id)
                param_preview = []
                for key, value in list(params.items())[:3]:  # Show first 3 params
                    if isinstance(value, float):
                        param_preview.append(f"{key}={value:.3f}")
                    else:
                        param_preview.append(f"{key}={value}")

                preview_str = ", ".join(param_preview)
                if len(params) > 3:
                    preview_str += f", +{len(params) - 3} more"

                print(
                    f"{self.worker_names[worker_id]:<15} #{data['trial_number']:<7} {score:<10.4f} {preview_str:<45}"
                )
            else:
                print(
                    f"{self.worker_names[worker_id]:<15} {'N/A':<8} {'N/A':<10} {'No successful trials':<45}"
                )

    def export_results(
        self, best_workers: Dict[str, Dict], filename: str = "best_worker_params.json"
    ):
        """Export results to JSON file."""
        try:
            import json

            export_data = {}
            for worker_id in self.worker_ids:
                if worker_id in best_workers:
                    data = best_workers[worker_id]
                    trial = data["trial"]

                    export_data[worker_id] = {
                        "worker_name": self.worker_names[worker_id],
                        "best_trial_number": data["trial_number"],
                        "best_score": data["score"],
                        "global_score": trial.value,
                        "parameters": self.extract_worker_params(trial, worker_id),
                        "behavior": self.get_worker_behavior(trial, worker_id),
                    }

            with open(filename, "w") as f:
                json.dump(export_data, f, indent=2, default=str)

            print(f"\n💾 Results exported to: {filename}")

        except ImportError:
            logger.warning("⚠️  JSON export failed - json module not available")
        except Exception as e:
            logger.error(f"❌ Export failed: {e}")

    def run_analysis(self):
        """Run the complete analysis."""
        print("🔍 Starting Optuna Results Analysis...")

        # Load study
        if not self.load_study():
            return

        # Analyze trials
        analysis = self.analyze_trials()
        if not analysis["completed"]:
            logger.error("❌ No completed trials found!")
            return

        # Print study summary
        self.print_study_summary(analysis)

        # Find best trials per worker
        best_workers = self.find_best_worker_trials(analysis["completed"])

        if not best_workers:
            logger.error("❌ No worker-specific data found!")
            logger.info("💡 Make sure you're using the updated optimize_hyperparams.py")
            return

        # Print detailed results
        self.print_worker_results(best_workers)

        # Print global best for reference
        self.print_global_best(analysis["completed"])

        # Generate summary table
        self.generate_summary_table(best_workers)

        # Export results
        self.export_results(best_workers)

        print(
            f"\n✅ Analysis complete! Found optimal parameters for {len(best_workers)} workers."
        )


def main():
    """Main function."""
    storage_name = "sqlite:///adan_progressive_optimization.db"
    study_name = "adan_progressive_optimization"

    analyzer = WorkerOptimizationAnalyzer(storage_name, study_name)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
