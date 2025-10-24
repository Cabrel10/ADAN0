import optuna
import logging
import sys
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("optuna_analysis.log"),
    ],
)


class WorkerOptimizationAnalyzer:
    """Analyzes Optuna optimization results with focus on per-worker performance."""

    def __init__(
        self,
        storage_name: str = "sqlite:///adan_progressive_optimization.db",
        study_name: str = "adan_progressive_optimization",
    ):
        self.storage_name = storage_name
        self.study_name = study_name
        self.study = None
        self.worker_ids = ["w1", "w2", "w3", "w4"]

    def load_study(self) -> bool:
        """Load the Optuna study from database."""
        try:
            self.study = optuna.load_study(
                study_name=self.study_name, storage=self.storage_name
            )
            logging.info(f"Successfully loaded study: '{self.study_name}'")
            return True
        except (KeyError, ValueError) as e:
            logging.error(
                f"Study '{self.study_name}' not found in '{self.storage_name}': {e}"
            )
            logging.info(
                f"Please make sure the optimization script has been run and the database file exists."
            )
            return False
        except Exception as e:
            logging.error(f"Unexpected error loading study: {e}")
            return False

    def get_trial_statistics(self) -> Dict[str, int]:
        """Get basic trial statistics."""
        if not self.study:
            return {}

        all_trials = self.study.trials
        stats = {
            "total": len(all_trials),
            "completed": len(
                [t for t in all_trials if t.state == optuna.trial.TrialState.COMPLETE]
            ),
            "pruned": len(
                [t for t in all_trials if t.state == optuna.trial.TrialState.PRUNED]
            ),
            "failed": len(
                [t for t in all_trials if t.state == optuna.trial.TrialState.FAIL]
            ),
        }
        return stats

    def extract_worker_data(self, trial) -> Dict[str, Any]:
        """Extract worker-specific data from a trial."""
        worker_data = {}

        # Extract individual worker scores
        for worker_id in self.worker_ids:
            score_key = f"{worker_id}_score"
            params_key = f"{worker_id}_params"

            worker_data[worker_id] = {
                "score": trial.user_attrs.get(score_key, None),
                "params": trial.user_attrs.get(params_key, {}),
                "available": score_key in trial.user_attrs,
            }

        # Extract additional behavioral data
        worker_data["behaviors"] = trial.user_attrs.get("worker_behaviors", {})
        worker_data["worker_scores"] = trial.user_attrs.get("worker_scores", {})

        return worker_data

    def find_best_trials_per_worker(self) -> Dict[str, Dict]:
        """Find the best trial for each worker based on their individual scores."""
        if not self.study:
            return {}

        completed_trials = [
            t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]

        best_trials = {}

        for worker_id in self.worker_ids:
            best_trial = None
            best_score = -float("inf")

            for trial in completed_trials:
                score_key = f"{worker_id}_score"
                if score_key in trial.user_attrs:
                    score = trial.user_attrs[score_key]
                    if score is not None and score > best_score:
                        best_score = score
                        best_trial = trial

            if best_trial:
                best_trials[worker_id] = {
                    "trial": best_trial,
                    "score": best_score,
                    "trial_number": best_trial.number,
                    "params": best_trial.user_attrs.get(f"{worker_id}_params", {}),
                    "behavior": best_trial.user_attrs.get("worker_behaviors", {}).get(
                        worker_id, {}
                    ),
                }
            else:
                logging.warning(f"No valid trials found for worker {worker_id}")

        return best_trials

    def analyze_worker_performance_trends(self) -> Dict[str, List]:
        """Analyze performance trends across trials for each worker."""
        if not self.study:
            return {}

        completed_trials = sorted(
            [
                t
                for t in self.study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
            ],
            key=lambda x: x.number,
        )

        trends = {worker_id: [] for worker_id in self.worker_ids}

        for trial in completed_trials:
            for worker_id in self.worker_ids:
                score_key = f"{worker_id}_score"
                if score_key in trial.user_attrs:
                    score = trial.user_attrs[score_key]
                    trends[worker_id].append(
                        {
                            "trial_number": trial.number,
                            "score": score,
                            "timestamp": trial.datetime_complete,
                        }
                    )

        return trends

    def generate_worker_report(self) -> str:
        """Generate a comprehensive report for worker optimization results."""
        if not self.study:
            return "❌ No study loaded"

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("🎯 WORKER-BASED OPTIMIZATION ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        report_lines.append(f"Study: {self.study_name}")
        report_lines.append("")

        # Basic statistics
        stats = self.get_trial_statistics()
        report_lines.append("📊 STUDY STATISTICS")
        report_lines.append("-" * 40)
        report_lines.append(f"Total trials: {stats['total']}")
        report_lines.append(f"✅ Completed: {stats['completed']}")
        report_lines.append(f"✂️ Pruned: {stats['pruned']}")
        report_lines.append(f"❌ Failed: {stats['failed']}")
        report_lines.append("")

        if stats["completed"] == 0:
            report_lines.append("⚠️ No completed trials to analyze.")
            return "\n".join(report_lines)

        # Best trials per worker
        best_trials = self.find_best_trials_per_worker()

        report_lines.append("🏆 BEST PERFORMANCE PER WORKER")
        report_lines.append("=" * 50)

        for worker_id in self.worker_ids:
            if worker_id in best_trials:
                trial_data = best_trials[worker_id]
                report_lines.append(f"\n🤖 {worker_id.upper()} - Best Performance")
                report_lines.append("-" * 30)
                report_lines.append(f"Trial Number: {trial_data['trial_number']}")
                report_lines.append(f"Score: {trial_data['score']:.4f}")

                # Display hyperparameters
                params = trial_data["params"]
                if params:
                    report_lines.append("\n📋 Optimal Hyperparameters:")
                    for param, value in params.items():
                        if isinstance(value, float):
                            report_lines.append(f"  • {param}: {value:.6f}")
                        else:
                            report_lines.append(f"  • {param}: {value}")

                # Display behavior metrics
                behavior = trial_data["behavior"]
                if behavior:
                    report_lines.append("\n📈 Performance Metrics:")
                    metrics = [
                        ("Total Trades", behavior.get("total_trades", "N/A")),
                        (
                            "Win Rate",
                            f"{behavior.get('win_rate', 0):.1%}"
                            if "win_rate" in behavior
                            else "N/A",
                        ),
                        (
                            "Total PnL",
                            f"${behavior.get('total_pnl', 0):.2f}"
                            if "total_pnl" in behavior
                            else "N/A",
                        ),
                        (
                            "Portfolio Growth",
                            f"{behavior.get('portfolio_growth', 0):.2%}"
                            if "portfolio_growth" in behavior
                            else "N/A",
                        ),
                        ("TF Diversity", behavior.get("tf_diversity", "N/A")),
                    ]

                    for metric, value in metrics:
                        report_lines.append(f"  • {metric}: {value}")

                    # Timeframe distribution
                    tf_trades = behavior.get("timeframe_trades", {})
                    if tf_trades:
                        report_lines.append("\n⏰ Timeframe Distribution:")
                        for tf, count in tf_trades.items():
                            report_lines.append(f"  • {tf}: {count} trades")

            else:
                report_lines.append(f"\n🤖 {worker_id.upper()} - No valid trials found")

        # Performance comparison
        if len(best_trials) > 1:
            report_lines.append("\n" + "=" * 50)
            report_lines.append("⚖️ WORKER PERFORMANCE COMPARISON")
            report_lines.append("-" * 35)

            # Sort workers by performance
            sorted_workers = sorted(
                best_trials.items(), key=lambda x: x[1]["score"], reverse=True
            )

            report_lines.append("\n🥇 Performance Ranking:")
            for rank, (worker_id, data) in enumerate(sorted_workers, 1):
                medal = (
                    "🥇"
                    if rank == 1
                    else "🥈"
                    if rank == 2
                    else "🥉"
                    if rank == 3
                    else "📊"
                )
                report_lines.append(
                    f"  {medal} {rank}. {worker_id.upper()}: {data['score']:.4f}"
                )

            # Statistical analysis
            scores = [data["score"] for data in best_trials.values()]
            report_lines.append(f"\n📊 Score Statistics:")
            report_lines.append(f"  • Best: {max(scores):.4f}")
            report_lines.append(f"  • Worst: {min(scores):.4f}")
            report_lines.append(f"  • Average: {np.mean(scores):.4f}")
            report_lines.append(f"  • Std Dev: {np.std(scores):.4f}")

        # Recommendations
        report_lines.append("\n" + "=" * 50)
        report_lines.append("💡 OPTIMIZATION RECOMMENDATIONS")
        report_lines.append("-" * 35)

        if best_trials:
            best_worker = max(best_trials.items(), key=lambda x: x[1]["score"])
            report_lines.append(
                f"\n🎯 Best Overall: {best_worker[0].upper()} (Score: {best_worker[1]['score']:.4f})"
            )

            # Analyze if any workers need attention
            avg_score = np.mean([data["score"] for data in best_trials.values()])
            underperforming = [
                worker_id
                for worker_id, data in best_trials.items()
                if data["score"] < avg_score * 0.8
            ]

            if underperforming:
                report_lines.append(
                    f"\n⚠️ Workers needing attention: {', '.join(underperforming)}"
                )

            # Check diversity
            all_behaviors = [
                data["behavior"] for data in best_trials.values() if data["behavior"]
            ]
            if all_behaviors:
                avg_diversity = np.mean(
                    [b.get("tf_diversity", 0) for b in all_behaviors]
                )
                if avg_diversity < 2:
                    report_lines.append(
                        "📊 Consider increasing timeframe diversification"
                    )

                avg_trades = np.mean([b.get("total_trades", 0) for b in all_behaviors])
                if avg_trades < 5:
                    report_lines.append(
                        "📊 Consider adjusting thresholds to increase trade frequency"
                    )

        return "\n".join(report_lines)

    def export_best_configs(self, output_file: str = None) -> Dict:
        """Export the best configurations for each worker to JSON."""
        if not self.study:
            return {}

        best_trials = self.find_best_trials_per_worker()

        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "study_name": self.study_name,
            "workers": {},
        }

        for worker_id, trial_data in best_trials.items():
            export_data["workers"][worker_id] = {
                "best_trial_number": trial_data["trial_number"],
                "best_score": trial_data["score"],
                "optimal_params": trial_data["params"],
                "performance_metrics": trial_data["behavior"],
            }

        if output_file:
            with open(output_file, "w") as f:
                json.dump(export_data, f, indent=2, default=str)
            logging.info(f"Best configurations exported to: {output_file}")

        return export_data

    def generate_visualization_plots(self):
        """Generate visualization plots for worker performance analysis."""
        if not self.study:
            return

        try:
            import optuna.visualization as vis
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            completed_trials = [
                t
                for t in self.study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
            ]

            # 1. Overall optimization history
            fig_history = vis.plot_optimization_history(self.study)
            fig_history.write_html("optuna_history.html")
            logging.info("✅ Generated optimization history plot: optuna_history.html")

            # 2. Per-worker performance evolution
            worker_scores_data = {worker_id: [] for worker_id in self.worker_ids}
            trial_numbers = []

            for trial in completed_trials:
                trial_numbers.append(trial.number)
                for worker_id in self.worker_ids:
                    score = trial.user_attrs.get(f"{worker_id}_score", None)
                    worker_scores_data[worker_id].append(score)

            if trial_numbers and any(worker_scores_data.values()):
                fig = make_subplots(
                    rows=2,
                    cols=2,
                    subplot_titles=[f"Worker {w.upper()}" for w in self.worker_ids],
                    vertical_spacing=0.1,
                )

                positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
                colors = ["blue", "red", "green", "orange"]

                for i, (worker_id, color) in enumerate(zip(self.worker_ids, colors)):
                    row, col = positions[i]
                    scores = worker_scores_data[worker_id]
                    valid_scores = [
                        (tn, s) for tn, s in zip(trial_numbers, scores) if s is not None
                    ]

                    if valid_scores:
                        valid_trials, valid_score_vals = zip(*valid_scores)
                        fig.add_trace(
                            go.Scatter(
                                x=valid_trials,
                                y=valid_score_vals,
                                mode="lines+markers",
                                name=f"{worker_id.upper()}",
                                line=dict(color=color),
                            ),
                            row=row,
                            col=col,
                        )

                fig.update_layout(
                    title="Worker Performance Evolution", showlegend=False, height=600
                )
                fig.write_html("worker_performance_evolution.html")
                logging.info(
                    "✅ Generated worker performance plot: worker_performance_evolution.html"
                )

            # 3. Parameter importance (if available)
            try:
                fig_importance = vis.plot_param_importances(self.study)
                fig_importance.write_html("optuna_importance.html")
                logging.info(
                    "✅ Generated parameter importance plot: optuna_importance.html"
                )
            except Exception as e:
                logging.warning(f"Could not generate parameter importance plot: {e}")

        except ImportError:
            logging.warning(
                "Could not generate plots. Please install plotly and kaleido: `pip install plotly kaleido`"
            )
        except Exception as e:
            logging.error(f"Error generating plots: {e}")


def show_worker_results():
    """Main function to analyze and display worker-based optimization results."""
    analyzer = WorkerOptimizationAnalyzer()

    if not analyzer.load_study():
        return

    # Generate comprehensive report
    print(analyzer.generate_worker_report())

    # Export best configurations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_file = f"optimal_worker_configs_{timestamp}.json"
    analyzer.export_best_configs(config_file)

    # Generate visualizations
    analyzer.generate_visualization_plots()

    print("\n" + "=" * 60)
    print("📁 FILES GENERATED:")
    print(f"  • Configuration: {config_file}")
    print(f"  • Optimization History: optuna_history.html")
    print(f"  • Worker Performance: worker_performance_evolution.html")
    print(f"  • Parameter Importance: optuna_importance.html (if available)")
    print(f"  • Analysis Log: optuna_analysis.log")
    print("=" * 60)


if __name__ == "__main__":
    show_worker_results()
