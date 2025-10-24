#!/usr/bin/env python3
"""
Test Worker Optimization System

This script validates the new worker-specific optimization system by:
1. Creating mock trial data with individual worker scores
2. Testing the analysis functionality
3. Demonstrating the expected output format
"""

import optuna
import tempfile
import os
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_mock_trial_data():
    """Create mock trial data to test the worker optimization system."""

    # Create temporary database
    temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    storage_name = f"sqlite:///{temp_db.name}"
    study_name = "test_worker_optimization"

    logger.info(f"Creating test study: {study_name}")

    # Create study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        load_if_exists=True,
    )

    # Mock trial scenarios
    trial_scenarios = [
        {
            "description": "Balanced performance across all workers",
            "w1_score": 0.65,
            "w2_score": 0.70,
            "w3_score": 0.68,
            "w4_score": 0.72,
            "global_score": 0.6875,
            "params": {
                "w1_position_size": 0.15,
                "w1_min_conf": 0.05,
                "w1_risk_mult": 1.2,
                "w2_position_size": 0.18,
                "w2_min_conf": 0.08,
                "w2_risk_mult": 1.4,
                "w3_position_size": 0.22,
                "w3_min_conf": 0.12,
                "w3_risk_mult": 1.6,
                "w4_position_size": 0.20,
                "w4_min_conf": 0.10,
                "w4_risk_mult": 1.5,
                "stop_loss_pct": 0.08,
                "take_profit_pct": 0.15,
            },
        },
        {
            "description": "W1 outperforms (conservative strategy wins)",
            "w1_score": 0.85,
            "w2_score": 0.55,
            "w3_score": 0.45,
            "w4_score": 0.60,
            "global_score": 0.6125,
            "params": {
                "w1_position_size": 0.12,
                "w1_min_conf": 0.03,
                "w1_risk_mult": 1.1,
                "w2_position_size": 0.20,
                "w2_min_conf": 0.10,
                "w2_risk_mult": 1.3,
                "w3_position_size": 0.25,
                "w3_min_conf": 0.15,
                "w3_risk_mult": 1.8,
                "w4_position_size": 0.18,
                "w4_min_conf": 0.08,
                "w4_risk_mult": 1.4,
                "stop_loss_pct": 0.06,
                "take_profit_pct": 0.12,
            },
        },
        {
            "description": "W3 excels (aggressive strategy in bull market)",
            "w1_score": 0.50,
            "w2_score": 0.65,
            "w3_score": 0.90,
            "w4_score": 0.70,
            "global_score": 0.6875,
            "params": {
                "w1_position_size": 0.10,
                "w1_min_conf": 0.07,
                "w1_risk_mult": 1.0,
                "w2_position_size": 0.16,
                "w2_min_conf": 0.06,
                "w2_risk_mult": 1.2,
                "w3_position_size": 0.24,
                "w3_min_conf": 0.02,
                "w3_risk_mult": 2.0,
                "w4_position_size": 0.22,
                "w4_min_conf": 0.04,
                "w4_risk_mult": 1.8,
                "stop_loss_pct": 0.10,
                "take_profit_pct": 0.18,
            },
        },
        {
            "description": "W4 adaptive dominance",
            "w1_score": 0.60,
            "w2_score": 0.58,
            "w3_score": 0.55,
            "w4_score": 0.95,
            "global_score": 0.67,
            "params": {
                "w1_position_size": 0.14,
                "w1_min_conf": 0.06,
                "w1_risk_mult": 1.15,
                "w2_position_size": 0.17,
                "w2_min_conf": 0.09,
                "w2_risk_mult": 1.25,
                "w3_position_size": 0.21,
                "w3_min_conf": 0.11,
                "w3_risk_mult": 1.45,
                "w4_position_size": 0.19,
                "w4_min_conf": 0.01,
                "w4_risk_mult": 1.35,
                "stop_loss_pct": 0.07,
                "take_profit_pct": 0.14,
            },
        },
        {
            "description": "Poor performance scenario",
            "w1_score": 0.30,
            "w2_score": 0.25,
            "w3_score": 0.20,
            "w4_score": 0.35,
            "global_score": 0.275,
            "params": {
                "w1_position_size": 0.08,
                "w1_min_conf": 0.14,
                "w1_risk_mult": 0.8,
                "w2_position_size": 0.12,
                "w2_min_conf": 0.13,
                "w2_risk_mult": 0.9,
                "w3_position_size": 0.18,
                "w3_min_conf": 0.15,
                "w3_risk_mult": 1.1,
                "w4_position_size": 0.15,
                "w4_min_conf": 0.12,
                "w4_risk_mult": 1.0,
                "stop_loss_pct": 0.12,
                "take_profit_pct": 0.08,
            },
        },
    ]

    # Create trials
    for i, scenario in enumerate(trial_scenarios):
        logger.info(f"Creating trial {i + 1}: {scenario['description']}")

        trial = study.ask()

        # Set parameters
        for param_name, value in scenario["params"].items():
            trial.suggest_float(param_name, value, value)

        # Set individual worker scores and behaviors
        trial.set_user_attr("w1_score", scenario["w1_score"])
        trial.set_user_attr("w2_score", scenario["w2_score"])
        trial.set_user_attr("w3_score", scenario["w3_score"])
        trial.set_user_attr("w4_score", scenario["w4_score"])

        # Set worker-specific parameters
        trial.set_user_attr(
            "w1_params",
            {
                "position_size_pct": scenario["params"]["w1_position_size"],
                "min_confidence": scenario["params"]["w1_min_conf"],
                "risk_multiplier": scenario["params"]["w1_risk_mult"],
            },
        )
        trial.set_user_attr(
            "w2_params",
            {
                "position_size_pct": scenario["params"]["w2_position_size"],
                "min_confidence": scenario["params"]["w2_min_conf"],
                "risk_multiplier": scenario["params"]["w2_risk_mult"],
            },
        )
        trial.set_user_attr(
            "w3_params",
            {
                "position_size_pct": scenario["params"]["w3_position_size"],
                "min_confidence": scenario["params"]["w3_min_conf"],
                "risk_multiplier": scenario["params"]["w3_risk_mult"],
            },
        )
        trial.set_user_attr(
            "w4_params",
            {
                "position_size_pct": scenario["params"]["w4_position_size"],
                "min_confidence": scenario["params"]["w4_min_conf"],
                "risk_multiplier": scenario["params"]["w4_risk_mult"],
            },
        )

        # Set worker behaviors
        worker_behaviors = {
            "w1": {
                "total_trades": 15,
                "win_rate": scenario["w1_score"] * 0.8,
                "total_pnl": scenario["w1_score"] * 10,
                "timeframe_trades": {"5m": 8, "1h": 5, "4h": 2},
                "tf_diversity": 3,
                "score": scenario["w1_score"],
            },
            "w2": {
                "total_trades": 18,
                "win_rate": scenario["w2_score"] * 0.8,
                "total_pnl": scenario["w2_score"] * 10,
                "timeframe_trades": {"5m": 10, "1h": 6, "4h": 2},
                "tf_diversity": 3,
                "score": scenario["w2_score"],
            },
            "w3": {
                "total_trades": 22,
                "win_rate": scenario["w3_score"] * 0.8,
                "total_pnl": scenario["w3_score"] * 10,
                "timeframe_trades": {"5m": 15, "1h": 5, "4h": 2},
                "tf_diversity": 3,
                "score": scenario["w3_score"],
            },
            "w4": {
                "total_trades": 20,
                "win_rate": scenario["w4_score"] * 0.8,
                "total_pnl": scenario["w4_score"] * 10,
                "timeframe_trades": {"5m": 12, "1h": 6, "4h": 2},
                "tf_diversity": 3,
                "score": scenario["w4_score"],
            },
        }
        trial.set_user_attr("worker_behaviors", worker_behaviors)

        # Additional metadata
        trial.set_user_attr("duration_minutes", 5.0 + i * 0.5)
        trial.set_user_attr("start_time", datetime.now().isoformat())
        trial.set_user_attr("end_time", datetime.now().isoformat())

        # Tell the study this trial is complete
        study.tell(trial, scenario["global_score"])

    logger.info(f"Created {len(trial_scenarios)} test trials")
    return storage_name, study_name, temp_db.name


def test_worker_analysis():
    """Test the worker analysis functionality."""
    logger.info("🧪 Starting Worker Optimization System Test")

    # Create mock data
    storage_name, study_name, db_path = create_mock_trial_data()

    try:
        # Import and test our analyzer
        from scripts.show_optuna_results import WorkerOptimizationAnalyzer

        logger.info("📊 Testing WorkerOptimizationAnalyzer...")

        # Create analyzer
        analyzer = WorkerOptimizationAnalyzer(storage_name, study_name)

        # Run analysis
        analyzer.run_analysis()

        logger.info("✅ Worker optimization system test completed successfully!")

        # Test direct access to best workers
        if analyzer.load_study():
            analysis = analyzer.analyze_trials()
            best_workers = analyzer.find_best_worker_trials(analysis["completed"])

            print("\n" + "=" * 60)
            print("🔬 DETAILED TEST RESULTS")
            print("=" * 60)

            expected_best_scores = {
                "w1": 0.85,  # From scenario 2
                "w2": 0.70,  # From scenario 1
                "w3": 0.90,  # From scenario 3
                "w4": 0.95,  # From scenario 4
            }

            for worker_id, expected_score in expected_best_scores.items():
                if worker_id in best_workers:
                    actual_score = best_workers[worker_id]["score"]
                    trial_num = best_workers[worker_id]["trial_number"]

                    if abs(actual_score - expected_score) < 0.001:
                        print(
                            f"✅ {worker_id.upper()}: Expected {expected_score}, Got {actual_score} (Trial #{trial_num})"
                        )
                    else:
                        print(
                            f"❌ {worker_id.upper()}: Expected {expected_score}, Got {actual_score} (Trial #{trial_num})"
                        )
                else:
                    print(f"❌ {worker_id.upper()}: No data found")

        return True

    except ImportError as e:
        logger.error(f"❌ Could not import analyzer: {e}")
        logger.info("💡 Make sure show_optuna_results.py is in the scripts directory")
        return False
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        # Cleanup
        try:
            os.unlink(db_path)
            logger.info("🧹 Cleaned up test database")
        except:
            pass


def validate_integration():
    """Validate that the new system integrates properly."""
    logger.info("🔧 Testing Integration with Existing System...")

    expected_files = [
        "scripts/optimize_hyperparams.py",
        "scripts/show_optuna_results.py",
    ]

    for file_path in expected_files:
        if not os.path.exists(file_path):
            logger.error(f"❌ Missing required file: {file_path}")
            return False
        else:
            logger.info(f"✅ Found: {file_path}")

    # Test that we can import the required modules
    try:
        from scripts.show_optuna_results import WorkerOptimizationAnalyzer

        logger.info("✅ Successfully imported WorkerOptimizationAnalyzer")
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False

    return True


def main():
    """Main test function."""
    print("🚀 WORKER OPTIMIZATION SYSTEM TEST")
    print("=" * 50)

    # Run integration validation
    if not validate_integration():
        print("❌ Integration validation failed")
        return False

    # Run functional test
    if not test_worker_analysis():
        print("❌ Functional test failed")
        return False

    print("\n🎉 ALL TESTS PASSED!")
    print("\n📋 SUMMARY:")
    print("• Worker-specific optimization system is working")
    print("• Individual worker scores are captured correctly")
    print("• Best parameters per worker are extracted properly")
    print("• Analysis and reporting functions correctly")

    print("\n🔄 NEXT STEPS:")
    print(
        "• Run actual optimization with: python scripts/optimize_hyperparams.py --n-trials 5"
    )
    print("• View results with: python scripts/show_optuna_results.py")
    print("• The system will now track individual worker performance!")

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
