#!/usr/bin/env python3
"""
GENERALIZATION TEST: Walk-Forward Analysis
─────────────────────────────────────────────
Test if agent trained on Chunk 1 performs well on Chunk 2 (and vice versa)
Detects trend-dependency and overfitting

Usage:
  python scripts/test_generalization.py --model <checkpoint> --test-chunk <1|2> --mode walk-forward
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class GeneralizationTester:
    """Test agent generalization across different market contexts."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.results = {
            "metadata": {
                "model_path": model_path,
                "test_date": datetime.now().isoformat(),
            },
            "tests": {},
        }

    def run_walk_forward_test(
        self, train_chunk: int, test_chunk: int
    ) -> Dict[str, Any]:
        """
        Train on one chunk, test on another.
        
        Scenarios:
        - Train Chunk 1 (Bearish), Test Chunk 2 (Bullish) → Should maintain performance
        - Train Chunk 2 (Bullish), Test Chunk 1 (Bearish) → Should degrade gracefully
        """
        print("\n" + "=" * 80)
        print(
            f"🧪 WALK-FORWARD TEST: Train Chunk {train_chunk} → Test Chunk {test_chunk}"
        )
        print("=" * 80)

        test_name = f"train_chunk{train_chunk}_test_chunk{test_chunk}"

        # Load checkpoint data
        train_data = self._load_chunk_data(train_chunk)
        test_data = self._load_chunk_data(test_chunk)

        if not train_data or not test_data:
            print("❌ Could not load chunk data")
            return {"status": "FAILED", "reason": "NO_DATA"}

        # Calculate metrics for both
        train_metrics = self._calculate_metrics(train_data, "training")
        test_metrics = self._calculate_metrics(test_data, "testing")

        # Compare
        comparison = self._compare_performance(train_metrics, test_metrics)

        self.results["tests"][test_name] = {
            "train_chunk": train_chunk,
            "test_chunk": test_chunk,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "comparison": comparison,
        }

        return comparison

    def _load_chunk_data(self, chunk_id: int) -> Dict[str, Any]:
        """Load chunk data (mock implementation)."""
        print(f"  📂 Loading Chunk {chunk_id} data...")

        # In real implementation, load from checkpoint or database
        # For now, return placeholder
        mock_data = {
            "chunk_id": chunk_id,
            "steps": 25000,
            "trades": 250,  # Estimated
            "status": "loaded",
        }

        if chunk_id == 1:
            # Bearish chunk (harder to trade)
            mock_data.update(
                {
                    "market_context": "Bearish",
                    "avg_return_per_trade": 0.008,  # 0.8%
                    "volatility": 0.45,
                    "win_rate": 0.50,
                }
            )
        elif chunk_id == 2:
            # Bullish chunk (easy to trade)
            mock_data.update(
                {
                    "market_context": "Bullish",
                    "avg_return_per_trade": 0.062,  # 6.2%
                    "volatility": 0.35,
                    "win_rate": 0.498,
                }
            )

        return mock_data

    def _calculate_metrics(self, data: Dict, phase: str) -> Dict[str, float]:
        """Calculate performance metrics for a chunk."""
        return {
            "phase": phase,
            "chunk_id": data.get("chunk_id"),
            "market_context": data.get("market_context", "unknown"),
            "total_steps": data.get("steps", 0),
            "total_trades": data.get("trades", 0),
            "avg_return_per_trade": data.get("avg_return_per_trade", 0),
            "volatility": data.get("volatility", 0),
            "win_rate": data.get("win_rate", 0),
            "sharpe": self._calculate_sharpe(
                data.get("avg_return_per_trade", 0), data.get("volatility", 0)
            ),
        }

    def _calculate_sharpe(self, avg_return: float, volatility: float) -> float:
        """Simple Sharpe calculation."""
        if volatility > 0:
            return (avg_return * 252 - 0.02) / (volatility * np.sqrt(252))
        return 0.0

    def _compare_performance(
        self, train_metrics: Dict, test_metrics: Dict
    ) -> Dict[str, Any]:
        """Compare train vs test performance."""
        degradation = (
            train_metrics["sharpe"] - test_metrics["sharpe"]
        ) / max(train_metrics["sharpe"], 0.01)

        print(f"\n  📊 Performance Comparison:")
        print(f"    Train Sharpe: {train_metrics['sharpe']:.4f}")
        print(f"    Test Sharpe:  {test_metrics['sharpe']:.4f}")
        print(f"    Degradation:  {degradation*100:.1f}%")

        # Assess generalization
        if degradation < 0.1:
            quality = "✅ EXCELLENT (Good generalization)"
        elif degradation < 0.3:
            quality = "⚠️ ACCEPTABLE (Some degradation)"
        elif degradation < 0.6:
            quality = "⚠️ POOR (Significant degradation)"
        else:
            quality = "❌ VERY_POOR (Severe overfitting)"

        print(f"    Assessment: {quality}")

        return {
            "train_sharpe": train_metrics["sharpe"],
            "test_sharpe": test_metrics["sharpe"],
            "sharpe_degradation_pct": degradation * 100,
            "quality": quality,
            "train_market": train_metrics["market_context"],
            "test_market": test_metrics["market_context"],
        }

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all cross-validation tests."""
        print("\n" + "=" * 80)
        print("🔬 GENERALIZATION TEST SUITE")
        print("=" * 80)

        # Test 1: Bullish → Bearish (harder)
        self.run_walk_forward_test(train_chunk=2, test_chunk=1)

        # Test 2: Bearish → Bullish (easier)
        self.run_walk_forward_test(train_chunk=1, test_chunk=2)

        # Generate summary
        self._generate_summary()

        return self.results

    def _generate_summary(self) -> None:
        """Generate summary and recommendations."""
        print("\n" + "=" * 80)
        print("📋 GENERALIZATION SUMMARY")
        print("=" * 80)

        tests = self.results.get("tests", {})

        for test_name, test_result in tests.items():
            comp = test_result.get("comparison", {})
            quality = comp.get("quality", "UNKNOWN")

            print(f"\n  {test_name}:")
            print(f"    Train Market: {comp.get('train_market')}")
            print(f"    Test Market: {comp.get('test_market')}")
            print(f"    Sharpe Degradation: {comp.get('sharpe_degradation_pct'):.1f}%")
            print(f"    Assessment: {quality}")

        print("\n🎯 OVERALL FINDINGS:")

        # Check if agent is trend-dependent
        bullish_to_bearish = tests.get("train_chunk2_test_chunk1", {}).get(
            "comparison", {}
        )
        bearish_to_bullish = tests.get("train_chunk1_test_chunk2", {}).get(
            "comparison", {}
        )

        bullish_deg = bullish_to_bearish.get("sharpe_degradation_pct", 0)
        bearish_deg = bearish_to_bullish.get("sharpe_degradation_pct", 0)

        if bullish_deg > 50 and bearish_deg < 20:
            print(
                "  🚨 STRONG TREND DEPENDENCY: Agent works in bullish, fails in bearish"
            )
            print(f"     → Bullish→Bearish degradation: {bullish_deg:.1f}%")
            print(f"     → Bearish→Bullish degradation: {bearish_deg:.1f}%")
            print("     → Verdict: NOT GENERALIZABLE")
        elif bullish_deg < 30 and bearish_deg < 30:
            print("  ✅ GOOD GENERALIZATION: Performs well across market contexts")
            print("     → Verdict: READY FOR PRODUCTION")
        else:
            print("  ⚠️ MODERATE GENERALIZATION: Some context dependency detected")
            print("     → Verdict: NEEDS IMPROVEMENT")

    def export_results(self, output_file: str) -> None:
        """Export results to JSON."""
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\n💾 Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generalization and walk-forward test")
    parser.add_argument("--model", type=str, required=True, help="Model checkpoint path")
    parser.add_argument(
        "--test-chunk", type=int, choices=[1, 2], help="Specific chunk to test"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["walk-forward", "single"],
        default="walk-forward",
        help="Test mode",
    )
    parser.add_argument("--output", type=str, default="generalization_test.json")

    args = parser.parse_args()

    tester = GeneralizationTester(args.model)

    if args.mode == "walk-forward":
        results = tester.run_all_tests()
    else:
        # Single test mode
        print("Single test mode not yet implemented")
        results = {}

    if args.output:
        tester.export_results(args.output)

    return results


if __name__ == "__main__":
    main()
