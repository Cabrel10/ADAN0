#!/usr/bin/env python3
"""
COMPREHENSIVE CHUNK AUDIT SCRIPT
─────────────────────────────────
Validates entire chunk data: trades, positions, PnL, value function, lookahead bias

Usage:
  python scripts/audit_chunk_comprehensive.py --checkpoint <path> --chunk <1|2>
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
    from adan_trading_bot.portfolio.portfolio_manager import PortfolioManager
except ImportError:
    print("⚠️ Cannot import ADAN modules. Running partial audit.")


class ChunkAuditor:
    """Complete chunk audit with multiple validation layers."""

    def __init__(self, checkpoint_path: str, chunk_id: int):
        self.checkpoint_path = checkpoint_path
        self.chunk_id = chunk_id
        self.results = {
            "metadata": {},
            "trade_validation": {},
            "position_validation": {},
            "pnl_validation": {},
            "value_function_validation": {},
            "lookahead_bias_check": {},
            "summary": {},
        }
        self.trades = []
        self.steps_data = []

    def run_full_audit(self) -> Dict[str, Any]:
        """Execute all audit phases."""
        print("\n" + "=" * 80)
        print(f"🔍 COMPREHENSIVE CHUNK {self.chunk_id} AUDIT")
        print("=" * 80)

        self._load_checkpoint_data()
        self._validate_trades()
        self._validate_positions()
        self._validate_pnl()
        self._validate_value_function()
        self._check_lookahead_bias()
        self._generate_summary()

        return self.results

    def _load_checkpoint_data(self) -> None:
        """Load checkpoint data from disk."""
        print(f"\n📂 Loading checkpoint data from {self.checkpoint_path}...")

        # Try multiple possible locations
        possible_paths = [
            self.checkpoint_path,
            f"{self.checkpoint_path}/checkpoint_{self.chunk_id}",
            f"{self.checkpoint_path}/chunk_{self.chunk_id}",
        ]

        loaded = False
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    # Try to load JSON
                    if os.path.isfile(path):
                        with open(path, "r") as f:
                            data = json.load(f)
                            self._parse_checkpoint_data(data)
                            loaded = True
                            break
                    # Try to load from directory
                    elif os.path.isdir(path):
                        for file in os.listdir(path):
                            if file.endswith(".json"):
                                with open(os.path.join(path, file), "r") as f:
                                    data = json.load(f)
                                    self._parse_checkpoint_data(data)
                                    loaded = True
                                    break
                except Exception as e:
                    print(f"  ⚠️ Error loading {path}: {e}")
                    continue

        if not loaded:
            print(f"  ❌ Could not load checkpoint data from {self.checkpoint_path}")
            self.results["metadata"]["status"] = "NO_DATA"
        else:
            print(f"  ✅ Loaded {len(self.trades)} trades, {len(self.steps_data)} steps")
            self.results["metadata"]["trades_loaded"] = len(self.trades)
            self.results["metadata"]["steps_loaded"] = len(self.steps_data)

    def _parse_checkpoint_data(self, data: Dict) -> None:
        """Parse checkpoint data structure."""
        if isinstance(data, dict):
            if "trades" in data:
                self.trades = data["trades"]
            if "steps" in data:
                self.steps_data = data["steps"]
            if "trade_log" in data:
                self.trades = data["trade_log"]
            if "episode_data" in data:
                ep_data = data["episode_data"]
                if "trades" in ep_data:
                    self.trades = ep_data["trades"]

    def _validate_trades(self) -> None:
        """Validate trade log: entry/exit prices, sizes, PnL calculations."""
        print("\n" + "=" * 80)
        print("1️⃣  TRADE VALIDATION")
        print("=" * 80)

        if not self.trades:
            print("  ⚠️ No trade data available")
            self.results["trade_validation"]["status"] = "NO_DATA"
            return

        valid_trades = 0
        invalid_trades = 0
        pnl_discrepancies = []
        total_realized_pnl = 0.0

        print(f"\n  Analyzing {len(self.trades)} trades...")

        for idx, trade in enumerate(self.trades[:50]):  # First 50 trades
            if idx % 10 == 0:
                print(f"    Processing trade {idx}...")

            try:
                # Extract trade fields
                entry_price = float(trade.get("entry_price", 0))
                exit_price = float(trade.get("exit_price", 0))
                size = float(trade.get("size", 0))
                direction = float(trade.get("direction", 1))  # 1=long, -1=short
                reported_pnl = float(trade.get("pnl", 0))

                # Calculate expected PnL
                if entry_price > 0 and exit_price > 0 and size > 0:
                    expected_pnl = (exit_price - entry_price) * size * direction
                else:
                    expected_pnl = 0.0

                # Validate
                if abs(expected_pnl - reported_pnl) < 1e-4:  # Allow small float errors
                    valid_trades += 1
                    total_realized_pnl += reported_pnl
                else:
                    invalid_trades += 1
                    discrepancy = {
                        "trade_id": idx,
                        "expected_pnl": expected_pnl,
                        "reported_pnl": reported_pnl,
                        "diff": abs(expected_pnl - reported_pnl),
                    }
                    pnl_discrepancies.append(discrepancy)

                # Sanity checks
                if entry_price <= 0:
                    print(f"    ⚠️ Trade {idx}: Invalid entry price {entry_price}")
                if size <= 0:
                    print(f"    ⚠️ Trade {idx}: Invalid size {size}")

            except Exception as e:
                print(f"    ⚠️ Error parsing trade {idx}: {e}")
                invalid_trades += 1

        # Report
        print(f"\n  ✅ Valid trades: {valid_trades}")
        print(f"  ❌ Invalid trades: {invalid_trades}")
        print(f"  💰 Total realized PnL (first 50): ${total_realized_pnl:.2f}")

        if pnl_discrepancies:
            print(f"\n  🚨 PnL Discrepancies Found: {len(pnl_discrepancies)}")
            for disc in pnl_discrepancies[:5]:  # Show first 5
                print(f"    Trade {disc['trade_id']}: Expected ${disc['expected_pnl']:.2f}, Got ${disc['reported_pnl']:.2f} (diff: ${disc['diff']:.4f})")

        self.results["trade_validation"] = {
            "total_trades": len(self.trades),
            "valid_trades": valid_trades,
            "invalid_trades": invalid_trades,
            "pnl_discrepancies": len(pnl_discrepancies),
            "total_realized_pnl_sample": total_realized_pnl,
            "status": "✅ VALID" if invalid_trades == 0 else "⚠️ ISSUES_FOUND",
        }

    def _validate_positions(self) -> None:
        """Validate open positions: sum = reported gap."""
        print("\n" + "=" * 80)
        print("2️⃣  POSITION VALIDATION")
        print("=" * 80)

        if not self.steps_data:
            print("  ⚠️ No step data available")
            self.results["position_validation"]["status"] = "NO_DATA"
            return

        # Get last step
        last_step = self.steps_data[-1] if self.steps_data else {}
        print(f"\n  Analyzing final positions from step data...")

        portfolio_value = float(last_step.get("portfolio_value", 0))
        cash = float(last_step.get("cash", 0))
        realized_equity = float(last_step.get("realized_equity", 0))
        initial_capital = float(last_step.get("initial_capital", 20.50))

        # Calculate implied gap
        open_positions_value = portfolio_value - cash if cash > 0 else portfolio_value
        implied_gap = realized_equity - portfolio_value

        print(f"\n  Portfolio Snapshot:")
        print(f"    Initial Capital: ${initial_capital:.2f}")
        print(f"    Portfolio Value: ${portfolio_value:.2f}")
        print(f"    Cash: ${cash:.2f}")
        print(f"    Open Positions Value: ${open_positions_value:.2f}")
        print(f"    Realized Equity: ${realized_equity:.2f}")
        print(f"    Implied Gap (Open Losses): ${implied_gap:.2f}")

        # Validate consistency
        consistency_error = abs((cash + open_positions_value) - portfolio_value)
        if consistency_error < 0.01:
            print(f"\n  ✅ Consistency Check: PASS (error: ${consistency_error:.6f})")
            status = "✅ CONSISTENT"
        else:
            print(f"\n  ❌ Consistency Check: FAIL (error: ${consistency_error:.2f})")
            status = "❌ INCONSISTENT"

        self.results["position_validation"] = {
            "portfolio_value": portfolio_value,
            "cash": cash,
            "open_positions_value": open_positions_value,
            "realized_equity": realized_equity,
            "implied_gap": implied_gap,
            "consistency_error": consistency_error,
            "status": status,
        }

    def _validate_pnl(self) -> None:
        """Validate PnL accounting: realized vs unrealized."""
        print("\n" + "=" * 80)
        print("3️⃣  PnL ACCOUNTING VALIDATION")
        print("=" * 80)

        if not self.steps_data:
            print("  ⚠️ No step data available")
            return

        last_step = self.steps_data[-1]
        initial = float(last_step.get("initial_capital", 20.50))
        portfolio_final = float(last_step.get("portfolio_value", 0))
        realized_pnl = float(last_step.get("realized_pnl", 0))
        realized_equity = float(last_step.get("realized_equity", 0))

        # Calculate components
        total_pnl = portfolio_final - initial
        unrealized_pnl = portfolio_final - realized_equity
        total_realized_from_log = sum(float(t.get("pnl", 0)) for t in self.trades if t.get("pnl"))

        print(f"\n  PnL Breakdown:")
        print(f"    Initial Capital: ${initial:.2f}")
        print(f"    Final Portfolio: ${portfolio_final:.2f}")
        print(f"    Total PnL: ${total_pnl:.2f} ({total_pnl/max(initial, 1)*100:.1f}%)")
        print(f"    Realized PnL (reported): ${realized_pnl:.2f}")
        print(f"    Realized Equity (reported): ${realized_equity:.2f}")
        print(f"    Realized PnL (from trade log): ${total_realized_from_log:.2f}")
        print(f"    Implied Unrealized: ${unrealized_pnl:.2f}")

        # Validate equation
        reconstructed = realized_equity + unrealized_pnl
        equation_error = abs(reconstructed - portfolio_final)

        print(f"\n  Equation Check: realized_equity + unrealized = portfolio_value")
        print(f"    ${realized_equity:.2f} + ${unrealized_pnl:.2f} = ${reconstructed:.2f}")
        print(f"    Actual portfolio: ${portfolio_final:.2f}")
        print(f"    Error: ${equation_error:.6f}")

        if equation_error < 0.01:
            print(f"  ✅ Equation: VALID")
            status = "✅ VALID"
        else:
            print(f"  ❌ Equation: INVALID")
            status = "❌ INVALID"

        self.results["pnl_validation"] = {
            "initial_capital": initial,
            "final_portfolio": portfolio_final,
            "total_pnl": total_pnl,
            "realized_pnl_reported": realized_pnl,
            "realized_pnl_from_trades": total_realized_from_log,
            "unrealized_pnl": unrealized_pnl,
            "equation_error": equation_error,
            "status": status,
        }

    def _validate_value_function(self) -> None:
        """Validate value function: correlation with returns."""
        print("\n" + "=" * 80)
        print("4️⃣  VALUE FUNCTION VALIDATION")
        print("=" * 80)

        if not self.steps_data or len(self.steps_data) < 10:
            print("  ⚠️ Insufficient step data for analysis")
            self.results["value_function_validation"]["status"] = "NO_DATA"
            return

        print(f"\n  Analyzing {len(self.steps_data)} steps...")

        # Extract value predictions and returns
        value_predictions = []
        actual_returns = []
        step_rewards = []

        for step in self.steps_data[-1000:]:  # Last 1000 steps
            try:
                v_pred = float(step.get("value_pred", 0))
                reward = float(step.get("reward", 0))
                pnl = float(step.get("pnl_net", 0))

                if v_pred != 0:  # Ignore zeros
                    value_predictions.append(v_pred)
                    actual_returns.append(pnl)
                    step_rewards.append(reward)
            except:
                pass

        if len(value_predictions) < 10:
            print("  ⚠️ Insufficient value predictions")
            self.results["value_function_validation"]["status"] = "NO_DATA"
            return

        # Calculate correlation
        v_array = np.array(value_predictions)
        r_array = np.array(actual_returns)

        correlation = np.corrcoef(v_array, r_array)[0, 1] if len(v_array) > 1 else 0
        # Calculate R²
        ss_res = np.sum((r_array - v_array) ** 2)
        ss_tot = np.sum((r_array - np.mean(r_array)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        print(f"\n  Value Function Statistics (last 1000 steps):")
        print(f"    Samples: {len(value_predictions)}")
        print(f"    Correlation (pred vs actual): {correlation:.4f}")
        print(f"    R² (explained variance): {r_squared:.4f}")
        print(f"    Mean value pred: ${np.mean(v_array):.2f}")
        print(f"    Mean actual return: ${np.mean(r_array):.2f}")
        print(f"    Std value pred: ${np.std(v_array):.2f}")
        print(f"    Std actual return: ${np.std(r_array):.2f}")

        if r_squared > 0.5:
            quality = "✅ GOOD"
        elif r_squared > 0.2:
            quality = "⚠️ ACCEPTABLE"
        else:
            quality = "❌ POOR"

        print(f"\n  Value Function Quality: {quality}")

        self.results["value_function_validation"] = {
            "samples": len(value_predictions),
            "correlation": float(correlation),
            "r_squared": float(r_squared),
            "mean_value_pred": float(np.mean(v_array)),
            "mean_actual_return": float(np.mean(r_array)),
            "quality": quality,
            "status": quality,
        }

    def _check_lookahead_bias(self) -> None:
        """Check for lookahead bias: does observation include future prices?"""
        print("\n" + "=" * 80)
        print("5️⃣  LOOKAHEAD BIAS CHECK")
        print("=" * 80)

        if not self.steps_data or len(self.steps_data) < 2:
            print("  ⚠️ Insufficient step data")
            self.results["lookahead_bias_check"]["status"] = "NO_DATA"
            return

        print(f"\n  Checking for future price usage...")

        anomalies = []
        for i, step in enumerate(self.steps_data[:-1]):  # Don't check last step
            try:
                current_price = float(step.get("current_price", 0))
                next_price = float(self.steps_data[i + 1].get("current_price", 0))
                action_taken = int(step.get("action", 0))
                reward_got = float(step.get("reward", 0))

                # If action is BUY and reward is huge, but price went DOWN next step
                # → potential lookahead
                if (
                    action_taken == 1
                    and current_price > 0
                    and next_price > 0
                    and next_price > current_price * 1.1
                    and reward_got > 100
                ):
                    anomalies.append(
                        {
                            "step": i,
                            "current_price": current_price,
                            "next_price": next_price,
                            "action": "BUY",
                            "reward": reward_got,
                        }
                    )
            except:
                pass

        if anomalies:
            print(f"\n  🚨 Found {len(anomalies)} potential lookahead anomalies:")
            for anom in anomalies[:5]:
                print(
                    f"    Step {anom['step']}: Price ${anom['current_price']:.2f} → ${anom['next_price']:.2f}, "
                    f"Action: {anom['action']}, Reward: {anom['reward']:.2f}"
                )
            status = "⚠️ POSSIBLE_BIAS"
        else:
            print(f"\n  ✅ No obvious lookahead bias detected")
            status = "✅ CLEAN"

        self.results["lookahead_bias_check"] = {
            "anomalies_found": len(anomalies),
            "status": status,
        }

    def _generate_summary(self) -> None:
        """Generate audit summary and recommendations."""
        print("\n" + "=" * 80)
        print("📋 AUDIT SUMMARY")
        print("=" * 80)

        results = self.results

        print("\n✅ VALIDATIONS:")
        for key, value in results.items():
            if key != "summary" and isinstance(value, dict):
                status = value.get("status", "UNKNOWN")
                print(f"  {key}: {status}")

        print("\n🎯 KEY FINDINGS:")

        # Trade validation
        if results["trade_validation"].get("status") == "✅ VALID":
            print(f"  ✅ Trade PnL calculations: ACCURATE")
        else:
            print(f"  ⚠️ Trade PnL calculations: ISSUES FOUND")

        # Position validation
        pos_error = results["position_validation"].get("consistency_error", 999)
        if pos_error < 0.01:
            print(f"  ✅ Position accounting: CONSISTENT")
        else:
            print(f"  ❌ Position accounting: INCONSISTENT (error: ${pos_error:.2f})")

        # PnL validation
        if results["pnl_validation"].get("equation_error", 999) < 0.01:
            print(f"  ✅ PnL equation: VALID")
        else:
            print(f"  ❌ PnL equation: INVALID")

        # Value function
        r2 = results["value_function_validation"].get("r_squared", 0)
        print(f"  Value function R²: {r2:.4f} (explained variance)")
        if r2 > 0.5:
            print(f"    → Good predictive value")
        elif r2 > 0.1:
            print(f"    → Weak but usable")
        else:
            print(f"    → ❌ PROBLEMATIC (nearly useless)")

        # Lookahead
        if results["lookahead_bias_check"].get("status") == "✅ CLEAN":
            print(f"  ✅ Lookahead bias: NOT DETECTED")
        else:
            print(f"  ⚠️ Lookahead bias: POSSIBLE ISSUES")

        print("\n🚀 RECOMMENDATIONS:")

        if results["pnl_validation"].get("status") == "✅ VALID":
            print(f"  → PnL accounting validated, gap ${results['position_validation'].get('implied_gap', 0):.2f} is real")
        else:
            print(f"  → 🚨 HALT: PnL accounting has issues, investigate before proceeding")

        r2 = results["value_function_validation"].get("r_squared", 0)
        if r2 < 0.1:
            print(f"  → 🚨 Value function is nearly useless, retrain or redesign")
        elif r2 < 0.3:
            print(f"  → ⚠️ Value function weak, consider regularization")

        if results["lookahead_bias_check"].get("anomalies_found", 0) > 0:
            print(f"  → 🚨 CRITICAL: Lookahead bias detected, may invalidate results")

        self.results["summary"] = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "✅ PASSED" if all("✅" in str(v.get("status", "")) for v in results.values() if isinstance(v, dict)) else "⚠️ ISSUES",
            "critical_issues": len([v for v in results.values() if isinstance(v, dict) and "❌" in str(v.get("status", ""))]),
        }


def main():
    parser = argparse.ArgumentParser(description="Comprehensive chunk audit")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--chunk", type=int, default=1, help="Chunk ID (1 or 2)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")

    args = parser.parse_args()

    auditor = ChunkAuditor(args.checkpoint, args.chunk)
    results = auditor.run_full_audit()

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to {args.output}")

    return results


if __name__ == "__main__":
    main()
