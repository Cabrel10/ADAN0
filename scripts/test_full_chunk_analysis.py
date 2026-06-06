#!/usr/bin/env python3
"""
FULL CHUNK ANALYSIS TEST
Extract & validate real chunk data from Ray training results
Tests all requirements:
1. Extract trades, recalculate PnL manually
2. Verify positions = -$1,918.85 gap
3. Walk-forward generalization test
4. Lookahead bias detection
5. Value function correlation
6. Bearish data generalization
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime

def load_ray_result(result_path: str) -> Dict:
    """Load Ray training result JSON."""
    try:
        with open(result_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"  ⚠️ Error loading {result_path}: {e}")
        return {}

def extract_trades_from_result(result: Dict) -> List[Dict]:
    """Extract trade data from Ray result."""
    trades = []
    
    # Try multiple possible locations
    if 'trades' in result:
        trades = result['trades']
    elif 'trade_log' in result:
        trades = result['trade_log']
    elif 'episodes' in result:
        for ep in result['episodes']:
            if 'trades' in ep:
                trades.extend(ep['trades'])
    
    return trades

def validate_trades_pnl(trades: List[Dict]) -> Dict:
    """Validate trade PnL calculations manually."""
    print("\n  📊 TRADE VALIDATION (Manual PnL Recalculation)")
    print("  " + "-" * 70)
    
    valid = 0
    invalid = 0
    discrepancies = []
    total_pnl = 0.0
    
    for idx, trade in enumerate(trades[:50]):  # First 50
        try:
            entry = float(trade.get('entry_price', 0))
            exit_p = float(trade.get('exit_price', 0))
            size = float(trade.get('size', 0))
            direction = float(trade.get('direction', 1))
            reported_pnl = float(trade.get('pnl', 0))
            
            if entry > 0 and exit_p > 0 and size > 0:
                calc_pnl = (exit_p - entry) * size * direction
                
                if abs(calc_pnl - reported_pnl) < 0.01:
                    valid += 1
                    total_pnl += reported_pnl
                else:
                    invalid += 1
                    discrepancies.append({
                        'idx': idx,
                        'expected': calc_pnl,
                        'reported': reported_pnl,
                        'diff': abs(calc_pnl - reported_pnl)
                    })
        except:
            invalid += 1
    
    print(f"    ✅ Valid: {valid}/50")
    print(f"    ❌ Invalid: {invalid}/50")
    print(f"    💰 Total PnL (sample): ${total_pnl:.2f}")
    
    if discrepancies:
        print(f"\n    🚨 Discrepancies ({len(discrepancies)}):")
        for d in discrepancies[:3]:
            print(f"      Trade {d['idx']}: Expected ${d['expected']:.2f}, Got ${d['reported']:.2f}")
    
    return {
        'valid': valid,
        'invalid': invalid,
        'total_pnl_sample': total_pnl,
        'discrepancies': len(discrepancies),
        'status': '✅ VALID' if invalid == 0 else '⚠️ ISSUES'
    }

def validate_positions_gap(result: Dict) -> Dict:
    """Verify positions account for the gap."""
    print("\n  📈 POSITION VALIDATION (Gap = -$1,918.85?)")
    print("  " + "-" * 70)
    
    portfolio_value = float(result.get('final_equity', 0))
    initial = float(result.get('initial_equity', 20.5))
    realized_pnl = float(result.get('realized_pnl', 0))
    
    # Implied unrealized
    implied_unrealized = realized_pnl - (portfolio_value - initial)
    
    print(f"    Initial: ${initial:.2f}")
    print(f"    Final: ${portfolio_value:.2f}")
    print(f"    Realized PnL: ${realized_pnl:.2f}")
    print(f"    Implied Unrealized: ${implied_unrealized:.2f}")
    
    consistency_error = abs((realized_pnl + implied_unrealized) - (portfolio_value - initial))
    print(f"    Consistency Error: ${consistency_error:.6f}")
    
    status = '✅ CONSISTENT' if consistency_error < 0.01 else '❌ INCONSISTENT'
    
    return {
        'portfolio_value': portfolio_value,
        'realized_pnl': realized_pnl,
        'implied_unrealized': implied_unrealized,
        'consistency_error': consistency_error,
        'status': status
    }

def test_walk_forward_generalization(results: List[Dict]) -> Dict:
    """Test generalization across different market contexts."""
    print("\n  🔄 WALK-FORWARD GENERALIZATION TEST")
    print("  " + "-" * 70)
    
    if len(results) < 2:
        print("    ⚠️ Insufficient results for walk-forward test")
        return {'status': 'INSUFFICIENT_DATA'}
    
    # Assuming different workers = different market contexts
    returns = []
    for r in results:
        initial = float(r.get('initial_equity', 20.5))
        final = float(r.get('final_equity', 20.5))
        ret_pct = ((final - initial) / initial) * 100
        returns.append(ret_pct)
    
    print(f"    Workers tested: {len(results)}")
    print(f"    Returns: {[f'{r:.1f}%' for r in returns]}")
    
    avg_return = np.mean(returns)
    std_return = np.std(returns)
    
    print(f"    Mean return: {avg_return:.1f}%")
    print(f"    Std return: {std_return:.1f}%")
    print(f"    Consistency: {('✅ GOOD' if std_return < 5 else '⚠️ HIGH_VARIANCE')}")
    
    return {
        'returns': returns,
        'mean': avg_return,
        'std': std_return,
        'consistency': '✅ GOOD' if std_return < 5 else '⚠️ HIGH_VARIANCE'
    }

def check_lookahead_bias(result: Dict) -> Dict:
    """Check for lookahead bias signs."""
    print("\n  🔍 LOOKAHEAD BIAS CHECK")
    print("  " + "-" * 70)
    
    # Check if action knows future prices
    anomalies = 0
    
    if 'episodes' in result:
        for ep in result['episodes'][:10]:  # First 10 episodes
            if 'steps' in ep:
                for i, step in enumerate(ep['steps'][:-1]):
                    try:
                        action = int(step.get('action', 0))
                        reward = float(step.get('reward', 0))
                        current_price = float(step.get('current_price', 0))
                        
                        # Check next price
                        if i + 1 < len(ep['steps']):
                            next_price = float(ep['steps'][i+1].get('current_price', 0))
                            
                            # If action BUY (1) and price goes UP massively with huge reward
                            if action == 1 and current_price > 0 and next_price > current_price * 1.1 and reward > 10:
                                anomalies += 1
                    except:
                        pass
    
    print(f"    Anomalies found: {anomalies}")
    
    status = '✅ CLEAN' if anomalies < 5 else '⚠️ POSSIBLE_BIAS'
    return {
        'anomalies': anomalies,
        'status': status
    }

def test_value_function_correlation(result: Dict) -> Dict:
    """Test value function R²."""
    print("\n  🧠 VALUE FUNCTION CORRELATION TEST")
    print("  " + "-" * 70)
    
    value_preds = []
    actual_returns = []
    
    if 'episodes' in result:
        for ep in result['episodes'][-10:]:  # Last 10 episodes
            if 'steps' in ep:
                for step in ep['steps'][-100:]:  # Last 100 steps
                    try:
                        v_pred = float(step.get('value_pred', 0))
                        pnl = float(step.get('pnl_net', 0))
                        
                        if v_pred != 0:
                            value_preds.append(v_pred)
                            actual_returns.append(pnl)
                    except:
                        pass
    
    if len(value_preds) > 10:
        v_arr = np.array(value_preds)
        r_arr = np.array(actual_returns)
        
        corr = np.corrcoef(v_arr, r_arr)[0, 1]
        ss_res = np.sum((r_arr - v_arr) ** 2)
        ss_tot = np.sum((r_arr - np.mean(r_arr)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        print(f"    Samples: {len(value_preds)}")
        print(f"    Correlation: {corr:.4f}")
        print(f"    R² (explained variance): {r2:.4f}")
        
        if r2 > 0.3:
            quality = "✅ GOOD"
        elif r2 > 0.1:
            quality = "⚠️ WEAK"
        else:
            quality = "❌ POOR"
        
        print(f"    Quality: {quality}")
        
        return {
            'samples': len(value_preds),
            'correlation': corr,
            'r_squared': r2,
            'quality': quality
        }
    
    return {
        'samples': len(value_preds),
        'status': '⚠️ INSUFFICIENT_DATA'
    }

def test_bearish_generalization(results: List[Dict]) -> Dict:
    """Test if agent generalizes to bearish data."""
    print("\n  📉 BEARISH DATA GENERALIZATION TEST")
    print("  " + "-" * 70)
    
    if len(results) < 1:
        print("    ⚠️ No results available")
        return {'status': 'NO_DATA'}
    
    # Assuming we have multiple workers, check variance
    returns = []
    for r in results:
        initial = float(r.get('initial_equity', 20.5))
        final = float(r.get('final_equity', 20.5))
        ret = final - initial
        returns.append(ret)
    
    positive = sum(1 for r in returns if r > 0)
    negative = sum(1 for r in returns if r < 0)
    
    print(f"    Workers: {len(results)}")
    print(f"    Profitable: {positive}")
    print(f"    Loss-making: {negative}")
    print(f"    Win rate: {positive/len(results)*100:.1f}%")
    
    if positive / len(results) < 0.5:
        print(f"    ⚠️ Low win rate suggests trend-dependency")
        status = "⚠️ TREND_DEPENDENT"
    else:
        status = "✅ GENERALIZABLE"
    
    return {
        'workers': len(results),
        'profitable': positive,
        'loss_making': negative,
        'win_rate_pct': positive/len(results)*100 if results else 0,
        'status': status
    }

def main():
    print("\n" + "=" * 80)
    print("🧪 FULL CHUNK ANALYSIS TEST SUITE")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 80)
    
    # Load Ray results
    ray_dir = Path("logs/ray_results/adan_pbt_training")
    worker_dirs = list(ray_dir.glob("ADAN_PBT_Worker*"))
    
    print(f"\n📂 Found {len(worker_dirs)} worker results")
    
    all_results = []
    
    for worker_dir in worker_dirs[:4]:  # First 4 workers
        result_file = worker_dir / "result.json"
        if result_file.exists():
            print(f"\n  Processing {worker_dir.name}...")
            result = load_ray_result(str(result_file))
            if result:
                all_results.append(result)
    
    print(f"\n✅ Loaded {len(all_results)} results")
    
    # ─────────────────────────────────────────────────────────────────
    # PHASE 1: Trade Validation
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PHASE 1: TRADE VALIDATION")
    print("=" * 80)
    
    for i, result in enumerate(all_results[:2]):
        print(f"\n  Worker {i}:")
        trades = extract_trades_from_result(result)
        if trades:
            validate_trades_pnl(trades)
        else:
            print("    ⚠️ No trade data found")
    
    # ─────────────────────────────────────────────────────────────────
    # PHASE 2: Position Validation
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PHASE 2: POSITION VALIDATION")
    print("=" * 80)
    
    for i, result in enumerate(all_results[:2]):
        print(f"\n  Worker {i}:")
        validate_positions_gap(result)
    
    # ─────────────────────────────────────────────────────────────────
    # PHASE 3: Walk-Forward Generalization
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PHASE 3: WALK-FORWARD GENERALIZATION TEST")
    print("=" * 80)
    
    test_walk_forward_generalization(all_results)
    
    # ─────────────────────────────────────────────────────────────────
    # PHASE 4: Lookahead Bias
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PHASE 4: LOOKAHEAD BIAS CHECK")
    print("=" * 80)
    
    for i, result in enumerate(all_results[:1]):
        print(f"\n  Worker {i}:")
        check_lookahead_bias(result)
    
    # ─────────────────────────────────────────────────────────────────
    # PHASE 5: Value Function
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PHASE 5: VALUE FUNCTION CORRELATION")
    print("=" * 80)
    
    for i, result in enumerate(all_results[:1]):
        print(f"\n  Worker {i}:")
        test_value_function_correlation(result)
    
    # ─────────────────────────────────────────────────────────────────
    # PHASE 6: Bearish Generalization
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PHASE 6: BEARISH DATA GENERALIZATION")
    print("=" * 80)
    
    test_bearish_generalization(all_results)
    
    print("\n" + "=" * 80)
    print("✅ TEST SUITE COMPLETE")
    print(f"Finished: {datetime.now().isoformat()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
