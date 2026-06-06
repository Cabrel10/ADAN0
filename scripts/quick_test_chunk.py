#!/usr/bin/env python3
"""
QUICK CHUNK TEST - Extract & Validate Real Data
Reads from existing logs and validates
"""

import json
import sys
from pathlib import Path
from datetime import datetime

def test_trades_from_logs():
    """Extract trades from reward logs and validate"""
    print("\n" + "=" * 80)
    print("🧪 QUICK TEST: Trade Validation")
    print("=" * 80)

    reward_file = "logs/rewards/worker_1_rewards_20260605_135933.jsonl"
    
    if not Path(reward_file).exists():
        print(f"❌ File not found: {reward_file}")
        return

    print(f"\n📂 Reading {reward_file}...")
    
    trades = []
    pnl_total = 0.0
    valid_count = 0
    invalid_count = 0

    try:
        with open(reward_file, 'r') as f:
            for line_num, line in enumerate(f):
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # Extract trade data if present
                    if 'realized_pnl' in data:
                        pnl = float(data.get('realized_pnl', 0))
                        pnl_total += pnl
                        trades.append({
                            'step': data.get('step'),
                            'pnl': pnl,
                            'portfolio_value': data.get('portfolio_value'),
                            'cash': data.get('cash'),
                        })
                        valid_count += 1
                    
                    if line_num % 1000 == 0 and line_num > 0:
                        print(f"  Processed {line_num} lines...")
                        
                except json.JSONDecodeError:
                    invalid_count += 1
                    continue
    
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return

    print(f"\n✅ Results:")
    print(f"  Total lines: {valid_count + invalid_count}")
    print(f"  Valid trades: {valid_count}")
    print(f"  Invalid: {invalid_count}")
    print(f"  Total realized PnL: ${pnl_total:.2f}")
    
    if trades:
        print(f"\n📊 Sample trades (first 10):")
        for i, trade in enumerate(trades[:10]):
            print(f"  Trade {i}: Step={trade['step']}, PnL=${trade['pnl']:.2f}, Portfolio=${trade.get('portfolio_value', 0):.2f}")

    return trades


def test_validation_data():
    """Test with validation backtest data"""
    print("\n" + "=" * 80)
    print("🧪 QUICK TEST: Validation Data")
    print("=" * 80)

    backtest_file = "logs/validation/backtest_5120.json"
    
    if not Path(backtest_file).exists():
        print(f"❌ File not found: {backtest_file}")
        return

    print(f"\n📂 Reading {backtest_file}...")
    
    try:
        with open(backtest_file, 'r') as f:
            data = json.load(f)
        
        print(f"\n✅ Backtest Data Loaded:")
        print(f"  Keys: {list(data.keys())[:10]}")
        
        if 'results' in data:
            results = data['results']
            print(f"\n  Results keys: {list(results.keys())[:10]}")
            
            if 'final_portfolio_value' in results:
                print(f"  Final Portfolio: ${results['final_portfolio_value']:.2f}")
            if 'total_pnl' in results:
                print(f"  Total PnL: ${results['total_pnl']:.2f}")
            if 'win_rate' in results:
                print(f"  Win Rate: {results['win_rate']:.1f}%")
            if 'sharpe' in results:
                print(f"  Sharpe: {results['sharpe']:.4f}")
        
        if 'trades' in data:
            trades = data['trades']
            print(f"\n  Total trades in backtest: {len(trades)}")
            if trades:
                print(f"  First trade: {json.dumps(trades[0], indent=4)[:200]}...")
        
        return data
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def test_metrics_correlation():
    """Test value function correlation from metrics"""
    print("\n" + "=" * 80)
    print("🧪 QUICK TEST: Metrics Correlation")
    print("=" * 80)

    metrics_file = "logs/metrics/metrics_20260605_073338.jsonl"
    
    if not Path(metrics_file).exists():
        print(f"❌ File not found: {metrics_file}")
        return

    print(f"\n📂 Reading {metrics_file}...")
    
    values = []
    returns = []
    step_count = 0

    try:
        with open(metrics_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # Extract value and return
                    if 'value_pred' in data and 'pnl_net' in data:
                        v = float(data.get('value_pred', 0))
                        r = float(data.get('pnl_net', 0))
                        
                        if v != 0:  # Ignore zeros
                            values.append(v)
                            returns.append(r)
                    
                    step_count += 1
                    if step_count % 1000 == 0:
                        print(f"  Processed {step_count} lines...")
                    
                except:
                    continue
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    if len(values) > 10:
        import numpy as np
        
        v_arr = np.array(values)
        r_arr = np.array(returns)
        
        corr = np.corrcoef(v_arr, r_arr)[0, 1]
        ss_res = np.sum((r_arr - v_arr) ** 2)
        ss_tot = np.sum((r_arr - np.mean(r_arr)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        print(f"\n✅ Value Function Analysis:")
        print(f"  Samples: {len(values)}")
        print(f"  Correlation: {corr:.4f}")
        print(f"  R² (explained variance): {r2:.4f}")
        print(f"  Mean value pred: ${np.mean(v_arr):.2f}")
        print(f"  Mean actual return: ${np.mean(r_arr):.2f}")
        print(f"  Std value pred: ${np.std(v_arr):.2f}")
        print(f"  Std actual return: ${np.std(r_arr):.2f}")
        
        if r2 < 0.1:
            print(f"\n  ⚠️ Value function is WEAK (R² = {r2:.4f})")
        elif r2 < 0.3:
            print(f"\n  ⚠️ Value function is MODERATE (R² = {r2:.4f})")
        else:
            print(f"\n  ✅ Value function is GOOD (R² = {r2:.4f})")
    else:
        print(f"  ⚠️ Insufficient data: only {len(values)} samples")


def test_oos_performance():
    """Test out-of-sample validation results"""
    print("\n" + "=" * 80)
    print("🧪 QUICK TEST: Out-of-Sample Performance")
    print("=" * 80)

    oos_file = "logs/validation/oos_w2_detailed.json"
    
    if not Path(oos_file).exists():
        print(f"❌ File not found: {oos_file}")
        return

    print(f"\n📂 Reading {oos_file}...")
    
    try:
        with open(oos_file, 'r') as f:
            data = json.load(f)
        
        print(f"\n✅ Out-of-Sample Results:")
        
        metrics = ['portfolio_value', 'total_pnl', 'return_pct', 'sharpe', 'win_rate', 'max_drawdown']
        
        for metric in metrics:
            if metric in data:
                val = data[metric]
                if isinstance(val, float):
                    if 'pct' in metric or metric == 'win_rate':
                        print(f"  {metric}: {val:.2f}%")
                    elif 'sharpe' in metric:
                        print(f"  {metric}: {val:.4f}")
                    else:
                        print(f"  {metric}: ${val:.2f}")
                else:
                    print(f"  {metric}: {val}")
        
        # Check for generalization issues
        if 'chunk_transitions' in data:
            trans = data['chunk_transitions']
            print(f"\n  Chunk transitions: {trans}")
        
        return data
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def main():
    print("\n" + "=" * 80)
    print("🔬 QUICK CHUNK TEST SUITE")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 80)

    print("\n[1/4] Testing trades from logs...")
    test_trades_from_logs()
    
    print("\n[2/4] Testing validation data...")
    test_validation_data()
    
    print("\n[3/4] Testing metrics correlation...")
    test_metrics_correlation()
    
    print("\n[4/4] Testing out-of-sample performance...")
    test_oos_performance()

    print("\n" + "=" * 80)
    print(f"✅ QUICK TEST COMPLETE")
    print(f"Finished: {datetime.now().isoformat()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
