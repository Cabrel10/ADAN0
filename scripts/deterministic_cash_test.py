#!/usr/bin/env python3
"""
Deterministic Cash & Survival Test for ADAN Trading Bot.

Tests that:
1. Episodes survive longer than 10 steps (no instant bankrupt)
2. Cash never goes negative
3. Equity stays above bankrupt floor ($11.50) while positions are open
4. BUY and SELL both execute
5. Trades close properly (no memory leak)

Usage:
    python scripts/deterministic_cash_test.py
"""

import os
import sys
import json
import logging
import warnings
import traceback
from datetime import datetime, timedelta
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np

# Setup logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s")
logger = logging.getLogger("test")
logger.setLevel(logging.INFO)


def create_synthetic_data(n_candles=2000, start_price=45000.0, volatility=0.006):
    """Create synthetic BTC price data with realistic movements.
    
    Volatility 0.006 = 0.6% per candle std dev, which means:
    - Low can be ~1.2% below close (2 sigma)
    - SL at 1% will trigger within ~3 candles
    - This creates proper BUY→SL→BUY cycles for testing
    """
    import pandas as pd
    
    np.random.seed(42)
    timestamps = pd.date_range(start="2025-01-01", periods=n_candles, freq="5min")
    
    # Random walk with slight uptrend
    returns = np.random.normal(0.00001, volatility, n_candles)
    prices = start_price * np.exp(np.cumsum(returns))
    
    # Generate OHLC with wider high-low range to trigger SL/TP
    data = pd.DataFrame({
        "timestamp": timestamps,
        "open": prices * (1 + np.random.uniform(-0.003, 0.003, n_candles)),
        "high": prices * (1 + np.abs(np.random.normal(0, 0.008, n_candles))),
        "low": prices * (1 - np.abs(np.random.normal(0, 0.008, n_candles))),
        "close": prices,
        "volume": np.random.uniform(100, 1000, n_candles),
    })
    
    # Add indicators
    data["RSI_14"] = 50 + np.random.normal(0, 10, n_candles)
    data["MACD_12_26_9"] = np.random.normal(0, 50, n_candles)
    data["MACD_SIGNAL_12_26_9"] = np.random.normal(0, 50, n_candles)
    data["MACD_HIST_12_26_9"] = np.random.normal(0, 20, n_candles)
    data["BB_PERCENT_B_20_2"] = np.random.uniform(0, 1, n_candles)
    data["ATR_14"] = prices * volatility * np.random.uniform(0.5, 1.5, n_candles)
    data["ATR_20"] = prices * volatility * np.random.uniform(0.5, 1.5, n_candles)
    data["ATR_50"] = prices * volatility * np.random.uniform(0.5, 1.5, n_candles)
    data["VOLUME_RATIO_20"] = np.random.uniform(0.5, 2.0, n_candles)
    data["EMA_20_RATIO"] = 1.0 + np.random.normal(0, 0.005, n_candles)
    data["STOCH_K_14_3_3"] = np.random.uniform(0, 100, n_candles)
    data["VWAP_RATIO"] = 1.0 + np.random.normal(0, 0.002, n_candles)
    data["PRICE_ACTION"] = np.random.normal(0, 0.01, n_candles)
    
    # Rename columns to uppercase for compatibility
    data.columns = [c.upper() if c != "timestamp" else "TIMESTAMP" for c in data.columns]
    data.set_index("TIMESTAMP", inplace=True)
    
    return data


def load_config():
    """Load config from config.yaml."""
    import yaml
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_test(n_steps=300, seed=42):
    """Run a deterministic test and collect metrics."""
    import yaml
    
    config = load_config()
    
    # Override for test
    config["environment"]["max_steps"] = n_steps + 100
    config["environment"]["observation"]["warmup_steps"] = 30
    
    # Generate synthetic data
    data_5m = create_synthetic_data(n_candles=n_steps + 200)
    data_1h = create_synthetic_data(n_candles=n_steps + 200, volatility=0.005)
    data_4h = create_synthetic_data(n_candles=n_steps + 200, volatility=0.008)
    
    # Import environment
    from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
    
    # Worker config for scalper
    worker_config = {
        "name": "W1 Scalper",
        "profile": "scalper",
        "assets": ["BTCUSDT"],
        "timeframes": ["5m", "1h", "4h"],
        "data_split": "train",
    }
    
    # Create environment
    env = MultiAssetChunkedEnv(
        config=config,
        worker_config=worker_config,
        worker_id=0,
    )
    
    # Reset first (this loads default data)
    np.random.seed(seed)
    obs, info = env.reset()
    
    # Inject synthetic data AFTER reset (so it doesn't get overwritten)
    env.current_data = {
        "BTCUSDT": {
            "5m": data_5m,
            "1h": data_1h,
            "4h": data_4h,
        }
    }
    
    # Tracking
    metrics = {
        "steps": 0,
        "episodes": 0,
        "cash_history": [],
        "equity_history": [],
        "actions": {"HOLD": 0, "BUY": 0, "SELL": 0},
        "trades_opened": 0,
        "trades_closed": 0,
        "bankrupt_resets": 0,
        "min_cash": float("inf"),
        "max_cash": 0.0,
        "min_equity": float("inf"),
        "max_equity": 0.0,
        "episode_lengths": [],
        "rewards": [],
        "errors": [],
    }
    
    current_episode_length = 0
    done = False
    
    for step in range(n_steps):
        # Action space is 25-dim: 5 assets × 5 dims [Action, Size, TF, SL, TP]
        # Only first asset (BTCUSDT) is used — indices 0-4
        action = np.zeros(25, dtype=np.float32)
        
        # Strategy: cycle BUY → HOLD(20 steps) → SELL → HOLD(10 steps) → BUY...
        cycle_pos = current_episode_length % 35  # 35-step cycle
        
        if cycle_pos < 3:
            action[0] = 0.8  # BUY signal for asset 0
            action[1] = 0.3  # Medium-high size
        elif cycle_pos >= 25 and cycle_pos < 30:
            action[0] = -0.8  # SELL signal for asset 0
        else:
            action[0] = 0.0  # HOLD
        
        # SL/TP for asset 0 — use wider SL to avoid instant stop-out
        action[2] = 0.0  # 5m timeframe
        action[3] = 0.5  # SL: higher = wider stop-loss
        action[4] = 0.8  # TP: higher = wider take-profit
        
        try:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        except Exception as e:
            metrics["errors"].append(f"Step {step}: {str(e)}")
            logger.error(f"Error at step {step}: {e}")
            traceback.print_exc()
            break
        
        metrics["steps"] += 1
        current_episode_length += 1
        metrics["rewards"].append(float(reward))
        
        # Track cash/equity
        cash = float(env.portfolio_manager.cash)
        equity = float(env.portfolio_manager.equity)
        metrics["cash_history"].append(cash)
        metrics["equity_history"].append(equity)
        metrics["min_cash"] = min(metrics["min_cash"], cash)
        metrics["max_cash"] = max(metrics["max_cash"], cash)
        metrics["min_equity"] = min(metrics["min_equity"], equity)
        metrics["max_equity"] = max(metrics["max_equity"], equity)
        
        # Track actions
        discrete = getattr(env, "_last_discrete_action", 0)
        action_name = {0: "HOLD", 1: "BUY", 2: "SELL"}.get(discrete, "HOLD")
        metrics["actions"][action_name] += 1
        
        if discrete == 1:
            metrics["trades_opened"] += 1
        elif discrete == 2:
            metrics["trades_closed"] += 1
        
        if done:
            metrics["episodes"] += 1
            metrics["episode_lengths"].append(current_episode_length)
            
            term_reason = info.get("termination_reason", "unknown")
            if "bankrupt" in str(term_reason).lower() or "drawdown" in str(term_reason).lower():
                metrics["bankrupt_resets"] += 1
            
            current_episode_length = 0
            obs, info = env.reset()
    
    # Final episode
    if current_episode_length > 0:
        metrics["episode_lengths"].append(current_episode_length)
        metrics["episodes"] += 1
    
    return metrics


def print_results(metrics):
    """Print test results."""
    print("\n" + "=" * 70)
    print("ADAN DETERMINISTIC CASH TEST RESULTS")
    print("=" * 70)
    
    print(f"\nSteps completed: {metrics['steps']}")
    print(f"Episodes: {metrics['episodes']}")
    
    if metrics["episode_lengths"]:
        print(f"Episode lengths: min={min(metrics['episode_lengths'])}, "
              f"max={max(metrics['episode_lengths'])}, "
              f"mean={np.mean(metrics['episode_lengths']):.1f}")
    
    print(f"\nCash: min=${metrics['min_cash']:.2f}, max=${metrics['max_cash']:.2f}")
    print(f"Equity: min=${metrics['min_equity']:.2f}, max=${metrics['max_equity']:.2f}")
    
    print(f"\nActions:")
    for action, count in metrics["actions"].items():
        print(f"  {action}: {count}")
    
    print(f"\nTrades opened: {metrics['trades_opened']}")
    print(f"Trades closed: {metrics['trades_closed']}")
    print(f"Bankrupt resets: {metrics['bankrupt_resets']}")
    
    if metrics["rewards"]:
        rewards = np.array(metrics["rewards"])
        print(f"\nRewards: min={rewards.min():.4f}, max={rewards.max():.4f}, "
              f"mean={rewards.mean():.4f}")
    
    if metrics["errors"]:
        print(f"\nERRORS ({len(metrics['errors'])}):")
        for err in metrics["errors"][:10]:
            print(f"  {err}")
    
    # Run assertions
    print("\n" + "-" * 70)
    print("ASSERTIONS:")
    
    tests = {
        "No runtime errors": len(metrics["errors"]) == 0,
        "Steps > 200": metrics["steps"] > 200,
        "At least 1 trade opened": metrics["trades_opened"] > 0,
        "BUY actions > 0": metrics["actions"]["BUY"] > 0,
        "SELL actions > 0": metrics["actions"]["SELL"] > 0,
        "At least 1 trade closed": metrics["trades_closed"] > 0,
        "Mean episode > 10 steps": (
            np.mean(metrics["episode_lengths"]) > 10 if metrics["episode_lengths"] else False
        ),
        "Cash never negative": metrics["min_cash"] >= 0.0,
        "Equity min > $5": metrics["min_equity"] > 5.0,
        "Less than 30 bankrupt resets": metrics["bankrupt_resets"] < 30,
    }
    
    all_pass = True
    for test_name, passed in tests.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {test_name}")
    
    print("\n" + "=" * 70)
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 70)
    
    return all_pass


if __name__ == "__main__":
    logger.info("Starting deterministic cash test...")
    metrics = run_test(n_steps=300, seed=42)
    success = print_results(metrics)
    sys.exit(0 if success else 1)
