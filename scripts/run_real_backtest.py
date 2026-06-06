#!/usr/bin/env python3
"""ADAN0 Real Backtest — uses ONLY MultiAssetChunkedEnv.

Loads a trained PPO checkpoint and runs deterministic evaluation on
the REAL environment with all production modules:
  - MultiAssetChunkedEnv
  - DynamicBehaviorEngine (HMM 3-state, DBE modulation)
  - ExogenousRegimeOracle
  - PortfolioManager (capital tiers, SL/TP, fee gate)
  - RewardCalculator (symlog compression)

Usage:
    python scripts/run_real_backtest.py \\
        --checkpoint checkpoints/ppo_adan0_sandbox_0k.zip \\
        --steps 2000

Output:
    JSON report to stdout with backtest metrics.
"""

import argparse
import copy
import json
import logging
import os
import sys
import time
from pathlib import Path

# Ensure src/ is importable
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

PROJECT_ROOT = _SCRIPT_DIR.parent

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from adan_trading_bot.common.config_loader import ConfigLoader
from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader
from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv


logger = logging.getLogger(__name__)


def run_backtest(
    checkpoint_path: str,
    config_path: str = None,
    steps: int = 2000,
    vecnorm_path: str = None,
    initial_capital: float = None,
    deterministic: bool = True,
) -> dict:
    """Run a deterministic backtest using the REAL MultiAssetChunkedEnv.

    Args:
        checkpoint_path: Path to PPO checkpoint (.zip).
        config_path: Path to config.yaml.
        steps: Number of environment steps to run.
        vecnorm_path: Path to VecNormalize stats (.pkl).
        initial_capital: Starting capital in USDT (None = read from config).
        deterministic: Use deterministic policy (no exploration noise).

    Returns:
        dict with backtest metrics.
    """
    # ── Load config ──
    if config_path is None:
        config_path = str(PROJECT_ROOT / "config" / "config.yaml")
    config = ConfigLoader.load_config(config_path)

    # Read initial_capital from config if not provided (Rule #16: no hardcoding)
    if initial_capital is None:
        sandbox_cfg = config.get("sandbox", {})
        initial_capital = float(sandbox_cfg.get("initial_capital",
                                config.get("environment", {}).get("initial_balance", 20.50)))

    if "environment" not in config:
        config["environment"] = {}
    config["environment"]["initial_capital"] = initial_capital

    # ── Worker config (w1 = scalper) ──
    worker_config = copy.deepcopy(config.get("workers", {}).get("w1", {}))
    worker_config["worker_id"] = 0
    # BACKTEST uses TEST data (out-of-sample) — NOT train data
    worker_config["data_split_override"] = "test"
    worker_config.setdefault(
        "timeframes",
        config.get("data", {}).get("timeframes", ["5m", "1h", "4h"]),
    )
    worker_config.setdefault(
        "assets",
        config.get("environment", {}).get("assets", ["BTCUSDT"]),
    )

    # ── Load data ──
    loader = ChunkedDataLoader(
        config=config, worker_config=worker_config, worker_id=0
    )
    data = loader.load_chunk(0)
    logger.info(
        f"[BACKTEST] Data loaded: {list(data.keys())} assets, "
        f"timeframes={list(list(data.values())[0].keys()) if data else 'none'}"
    )

    # ── Create the REAL MultiAssetChunkedEnv ──
    env = MultiAssetChunkedEnv(
        data=data,
        config=config,
        worker_config=worker_config,
        worker_id=0,
        live_mode=False,
    )

    # ── Wrap in VecNormalize ──
    vec_env = DummyVecEnv([lambda: env])
    gamma = config.get("agent", {}).get("gamma", 0.99)

    if vecnorm_path and os.path.exists(vecnorm_path):
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        logger.info(f"[BACKTEST] VecNormalize loaded from {vecnorm_path}")
    else:
        # Anomalie 8 fix: exclude context_vector from normalization
        _norm_keys = ["5m", "1h", "4h", "portfolio_state"]
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=gamma,
            training=False,
            norm_obs_keys=_norm_keys,
        )
        logger.info("[BACKTEST] VecNormalize created (no pre-trained stats, context_vector excluded)")

    # ── Load PPO model ──
    model = PPO.load(checkpoint_path, env=vec_env, device="cpu")
    logger.info(f"[BACKTEST] PPO model loaded from {checkpoint_path}")

    # ── Run backtest ──
    obs = vec_env.reset()
    total_reward = 0.0
    step_rewards = []
    portfolio_values = []
    trade_count = 0

    t0 = time.time()
    for step_i in range(steps):
        action, _states = model.predict(obs, deterministic=deterministic)
        obs, reward, done, info = vec_env.step(action)
        total_reward += float(reward[0])
        step_rewards.append(float(reward[0]))

        # Extract portfolio value
        if isinstance(info, (list, tuple)) and len(info) > 0:
            pv = info[0].get("portfolio_value", info[0].get("portfolio", {}).get("value", 0.0))
            if pv > 0:
                portfolio_values.append(float(pv))

        if done[0]:
            obs = vec_env.reset()
            # Count trades from info
            if isinstance(info, (list, tuple)) and len(info) > 0:
                tc = info[0].get("total_trades", info[0].get("n_trades", 0))
                if isinstance(tc, (int, float)):
                    trade_count += int(tc)

    elapsed = time.time() - t0

    # ── Collect final metrics from env ──
    try:
        env_info = env.get_info() if hasattr(env, "get_info") else {}
    except Exception:
        env_info = {}

    final_pv = portfolio_values[-1] if portfolio_values else initial_capital
    n_trades = env_info.get(
        "total_trades",
        env_info.get("n_trades", trade_count),
    )

    # Compute returns for Sharpe
    if len(step_rewards) > 1:
        returns = np.array(step_rewards)
        mean_ret = float(np.mean(returns))
        std_ret = float(np.std(returns))
        sharpe = float(mean_ret / std_ret * np.sqrt(252)) if std_ret > 1e-9 else 0.0
    else:
        mean_ret = 0.0
        std_ret = 0.0
        sharpe = 0.0

    # Max drawdown
    if portfolio_values:
        pv_arr = np.array(portfolio_values)
        running_max = np.maximum.accumulate(pv_arr)
        drawdowns = (running_max - pv_arr) / np.maximum(running_max, 1e-9)
        max_dd = float(np.max(drawdowns))
    else:
        max_dd = 0.0

    report = {
        "backtest_steps": steps,
        "elapsed_seconds": round(elapsed, 1),
        "checkpoint": checkpoint_path,
        "initial_capital": initial_capital,
        "final_portfolio_value": round(final_pv, 2),
        "total_return_pct": round((final_pv - initial_capital) / max(initial_capital, 1e-8) * 100, 2),
        "total_reward": round(total_reward, 4),
        "mean_step_reward": round(mean_ret, 6),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "total_trades": n_trades,
        "env_class": "MultiAssetChunkedEnv",
        "deterministic": deterministic,
        "hmm_fitted": getattr(env.dbe, "_hmm_fitted", False) if hasattr(env, "dbe") else False,
    }

    logger.info(f"[BACKTEST] Complete: {json.dumps(report, indent=2, default=str)}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ADAN0 Real Backtest")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/ppo_adan0_sandbox_0k.zip",
        help="Path to PPO checkpoint (.zip)",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--steps", type=int, default=2000, help="Backtest steps")
    parser.add_argument(
        "--vecnorm",
        type=str,
        default="checkpoints/vecnormalize_sandbox.pkl",
        help="Path to VecNormalize stats (.pkl)",
    )
    parser.add_argument(
        "--capital", type=float, default=20.50, help="Initial capital (USDT)"
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy (default: deterministic)",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    report = run_backtest(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        steps=args.steps,
        vecnorm_path=args.vecnorm,
        initial_capital=args.capital,
        deterministic=not args.stochastic,
    )
    print(json.dumps(report, indent=2, default=str))
