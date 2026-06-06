#!/usr/bin/env python3
"""
REAL Pipeline Integration Test
===============================
Uses ACTUAL source modules — no mocks, no mini-PPO, no shortcuts.
Tests: ChunkedDataLoader → MultiAssetChunkedEnv → SB3 PPO → training loop.

This is what runs in CI and proves the pipeline works end-to-end.
Ray Tune PBT is excluded (requires 8+ CPUs and 16+ GB RAM) but all components
it orchestrates are tested individually with real data.
"""

import sys
import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime

# ─── Path setup ───
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ["PYTHONPATH"] = str(PROJECT_ROOT / "src")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("real_pipeline_test")

# ─── REAL imports from adan_trading_bot ───
from adan_trading_bot.common.config_loader import ConfigLoader
from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader
from adan_trading_bot.data_processing.feature_engineer import FeatureEngineer
from adan_trading_bot.data_processing.state_builder import StateBuilder
from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
from adan_trading_bot.environment.reward_calculator import RewardCalculator
from adan_trading_bot.agent.feature_extractors import ContextualTemporalFusionExtractor

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback


class ProgressCallback(BaseCallback):
    """Minimal callback: logs every N steps."""
    def __init__(self, log_interval=500, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval
    def _on_step(self):
        if self.num_timesteps % self.log_interval == 0:
            infos = self.locals.get("infos", [])
            rewards = self.locals.get("rewards", [0])
            logger.info(f"  step={self.num_timesteps} | reward={np.mean(rewards):.4f}")
        return True


def test_data_loading(config):
    """TEST 1: ChunkedDataLoader loads real parquet data."""
    logger.info("=" * 60)
    logger.info("TEST 1: ChunkedDataLoader — real data loading")
    
    worker_config = config.get("workers", {}).get("w1", {})
    worker_config["worker_id"] = 0
    
    loader = ChunkedDataLoader(config=config, worker_config=worker_config, worker_id=0)
    data = loader.load_chunk(0)
    
    assert isinstance(data, dict), f"Expected dict, got {type(data)}"
    assert "BTCUSDT" in data, f"BTCUSDT not in keys: {data.keys()}"
    
    # Data is nested: data["BTCUSDT"]["5m"], data["BTCUSDT"]["1h"], etc.
    # Or flat: data["BTCUSDT/5m"], etc.
    btc_data = data["BTCUSDT"]
    if isinstance(btc_data, dict):
        tf_found = set(btc_data.keys())
        logger.info(f"  Nested structure: BTCUSDT has {tf_found}")
        assert len(tf_found) >= 2, f"Expected >= 2 timeframes, got {tf_found}"
        for tf, df in btc_data.items():
            assert isinstance(df, pd.DataFrame), f"BTCUSDT/{tf}: not DataFrame"
            assert len(df) > 100, f"BTCUSDT/{tf}: only {len(df)} rows"
            assert "close" in df.columns, f"BTCUSDT/{tf}: missing 'close' column"
            logger.info(f"  BTCUSDT/{tf}: {df.shape}")
    else:
        # Flat structure
        assert isinstance(btc_data, pd.DataFrame), f"BTCUSDT: not DataFrame"
        logger.info(f"  BTCUSDT: {btc_data.shape}")
        tf_found = {"flat"}
    
    logger.info(f"  ✓ TEST 1 PASSED: {len(data)} timeframes loaded")
    return data


def test_environment_creation(config, preloaded_data):
    """TEST 2: MultiAssetChunkedEnv instantiates with real data."""
    logger.info("=" * 60)
    logger.info("TEST 2: MultiAssetChunkedEnv — real environment creation")
    
    import copy
    worker_config = copy.deepcopy(config.get("workers", {}).get("w1", {}))
    worker_config["worker_id"] = 0
    
    env = MultiAssetChunkedEnv(
        data=preloaded_data,
        config=config,
        worker_config=worker_config,
        worker_id=0,
        live_mode=False,
    )
    
    obs, info = env.reset()
    assert obs is not None, "Reset returned None obs"
    
    if isinstance(obs, dict):
        logger.info(f"  Obs type: dict with keys {list(obs.keys())}")
        for k, v in obs.items():
            if isinstance(v, np.ndarray):
                logger.info(f"    {k}: shape={v.shape}, dtype={v.dtype}")
                assert not np.any(np.isnan(v)), f"NaN in obs[{k}]"
                assert not np.any(np.isinf(v)), f"Inf in obs[{k}]"
    else:
        logger.info(f"  Obs type: {type(obs)}, shape={getattr(obs, 'shape', '?')}")
    
    # Step test
    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info2 = env.step(action)
    assert obs2 is not None, "Step returned None obs"
    assert isinstance(reward, (float, int, np.floating)), f"Bad reward type: {type(reward)}"
    
    logger.info(f"  Action space: {env.action_space}")
    logger.info(f"  Obs space: {env.observation_space}")
    logger.info(f"  Sample reward: {reward:.6f}")
    logger.info(f"  ✓ TEST 2 PASSED: Environment operational")
    
    env.close()
    return env


def test_vectorized_env(config, preloaded_data):
    """TEST 3: DummyVecEnv + VecNormalize wrapping."""
    logger.info("=" * 60)
    logger.info("TEST 3: VecEnv + VecNormalize — SB3 integration")
    
    import copy
    
    def make_env():
        wc = copy.deepcopy(config.get("workers", {}).get("w1", {}))
        wc["worker_id"] = 0
        return MultiAssetChunkedEnv(
            data=preloaded_data, config=config, worker_config=wc,
            worker_id=0, live_mode=False,
        )
    
    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
    
    obs = vec_env.reset()
    logger.info(f"  VecEnv obs type: {type(obs)}")
    if isinstance(obs, dict):
        for k, v in obs.items():
            logger.info(f"    {k}: shape={v.shape}")
    
    # Multi-step test
    for i in range(5):
        action = [vec_env.action_space.sample()]
        obs, rewards, dones, infos = vec_env.step(action)
    
    logger.info(f"  5 steps OK, last reward={rewards[0]:.6f}")
    logger.info(f"  ✓ TEST 3 PASSED: VecNormalize operational")
    
    return vec_env


def test_ppo_training(vec_env, config, total_steps=2048):
    """TEST 4: REAL SB3 PPO training with ContextualTemporalFusionExtractor."""
    logger.info("=" * 60)
    logger.info(f"TEST 4: SB3 PPO training — {total_steps} steps with CTFE")
    
    agent_cfg = config.get("agent", {})
    fe_kwargs = agent_cfg.get("features_extractor_kwargs", {})
    
    policy_kwargs = {}
    if ContextualTemporalFusionExtractor is not None:
        policy_kwargs["features_extractor_class"] = ContextualTemporalFusionExtractor
        valid_keys = {"features_dim", "context_dim", "cnn_hidden", "dropout"}
        safe_fe = {k: v for k, v in fe_kwargs.items() if k in valid_keys}
        safe_fe.setdefault("context_dim", 14)
        policy_kwargs["features_extractor_kwargs"] = safe_fe
        policy_kwargs["share_features_extractor"] = True
        logger.info(f"  Using ContextualTemporalFusionExtractor: {safe_fe}")
    
    n_steps = min(512, total_steps // 2)
    batch_size = min(64, n_steps)
    # Ensure divisibility
    if n_steps % batch_size != 0:
        batch_size = max(1, n_steps // (n_steps // batch_size))
    
    model = PPO(
        policy="MultiInputPolicy",
        env=vec_env,
        device="cpu",
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs if policy_kwargs else None,
        verbose=1,
        seed=42,
    )
    
    logger.info(f"  PPO model created: n_steps={n_steps}, batch={batch_size}")
    logger.info(f"  Policy: {model.policy.__class__.__name__}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.policy.parameters())
    trainable_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    logger.info(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    callback = ProgressCallback(log_interval=500)
    
    t0 = time.time()
    model.learn(total_timesteps=total_steps, callback=callback, progress_bar=False)
    duration = time.time() - t0
    
    logger.info(f"  Training completed in {duration:.1f}s ({total_steps/duration:.0f} steps/s)")
    
    # Evaluate
    obs = vec_env.reset()
    total_reward = 0
    for _ in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        total_reward += reward[0]
    
    logger.info(f"  Eval (100 steps): total_reward={total_reward:.4f}, avg={total_reward/100:.6f}")
    logger.info(f"  ✓ TEST 4 PASSED: PPO trained {total_steps} steps successfully")
    
    return model, duration


def test_reward_calculator(config):
    """TEST 5: RewardCalculator with real config values."""
    logger.info("=" * 60)
    logger.info("TEST 5: RewardCalculator — real instantiation")
    
    env_config = config.get("environment", {})
    rc = RewardCalculator(env_config)
    rc.reset_reward_state(initial_capital=1000.0)
    
    # Test C1: drawdown penalty weight (the real attribute name)
    assert hasattr(rc, 'drawdown_penalty_weight'), "Missing drawdown_penalty_weight"
    logger.info(f"  Drawdown penalty weight: {rc.drawdown_penalty_weight}")
    
    # Test C2: initial capital set by reset_reward_state
    assert hasattr(rc, '_initial_capital'), "Missing _initial_capital (C2)"
    assert rc._initial_capital == 1000.0, f"Expected 1000, got {rc._initial_capital}"
    logger.info(f"  Initial capital: {rc._initial_capital}")
    
    # Test anti-hack parameters
    assert hasattr(rc, '_alpha'), "Missing _alpha (loss penalty multiplier)"
    assert hasattr(rc, '_delta'), "Missing _delta (failsafe multiplier)"
    assert hasattr(rc, '_consecutive_losses'), "Missing _consecutive_losses"
    logger.info(f"  Anti-hack params: alpha={rc._alpha}, delta={rc._delta}, gamma_streak={rc._gamma_streak}")
    
    # Test reward calculation using the REAL API signature:
    #   calculate(portfolio_metrics, trade_pnl, action, chunk_id=None, ...)
    portfolio = {
        "portfolio_value": 950.0,
        "balance": 950.0,
        "total_commission": 0.0,
        "closed_positions": [],
    }
    reward = rc.calculate(
        portfolio_metrics=portfolio,
        trade_pnl=0.0,
        action=0,
    )
    
    assert isinstance(reward, (float, int, np.floating)), f"Bad reward type: {type(reward)}"
    assert np.isfinite(reward), f"Non-finite reward: {reward}"
    logger.info(f"  Sample reward (HOLD, no trade): {reward:.6f}")
    
    # Test with a losing trade
    reward_loss = rc.calculate(
        portfolio_metrics={"portfolio_value": 900.0, "balance": 900.0, "total_commission": 0.5, "closed_positions": [{}]},
        trade_pnl=-50.0,
        action=2,  # SELL
    )
    assert reward_loss < 0, f"Losing trade should give negative reward, got {reward_loss}"
    logger.info(f"  Sample reward (loss -$50): {reward_loss:.6f} (negative ✓)")
    
    # Test with a winning trade
    rc.reset_reward_state(initial_capital=1000.0)  # Reset for clean test
    reward_win = rc.calculate(
        portfolio_metrics={"portfolio_value": 1050.0, "balance": 1050.0, "total_commission": 0.5, "closed_positions": [{}]},
        trade_pnl=50.0,
        action=1,  # BUY
    )
    assert reward_win > 0, f"Winning trade should give positive reward, got {reward_win}"
    logger.info(f"  Sample reward (win +$50): {reward_win:.6f} (positive ✓)")
    
    logger.info(f"  ✓ TEST 5 PASSED: RewardCalculator operational — anti-hack verified")
    return rc


def test_state_builder(config):
    """TEST 6: StateBuilder with real config — same construction as MultiAssetChunkedEnv."""
    logger.info("=" * 60)
    logger.info("TEST 6: StateBuilder — real tensor construction")
    
    # StateBuilder expects features_config (dict of tf→feature_list), NOT the full config.
    # This mirrors how MultiAssetChunkedEnv creates it.
    sb = StateBuilder(
        features_config=None,  # Uses default 21-feature config per timeframe
        window_sizes={"5m": 20, "1h": 20, "4h": 20},
        include_portfolio_state=True,
        normalize=True,
    )
    
    # Verify it has the expected timeframes
    assert hasattr(sb, 'timeframes'), "Missing timeframes attribute"
    assert set(sb.timeframes) == {"5m", "1h", "4h"}, f"Bad timeframes: {sb.timeframes}"
    logger.info(f"  Timeframes: {sb.timeframes}")
    
    # Verify features per timeframe = 21
    for tf in sb.timeframes:
        n_feat = len(sb.features_config[tf])
        assert n_feat == 21, f"{tf}: expected 21 features, got {n_feat}"
        logger.info(f"  {tf}: {n_feat} features")
    
    logger.info(f"  ✓ TEST 6 PASSED: StateBuilder operational — 3 TFs × 21 features")
    return sb


def main():
    logger.info("=" * 70)
    logger.info("REAL PIPELINE INTEGRATION TEST")
    logger.info(f"Date: {datetime.now().isoformat()}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    logger.info("=" * 70)
    
    # Load config
    config = ConfigLoader.load_config(str(PROJECT_ROOT / "config" / "config.yaml"))
    
    # Reduce assets to BTC only for CI
    config["data"]["assets"] = ["BTCUSDT"]
    config["data"]["include"] = ["BTCUSDT"]
    
    results = {}
    all_passed = True
    
    # TEST 1: Data loading
    try:
        data = test_data_loading(config)
        results["T1_data_loading"] = {"passed": True}
    except Exception as e:
        logger.error(f"TEST 1 FAILED: {e}")
        results["T1_data_loading"] = {"passed": False, "error": str(e)}
        all_passed = False
        data = None
    
    if data is None:
        logger.error("Cannot continue without data")
        return False
    
    # TEST 2: Environment creation
    try:
        env = test_environment_creation(config, data)
        results["T2_environment"] = {"passed": True}
    except Exception as e:
        logger.error(f"TEST 2 FAILED: {e}")
        results["T2_environment"] = {"passed": False, "error": str(e)}
        all_passed = False
    
    # TEST 3: VecEnv
    try:
        vec_env = test_vectorized_env(config, data)
        results["T3_vecenv"] = {"passed": True}
    except Exception as e:
        logger.error(f"TEST 3 FAILED: {e}")
        results["T3_vecenv"] = {"passed": False, "error": str(e)}
        all_passed = False
        vec_env = None
    
    # TEST 4: PPO Training
    if vec_env is not None:
        try:
            model, duration = test_ppo_training(vec_env, config, total_steps=2048)
            results["T4_ppo_training"] = {"passed": True, "duration": duration}
        except Exception as e:
            logger.error(f"TEST 4 FAILED: {e}", exc_info=True)
            results["T4_ppo_training"] = {"passed": False, "error": str(e)}
            all_passed = False
        finally:
            vec_env.close()
    
    # TEST 5: RewardCalculator
    try:
        rc = test_reward_calculator(config)
        results["T5_reward_calc"] = {"passed": True}
    except Exception as e:
        logger.error(f"TEST 5 FAILED: {e}")
        results["T5_reward_calc"] = {"passed": False, "error": str(e)}
        all_passed = False
    
    # TEST 6: StateBuilder
    try:
        sb = test_state_builder(config)
        results["T6_state_builder"] = {"passed": True}
    except Exception as e:
        logger.error(f"TEST 6 FAILED: {e}")
        results["T6_state_builder"] = {"passed": False, "error": str(e)}
        all_passed = False
    
    # Summary
    passed = sum(1 for v in results.values() if v["passed"])
    total = len(results)
    
    logger.info("\n" + "=" * 70)
    logger.info(f"RESULTS: {passed}/{total} PASSED")
    for name, res in results.items():
        status = "✓" if res["passed"] else "✗"
        logger.info(f"  {status} {name}")
        if not res["passed"]:
            logger.info(f"      Error: {res.get('error', 'unknown')}")
    
    if all_passed:
        logger.info("\nALL REAL PIPELINE TESTS PASSED ✓")
    else:
        logger.info("\nSOME TESTS FAILED")
    logger.info("=" * 70)
    
    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "pytorch_version": torch.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "results": results,
        "all_passed": all_passed,
        "passed_count": passed,
        "total_tests": total,
    }
    os.makedirs("data/validation", exist_ok=True)
    with open("data/validation/real_pipeline_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
