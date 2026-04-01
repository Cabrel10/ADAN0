#!/usr/bin/env python3
"""
ADAN Backtest Engine — Production Version (v2)
================================================
Runs backtests using the **real** MultiAssetChunkedEnv which internally
invokes ``StateBuilder.build_observation()`` to produce the 12-dim
``context_vector`` (6 market + 6 Time2Vec cyclical).  The observation
dict therefore already contains the FiLM-compatible keys:
    '5m', '1h', '4h', 'portfolio_state', 'context_vector'

Action interpretation:
    The PPO model outputs a continuous Box(25,) action vector.
    - action[0]: direction signal (buy/sell/hold)
    - action[1..4]: per-timeframe size targets
    The environment converts these into Target-Weight orders.
    A negative action[0] while long means DYNAMIC EXIT (immediate close).

Usage:
    # Single-model backtest:
    python scripts/backtest_engine.py --model models/rl_agents/ppo_adan_simple.zip

    # Ensemble backtest (loads w1..w4 from models/<wN>/model.zip):
    python scripts/backtest_engine.py --ensemble

    # Quick validation (100 steps):
    python scripts/backtest_engine.py --steps 100
"""

import argparse
import copy
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from adan_trading_bot.common.config_loader import ConfigLoader
from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader
from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv

try:
    from adan_trading_bot.agent.feature_extractors import (
        ContextualTemporalFusionExtractor,
        WorldModelPPO,
    )
except ImportError:
    ContextualTemporalFusionExtractor = None
    WorldModelPPO = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("backtest_engine")


class BacktestEngine:
    """Run backtests with the real ADAN environment and PPO model(s)."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = ConfigLoader.load_config(config_path)
        self.output_dir = PROJECT_ROOT / "results" / "backtest"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = PROJECT_ROOT / "models"

    # ── Environment creation (mirrors training exactly) ─────────────────
    def _make_env(self, worker_idx: int = 0, data=None):
        """Create a DummyVecEnv + VecNormalize identical to training."""
        worker_key = f"w{worker_idx + 1}"
        wc = copy.deepcopy(self.config.get("workers", {}).get(worker_key, {}))
        wc["worker_id"] = worker_idx

        def _init():
            return MultiAssetChunkedEnv(
                data=data,
                config=self.config,
                worker_config=wc,
                worker_id=worker_idx,
                live_mode=False,
            )

        vec = DummyVecEnv([_init])
        gamma = self.config.get("agent", {}).get("gamma", 0.99)
        vec = VecNormalize(
            vec,
            norm_obs=True,
            norm_reward=False,   # no reward norm during backtest
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=gamma,
            training=False,      # freeze running stats
        )

        # Load saved VecNormalize stats if available
        vecnorm_path = self.models_dir / "rl_agents" / "vecnormalize.pkl"
        if vecnorm_path.exists():
            vec = VecNormalize.load(str(vecnorm_path), vec.venv)
            vec.training = False
            vec.norm_reward = False
            logger.info(f"Loaded VecNormalize stats from {vecnorm_path}")

        return vec

    # ── Data loading ────────────────────────────────────────────────────
    def _load_data(self, worker_idx: int = 0):
        """Load data through ChunkedDataLoader (respects Master Clock)."""
        wc = copy.deepcopy(self.config.get("workers", {}).get(f"w{worker_idx + 1}", {}))
        wc["worker_id"] = worker_idx
        try:
            loader = ChunkedDataLoader(
                config=self.config, worker_config=wc, worker_id=worker_idx
            )
            return loader.load_chunk(0)
        except Exception as e:
            logger.warning(f"Data loading failed: {e}. Env will load its own data.")
            return None

    # ── Model loading ───────────────────────────────────────────────────
    def load_single_model(self, model_path: str):
        """Load a single PPO model (.zip)."""
        PPOClass = WorldModelPPO if WorldModelPPO is not None else PPO
        model = PPOClass.load(model_path, device="cpu")
        logger.info(f"Loaded model from {model_path} ({type(model).__name__})")
        return model

    def load_ensemble_models(self):
        """Load w1..w4 models from models/<wN>/."""
        models = {}
        for wid in ["w1", "w2", "w3", "w4"]:
            for fname in [f"{wid}_model_final.zip", "model.zip"]:
                p = self.models_dir / wid / fname
                if p.exists():
                    models[wid] = PPO.load(str(p), device="cpu")
                    logger.info(f"Loaded {wid} from {p}")
                    break
            if wid not in models:
                # Try rl_agents directory
                alt = self.models_dir / "rl_agents" / "ppo_adan_simple.zip"
                if alt.exists():
                    models[wid] = PPO.load(str(alt), device="cpu")
                    logger.info(f"Loaded {wid} from fallback {alt}")

        if not models:
            logger.error("No models found for ensemble!")
        return models

    # ── Action interpretation ───────────────────────────────────────────
    def interpret_action(self, action_raw, has_long_position=False):
        """Interpret continuous action vector.

        Returns:
            str: 'BUY', 'SELL', 'HOLD', or 'DYNAMIC_EXIT'
        """
        if hasattr(action_raw, "__len__"):
            signal = float(action_raw[0])
        else:
            signal = float(action_raw)

        # Dynamic exit: agent signals negative while already long
        if has_long_position and signal < -0.1:
            return "DYNAMIC_EXIT"

        if signal > 0.33:
            return "BUY"
        elif signal < -0.33:
            return "SELL"
        return "HOLD"

    # ── Ensemble voting ─────────────────────────────────────────────────
    def ensemble_predict(self, models, obs, weights=None):
        """Weighted average of model predictions (continuous actions)."""
        if not models:
            return np.zeros(25, dtype=np.float32)

        if weights is None:
            weights = {k: 1.0 / len(models) for k in models}

        total = np.zeros(25, dtype=np.float32)
        w_sum = 0.0
        for wid, model in models.items():
            action, _ = model.predict(obs, deterministic=True)
            w = weights.get(wid, 0.25)
            total += action.flatten()[:25] * w
            w_sum += w
        if w_sum > 0:
            total /= w_sum
        return total

    # ── Main backtest loop ──────────────────────────────────────────────
    def run(
        self,
        model_path: str = None,
        use_ensemble: bool = False,
        max_steps: int = 2000,
        worker_idx: int = 0,
    ) -> dict:
        """Execute the backtest and return a report dict."""
        logger.info("=" * 70)
        logger.info("ADAN Backtest Engine v2 (context_vector + Target-Weight)")
        logger.info("=" * 70)

        # Load data
        data = self._load_data(worker_idx)

        # Create env (obs includes context_vector automatically)
        vec_env = self._make_env(worker_idx=worker_idx, data=data)

        # Load model(s)
        models = {}
        if use_ensemble:
            models = self.load_ensemble_models()
            if not models:
                logger.error("Cannot run ensemble: no models loaded.")
                vec_env.close()
                return {"error": "no models"}
        elif model_path:
            models = {"single": self.load_single_model(model_path)}
        else:
            # Default: try rl_agents directory
            default_path = self.models_dir / "rl_agents" / "ppo_adan_simple.zip"
            if default_path.exists():
                models = {"single": self.load_single_model(str(default_path))}
            else:
                logger.error(f"No model found at {default_path}")
                vec_env.close()
                return {"error": "no model"}

        # Run
        obs = vec_env.reset()
        total_reward = 0.0
        rewards = []
        step = 0
        done = False
        t0 = time.time()

        logger.info(f"Running backtest for up to {max_steps} steps...")

        while not done and step < max_steps:
            if use_ensemble:
                action = self.ensemble_predict(models, obs)
            else:
                model = list(models.values())[0]
                action, _ = model.predict(obs, deterministic=True)

            obs, reward, dones, infos = vec_env.step(action.reshape(1, -1) if len(action.shape) == 1 else action)
            r = float(reward[0]) if hasattr(reward, "__len__") else float(reward)
            total_reward += r
            rewards.append(r)
            step += 1

            if dones[0] if hasattr(dones, "__len__") else dones:
                done = True

            if step % 200 == 0:
                logger.info(
                    f"  Step {step}/{max_steps}: "
                    f"cum_reward={total_reward:.4f}, "
                    f"last_reward={r:.4f}"
                )

        elapsed = time.time() - t0
        logger.info(f"Backtest completed: {step} steps in {elapsed:.1f}s")

        # Extract portfolio info from inner env
        try:
            inner_env = vec_env.venv.envs[0]
            pm = inner_env.portfolio_manager
            final_value = float(pm.get_portfolio_value())
            initial_cap = float(pm.initial_capital)
            trades = list(pm.trade_log) if hasattr(pm, "trade_log") else []
        except Exception as e:
            logger.warning(f"Could not extract portfolio info: {e}")
            final_value = 0.0
            initial_cap = 20.0
            trades = []

        vec_env.close()

        # Report
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_steps": step,
            "total_reward": float(total_reward),
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "trades_count": len(trades),
            "initial_balance": initial_cap,
            "final_balance": final_value,
            "return_pct": ((final_value - initial_cap) / max(initial_cap, 1e-8)) * 100,
            "elapsed_seconds": elapsed,
            "fps": step / max(elapsed, 1e-8),
        }

        # Save report
        report_path = self.output_dir / "backtest_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info("=" * 70)
        logger.info(f"BACKTEST REPORT")
        logger.info(f"  Steps:   {step}")
        logger.info(f"  Trades:  {len(trades)}")
        logger.info(f"  Return:  {report['return_pct']:.2f}%")
        logger.info(f"  Reward:  {total_reward:.4f}")
        logger.info(f"  Report:  {report_path}")
        logger.info("=" * 70)

        return report


def main():
    parser = argparse.ArgumentParser(description="ADAN Backtest Engine v2")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--model", default=None, help="Path to single model .zip")
    parser.add_argument("--ensemble", action="store_true", help="Use ensemble w1-w4")
    parser.add_argument("--steps", type=int, default=2000, help="Max backtest steps")
    parser.add_argument("--worker", type=int, default=0, help="Worker index (0-3)")
    args = parser.parse_args()

    engine = BacktestEngine(config_path=args.config)
    report = engine.run(
        model_path=args.model,
        use_ensemble=args.ensemble,
        max_steps=args.steps,
        worker_idx=args.worker,
    )
    return 0 if "error" not in report else 1


if __name__ == "__main__":
    sys.exit(main())
