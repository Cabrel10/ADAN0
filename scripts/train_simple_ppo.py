#!/usr/bin/env python3
"""
Simple PPO training script for ADAN — direct SB3 without Ray Tune.

Usage:
    cd ADAN/bot && python scripts/train_simple_ppo.py --steps 30000
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

# .env loading
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

# ADAN imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from adan_trading_bot.common.config_loader import ConfigLoader
from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader
from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv

try:
    from adan_trading_bot.agent.feature_extractors import ContextualTemporalFusionExtractor
except ImportError:
    ContextualTemporalFusionExtractor = None

try:
    from adan_trading_bot.agent.feature_extractors import WorldModelPPO
except ImportError:
    WorldModelPPO = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("train_simple_ppo")


class TrainingMetricsCallback(BaseCallback):
    """Logs training metrics at regular intervals."""

    def __init__(self, log_interval=500, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.start_time = time.time()
        self.best_reward = float("-inf")

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_interval == 0:
            elapsed = time.time() - self.start_time
            fps = self.num_timesteps / max(elapsed, 1)

            # Try to get episode info
            mean_reward = 0.0
            try:
                ep_info = self.model.ep_info_buffer
                if ep_info and len(ep_info) > 0:
                    mean_reward = float(np.mean([e["r"] for e in ep_info]))
            except Exception:
                pass

            if mean_reward > self.best_reward:
                self.best_reward = mean_reward

            logger.info(
                f"[PROGRESS] Steps={self.num_timesteps:,} | "
                f"FPS={fps:.1f} | "
                f"ep_rew_mean={mean_reward:.4f} | "
                f"best_rew={self.best_reward:.4f} | "
                f"elapsed={elapsed:.0f}s"
            )

            # Log PPO-specific metrics
            try:
                if hasattr(self.model, "logger") and hasattr(self.model.logger, "name_to_value"):
                    vals = self.model.logger.name_to_value
                    pl = vals.get("train/policy_gradient_loss", 0)
                    vl = vals.get("train/value_loss", 0)
                    ev = vals.get("train/explained_variance", 0)
                    logger.info(
                        f"[PPO_METRICS] policy_gradient_loss={pl:.6f} | "
                        f"value_loss={vl:.4f} | "
                        f"explained_variance={ev:.4f}"
                    )
            except Exception:
                pass

            # Log auxiliary world-model prediction magnitude
            try:
                if hasattr(self.model, '_aux_loss_history') and self.model._aux_loss_history:
                    aux_mag = self.model._aux_loss_history[-1]
                    logger.info(
                        f"[AUX_WORLD_MODEL] aux_pred_magnitude={aux_mag:.6f}"
                    )
            except Exception:
                pass

        return True


def make_env(config, worker_idx=0, preloaded_data=None):
    """Create a simple DummyVecEnv with VecNormalize."""
    worker_key = f"w{worker_idx + 1}"
    worker_config = copy.deepcopy(config.get("workers", {}).get(worker_key, {}))
    worker_config["worker_id"] = worker_idx

    def _init():
        return MultiAssetChunkedEnv(
            data=preloaded_data,
            config=config,
            worker_config=worker_config,
            worker_id=worker_idx,
            live_mode=False,
        )

    vec_env = DummyVecEnv([_init])
    gamma = config.get("agent", {}).get("gamma", 0.99)
    vec_env = VecNormalize(
        vec_env, norm_obs=True, norm_reward=True,
        clip_obs=10.0, clip_reward=10.0, gamma=gamma, training=True,
    )
    return vec_env


def main():
    parser = argparse.ArgumentParser(description="Simple ADAN PPO Training")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    # Load config
    config = ConfigLoader.load_config(args.config)
    agent_cfg = config.get("agent", {})

    # Load data
    worker_key = "w1"
    worker_config = copy.deepcopy(config.get("workers", {}).get(worker_key, {}))
    worker_config["worker_id"] = 0
    preloaded_data = None
    try:
        loader = ChunkedDataLoader(config=config, worker_config=worker_config, worker_id=0)
        preloaded_data = loader.load_chunk(0)
        logger.info(f"Loaded data chunk: {type(preloaded_data).__name__}")
    except Exception as e:
        logger.warning(f"Failed to load data: {e}")

    # Create env
    vec_env = make_env(config, worker_idx=0, preloaded_data=preloaded_data)
    logger.info(f"Environment created. Obs space: {vec_env.observation_space}")
    logger.info(f"Action space: {vec_env.action_space}")

    # Policy kwargs
    policy_kwargs = {}
    if ContextualTemporalFusionExtractor is not None:
        # Only pass valid kwargs for ContextualTemporalFusionExtractor
        fe_kwargs = agent_cfg.get("features_extractor_kwargs", {})
        valid_fe_kwargs = {}
        for k in ("features_dim", "context_dim", "cnn_hidden", "dropout"):
            if k in fe_kwargs:
                valid_fe_kwargs[k] = fe_kwargs[k]
        policy_kwargs = {
            "features_extractor_class": ContextualTemporalFusionExtractor,
            "features_extractor_kwargs": valid_fe_kwargs,
        }

    # Create PPO model
    n_steps = agent_cfg.get("n_steps", 512)
    batch_size = min(agent_cfg.get("batch_size", 64), n_steps)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create PPO model (use WorldModelPPO if available for aux forward-prediction)
    PPOClass = WorldModelPPO if WorldModelPPO is not None else PPO
    logger.info(f"Using PPO class: {PPOClass.__name__}")

    model = PPOClass(
        "MultiInputPolicy",
        vec_env,
        device=device,
        learning_rate=float(agent_cfg.get("learning_rate", 3e-4)),
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=agent_cfg.get("n_epochs", 10),
        gamma=float(agent_cfg.get("gamma", 0.99)),
        gae_lambda=float(agent_cfg.get("gae_lambda", 0.95)),
        clip_range=float(agent_cfg.get("clip_range", 0.2)),
        ent_coef=float(agent_cfg.get("ent_coef", 0.01)),
        vf_coef=float(agent_cfg.get("vf_coef", 0.5)),
        max_grad_norm=float(agent_cfg.get("max_grad_norm", 0.5)),
        policy_kwargs=policy_kwargs if policy_kwargs else None,
        verbose=1,
        seed=42,
    )

    logger.info("=" * 70)
    logger.info(f"🔥 ADAN Simple PPO Training")
    logger.info(f"   Steps: {args.steps:,}, n_steps: {n_steps}, batch_size: {batch_size}")
    logger.info(f"   Device: {device}")
    logger.info(f"   Initial balance: {config.get('portfolio', {}).get('initial_balance', '?')}")
    logger.info("=" * 70)

    # Train
    callback = TrainingMetricsCallback(log_interval=500)
    try:
        model.learn(
            total_timesteps=args.steps,
            callback=callback,
            progress_bar=False,
        )
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)

    # Save model
    save_dir = str(PROJECT_ROOT / "models" / "rl_agents")
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "ppo_adan_simple")
    model.save(model_path)
    vec_env.save(os.path.join(save_dir, "vecnormalize.pkl"))
    logger.info(f"Model saved to {model_path}")

    # Summary
    logger.info("=" * 70)
    logger.info("🎯 TRAINING COMPLETE")
    logger.info(f"   Total steps: {args.steps:,}")
    logger.info(f"   Best reward: {callback.best_reward:.4f}")
    logger.info(f"   Duration: {time.time() - callback.start_time:.0f}s")
    logger.info("=" * 70)

    vec_env.close()


if __name__ == "__main__":
    main()
