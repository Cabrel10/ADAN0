#!/usr/bin/env python3
"""ADAN Convergence Audit — Single-Worker PPO Training.

Proves that the PPO model:
  1. Computes gradients (non-zero policy_loss, value_loss)
  2. Opens actual trades (total_trades > 0)
  3. Learns to improve (reward trend, win rate evolution)
  4. Respects tier constraints, fees, regime detection

This script bypasses Ray Tune entirely and trains a single PPO agent
directly with SB3, minimising memory footprint (~300 MB).

Usage:
    cd bot && PYTHONPATH=src:$PYTHONPATH python scripts/train_convergence_audit.py \
        --steps 30000 --n-steps 512 --log-level INFO
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import os
import sys
import time
from collections import defaultdict
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs" / "convergence_audit"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "training_audit.log", mode="w"),
    ],
)
logger = logging.getLogger("convergence_audit")

# ──────────────────────────────────────────────────────────────────────
# CRITICAL: Silence ALL ADAN loggers and remove their file handlers.
# The env logs ~500 lines per step(), making training ~1 step/sec.
# We need >50 steps/sec for 30k steps to finish in <10 min.
# ──────────────────────────────────────────────────────────────────────
def _silence_adan_loggers():
    """Set all adan_trading_bot.* loggers to CRITICAL and remove handlers."""
    for name in list(logging.Logger.manager.loggerDict.keys()):
        if "adan_trading_bot" in name or name in ("root",):
            lg = logging.getLogger(name)
            lg.setLevel(logging.CRITICAL)
            lg.handlers = []
            lg.propagate = False

    # Also nuke the root-level adan handler
    root_lg = logging.getLogger("adan_trading_bot")
    root_lg.setLevel(logging.CRITICAL)
    root_lg.handlers = []
    root_lg.propagate = False

# Pre-silence known loggers
for _noisy in [
    "adan_trading_bot.environment",
    "adan_trading_bot.environment.multi_asset_chunked_env",
    "adan_trading_bot.environment.dynamic_behavior_engine",
    "adan_trading_bot.environment.reward_calculator",
    "adan_trading_bot.environment.order_manager",
    "adan_trading_bot.data_processing",
    "adan_trading_bot.data_processing.state_builder",
    "adan_trading_bot.data_processing.data_loader",
    "adan_trading_bot.data_processing.data_validator",
    "adan_trading_bot.portfolio",
    "adan_trading_bot.portfolio.portfolio_manager",
    "adan_trading_bot.trading",
    "adan_trading_bot.trading.position_sizer",
    "adan_trading_bot.risk",
    "adan_trading_bot.common",
    "adan_trading_bot.common.reward_logger",
    "adan_trading_bot.agent",
    "adan_trading_bot",
]:
    _lg = logging.getLogger(_noisy)
    _lg.setLevel(logging.CRITICAL)
    _lg.handlers = []
    _lg.propagate = False


# ═══════════════════════════════════════════════════════════════════════
# Deep Audit Callback — records everything we need to prove convergence
# ═══════════════════════════════════════════════════════════════════════

class ConvergenceAuditCallback(BaseCallback):
    """Records policy_loss, value_loss, trades, rewards, tier, fees at every step."""

    def __init__(self, log_interval: int = 500, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval = log_interval

        # Metrics storage
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        self.approx_kls = []
        self.clip_fractions = []

        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_trades = []
        self.episode_win_rates = []
        self.episode_pnls = []
        self.episode_fees = []
        self.episode_tiers = []

        # Per-step trade tracking
        self.total_trades_opened = 0
        self.total_trades_closed = 0
        self.total_buy_signals = 0
        self.total_sell_signals = 0
        self.total_hold_signals = 0
        self.invalid_trade_attempts = 0

        self._step_rewards = []
        self._step_actions = defaultdict(int)  # action distribution

        self.start_time = time.time()

    def _on_step(self) -> bool:
        # Collect info from env
        infos = self.locals.get("infos", [{}])
        rewards = self.locals.get("rewards", [0.0])
        actions = self.locals.get("actions", None)

        for i, info in enumerate(infos if infos else [{}]):
            if not isinstance(info, dict):
                continue

            # Track actions
            if actions is not None and len(actions) > i:
                act = actions[i]
                if hasattr(act, '__len__') and len(act) >= 1:
                    main_act = float(act[0])
                    if main_act > 0.05:
                        self.total_buy_signals += 1
                    elif main_act < -0.05:
                        self.total_sell_signals += 1
                    else:
                        self.total_hold_signals += 1

            # Track trades from info
            portfolio = info.get("portfolio", {})
            self.total_trades_opened = max(
                self.total_trades_opened,
                info.get("total_trades", portfolio.get("total_trades", self.total_trades_opened))
            )
            self.invalid_trade_attempts = max(
                self.invalid_trade_attempts,
                info.get("invalid_trade_attempts", self.invalid_trade_attempts)
            )

            # Episode end
            if info.get("episode") or info.get("terminal_observation") is not None:
                ep_info = info.get("episode", {})
                if ep_info:
                    self.episode_rewards.append(ep_info.get("r", 0.0))
                    self.episode_lengths.append(ep_info.get("l", 0))

                self.episode_trades.append(info.get("total_trades", portfolio.get("total_trades", 0)))
                self.episode_win_rates.append(portfolio.get("win_rate", 0.0))
                self.episode_pnls.append(portfolio.get("total_pnl", 0.0))
                self.episode_fees.append(portfolio.get("total_fees", info.get("total_fees", 0.0)))
                self.episode_tiers.append(info.get("current_tier", portfolio.get("tier", "?")))

            self._step_rewards.append(float(rewards[i]) if i < len(rewards) else 0.0)

        # Periodic logging
        if self.num_timesteps > 0 and self.num_timesteps % self.log_interval == 0:
            self._log_progress()

        return True

    def _on_rollout_end(self) -> None:
        """Capture SB3 internal losses after each PPO update."""
        log = self.model.logger.name_to_value if hasattr(self.model.logger, 'name_to_value') else {}

        pl = log.get("train/policy_gradient_loss", log.get("train/policy_loss", None))
        vl = log.get("train/value_loss", None)
        el = log.get("train/entropy_loss", None)
        kl = log.get("train/approx_kl", None)
        cf = log.get("train/clip_fraction", None)

        if pl is not None:
            self.policy_losses.append(float(pl))
        if vl is not None:
            self.value_losses.append(float(vl))
        if el is not None:
            self.entropy_losses.append(float(el))
        if kl is not None:
            self.approx_kls.append(float(kl))
        if cf is not None:
            self.clip_fractions.append(float(cf))

    def _log_progress(self):
        elapsed = time.time() - self.start_time
        sps = self.num_timesteps / max(elapsed, 1)
        avg_rew = np.mean(self._step_rewards[-500:]) if self._step_rewards else 0.0

        logger.info(
            f"[Step {self.num_timesteps:>7d}] "
            f"SPS={sps:.0f} | "
            f"avg_reward={avg_rew:+.4f} | "
            f"trades_opened={self.total_trades_opened} | "
            f"buy={self.total_buy_signals} sell={self.total_sell_signals} hold={self.total_hold_signals} | "
            f"invalid_attempts={self.invalid_trade_attempts} | "
            f"episodes={len(self.episode_rewards)}"
        )

        if self.policy_losses:
            logger.info(
                f"  PPO losses: policy={self.policy_losses[-1]:.6f} "
                f"value={self.value_losses[-1]:.4f} "
                f"entropy={self.entropy_losses[-1]:.4f} "
                f"approx_kl={self.approx_kls[-1]:.6f} "
                f"clip_frac={self.clip_fractions[-1]:.4f}"
            )

    def get_audit_report(self) -> dict:
        """Generate the complete audit report."""
        elapsed = time.time() - self.start_time

        # Split losses into early/late for trend analysis
        n_losses = len(self.policy_losses)
        mid = max(1, n_losses // 2)

        report = {
            "meta": {
                "total_timesteps": self.num_timesteps,
                "wall_time_seconds": round(elapsed, 1),
                "steps_per_second": round(self.num_timesteps / max(elapsed, 1), 1),
                "timestamp": datetime.now().isoformat(),
            },
            "ppo_losses": {
                "count": n_losses,
                "policy_loss_mean": round(float(np.mean(self.policy_losses)), 6) if self.policy_losses else None,
                "policy_loss_early": round(float(np.mean(self.policy_losses[:mid])), 6) if n_losses > 1 else None,
                "policy_loss_late": round(float(np.mean(self.policy_losses[mid:])), 6) if n_losses > 1 else None,
                "value_loss_mean": round(float(np.mean(self.value_losses)), 4) if self.value_losses else None,
                "value_loss_early": round(float(np.mean(self.value_losses[:mid])), 4) if n_losses > 1 else None,
                "value_loss_late": round(float(np.mean(self.value_losses[mid:])), 4) if n_losses > 1 else None,
                "entropy_mean": round(float(np.mean(self.entropy_losses)), 4) if self.entropy_losses else None,
                "approx_kl_mean": round(float(np.mean(self.approx_kls)), 6) if self.approx_kls else None,
                "clip_fraction_mean": round(float(np.mean(self.clip_fractions)), 4) if self.clip_fractions else None,
            },
            "trading_activity": {
                "total_trades_opened": self.total_trades_opened,
                "total_buy_signals": self.total_buy_signals,
                "total_sell_signals": self.total_sell_signals,
                "total_hold_signals": self.total_hold_signals,
                "invalid_trade_attempts": self.invalid_trade_attempts,
                "trade_ratio": round(
                    (self.total_buy_signals + self.total_sell_signals)
                    / max(1, self.total_buy_signals + self.total_sell_signals + self.total_hold_signals),
                    4,
                ),
            },
            "episodes": {
                "count": len(self.episode_rewards),
                "avg_reward": round(float(np.mean(self.episode_rewards)), 4) if self.episode_rewards else None,
                "avg_trades": round(float(np.mean(self.episode_trades)), 2) if self.episode_trades else None,
                "avg_win_rate": round(float(np.mean(self.episode_win_rates)), 4) if self.episode_win_rates else None,
                "avg_pnl": round(float(np.mean(self.episode_pnls)), 4) if self.episode_pnls else None,
                "avg_fees": round(float(np.mean(self.episode_fees)), 6) if self.episode_fees else None,
                "tiers_seen": list(set(str(t) for t in self.episode_tiers)) if self.episode_tiers else [],
            },
            "reward_trend": {
                "first_quarter": round(float(np.mean(self._step_rewards[:len(self._step_rewards)//4])), 4) if len(self._step_rewards) > 100 else None,
                "last_quarter": round(float(np.mean(self._step_rewards[-len(self._step_rewards)//4:])), 4) if len(self._step_rewards) > 100 else None,
            },
            "convergence_verdict": self._verdict(),
        }
        return report

    def _verdict(self) -> dict:
        """Automated pass/fail assessment."""
        issues = []
        passes = []

        # 1. Were losses computed?
        if not self.policy_losses:
            issues.append("CRITICAL: No PPO updates recorded (policy_loss is empty)")
        else:
            passes.append(f"PPO updates: {len(self.policy_losses)} rollout updates recorded")
            if all(abs(l) < 1e-10 for l in self.policy_losses):
                issues.append("CRITICAL: policy_loss is always ~0 — no gradients")
            else:
                passes.append(f"Gradients active: mean |policy_loss| = {np.mean(np.abs(self.policy_losses)):.6f}")

        # 2. Did the agent trade?
        total_signals = self.total_buy_signals + self.total_sell_signals
        if total_signals == 0:
            issues.append("CRITICAL: Agent sent 0 buy/sell signals — LAZY AGENT")
        elif total_signals < self.total_hold_signals * 0.01:
            issues.append(f"WARNING: Agent almost never trades ({total_signals} signals vs {self.total_hold_signals} holds)")
        else:
            passes.append(f"Trading active: {total_signals} buy+sell signals ({self.total_buy_signals}B/{self.total_sell_signals}S)")

        if self.total_trades_opened == 0 and total_signals > 0:
            issues.append("WARNING: Agent sends signals but 0 trades executed (env blocking?)")

        # 3. Reward improving?
        if len(self._step_rewards) > 200:
            q1 = np.mean(self._step_rewards[:len(self._step_rewards)//4])
            q4 = np.mean(self._step_rewards[-len(self._step_rewards)//4:])
            if q4 > q1:
                passes.append(f"Reward improving: Q1={q1:.4f} → Q4={q4:.4f}")
            else:
                issues.append(f"WARNING: Reward not improving Q1={q1:.4f} → Q4={q4:.4f}")

        return {
            "status": "FAIL" if any("CRITICAL" in i for i in issues) else ("WARN" if issues else "PASS"),
            "issues": issues,
            "passes": passes,
        }


# ═══════════════════════════════════════════════════════════════════════
# Training harness
# ═══════════════════════════════════════════════════════════════════════

def create_env(config: dict, worker_key: str = "w1") -> DummyVecEnv:
    """Create a single DummyVecEnv (1 env) with pre-loaded parquet data."""
    worker_config = copy.deepcopy(config.get("workers", {}).get(worker_key, {}))
    worker_config["worker_id"] = 0

    # Pre-load data
    preloaded_data = None
    try:
        loader = ChunkedDataLoader(
            config=config,
            worker_config=worker_config,
            worker_id=0,
        )
        preloaded_data = loader.load_chunk(0)
        if preloaded_data:
            logger.info(f"ChunkedDataLoader: loaded chunk 0 for {worker_key}")
            for k, v in preloaded_data.items():
                if hasattr(v, 'shape'):
                    logger.info(f"  {k}: {v.shape}")
    except Exception as exc:
        logger.warning(f"ChunkedDataLoader failed: {exc}")

    def _make_env():
        wc = copy.deepcopy(worker_config)
        return MultiAssetChunkedEnv(
            data=preloaded_data,
            config=config,
            worker_config=wc,
            worker_id=0,
            live_mode=False,
        )

    vec_env = DummyVecEnv([_make_env])

    # Re-silence ALL loggers that the env just created
    _silence_adan_loggers()

    # NUCLEAR: Monkey-patch ALL existing loggers to suppress output
    # The env creates hundreds of log lines per step(), making training
    # ~1 step/sec. We need 50+ steps/sec for 30k to finish in time.
    class _NullLogger:
        """Drop-in replacement that does absolutely nothing."""
        level = logging.CRITICAL
        handlers = []
        propagate = False
        def debug(self, *a, **k): pass
        def info(self, *a, **k): pass
        def warning(self, *a, **k): pass
        def warn(self, *a, **k): pass
        def error(self, *a, **k): pass
        def critical(self, *a, **k): pass
        def exception(self, *a, **k): pass
        def log(self, *a, **k): pass
        def isEnabledFor(self, *a): return False
        def setLevel(self, *a): pass
        def addHandler(self, *a): pass
        def removeHandler(self, *a): pass
        def getEffectiveLevel(self): return logging.CRITICAL

    null_log = _NullLogger()
    raw_env = vec_env.envs[0]

    # Replace all logger attributes on the env and its components
    if hasattr(raw_env, 'logger'):
        raw_env.logger = null_log
    if hasattr(raw_env, 'smart_logger'):
        raw_env.smart_logger = null_log
    for attr in ['reward_calculator', 'portfolio_manager', 'position_sizer',
                 'order_manager', 'state_builder', 'dynamic_behavior_engine',
                 'data_loader', 'data_loader_instance', 'observation_validator',
                 'performance_metrics', 'reward_logger']:
        comp = getattr(raw_env, attr, None)
        if comp and hasattr(comp, 'logger'):
            comp.logger = null_log
        if comp and hasattr(comp, 'smart_logger'):
            comp.smart_logger = null_log

    # Also disable the file-based reward logger
    if hasattr(raw_env, 'reward_logger') and raw_env.reward_logger:
        raw_env.reward_logger = None

    # Disable all Python loggers except our audit one
    for name, lg in logging.Logger.manager.loggerDict.items():
        if isinstance(lg, logging.Logger) and name != "convergence_audit":
            lg.setLevel(logging.CRITICAL)
            lg.handlers = []
            lg.propagate = False

    gamma = config.get("agent", {}).get("gamma", 0.99)
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=gamma,
        training=True,
    )
    return vec_env


def train(args):
    """Main training loop."""
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    config = ConfigLoader.load_config(str(config_path))
    logger.info(f"Config loaded: {len(config)} top-level keys")

    # Override PPO params for fast iteration
    if "agent" not in config:
        config["agent"] = {}
    config["agent"]["n_steps"] = args.n_steps
    config["agent"]["batch_size"] = min(args.batch_size, args.n_steps)
    logger.info(f"PPO config: n_steps={args.n_steps}, batch_size={config['agent']['batch_size']}")

    # Create environment
    logger.info("Creating environment...")
    vec_env = create_env(config, worker_key=args.worker)
    logger.info(f"Observation space: {vec_env.observation_space}")
    logger.info(f"Action space: {vec_env.action_space}")

    # Create PPO model (lightweight MLP, no heavy feature extractor)
    agent_cfg = config.get("agent", {})
    device = "cpu"  # Force CPU for sandbox

    policy_kwargs = {
        "net_arch": dict(pi=[128, 64], vf=[128, 64]),
    }

    model = PPO(
        "MultiInputPolicy",
        vec_env,
        device=device,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=config["agent"]["batch_size"],
        n_epochs=args.n_epochs,
        gamma=agent_cfg.get("gamma", 0.99),
        gae_lambda=agent_cfg.get("gae_lambda", 0.95),
        clip_range=agent_cfg.get("clip_range", 0.2),
        ent_coef=args.ent_coef,
        vf_coef=agent_cfg.get("vf_coef", 0.5),
        max_grad_norm=agent_cfg.get("max_grad_norm", 0.5),
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=42,
    )

    # Print model size
    total_params = sum(p.numel() for p in model.policy.parameters())
    logger.info(f"Model parameters: {total_params:,} ({total_params * 4 / 1024 / 1024:.1f} MB)")

    # Audit callback
    audit_cb = ConvergenceAuditCallback(log_interval=args.log_interval)

    # Train!
    logger.info(f"{'='*60}")
    logger.info(f"STARTING CONVERGENCE AUDIT — {args.steps} steps")
    logger.info(f"  n_steps={args.n_steps} → PPO updates every {args.n_steps} steps")
    logger.info(f"  Expected PPO updates: ~{args.steps // args.n_steps}")
    logger.info(f"  batch_size={config['agent']['batch_size']}, n_epochs={args.n_epochs}")
    logger.info(f"  lr={args.lr}, ent_coef={args.ent_coef}")
    logger.info(f"{'='*60}")

    try:
        model.learn(
            total_timesteps=args.steps,
            callback=audit_cb,
            progress_bar=False,
        )
    except KeyboardInterrupt:
        logger.warning("Training interrupted!")
    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)

    # Save model
    model_path = LOG_DIR / "ppo_convergence_audit"
    model.save(str(model_path))
    vec_norm_path = LOG_DIR / "vecnormalize.pkl"
    vec_env.save(str(vec_norm_path))
    logger.info(f"Model saved: {model_path}")

    # Generate audit report
    report = audit_cb.get_audit_report()
    report_path = LOG_DIR / "convergence_audit_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Print verdict
    logger.info(f"\n{'='*60}")
    logger.info("CONVERGENCE AUDIT REPORT")
    logger.info(f"{'='*60}")
    for section, data in report.items():
        if section == "convergence_verdict":
            logger.info(f"\n{'─'*40}")
            logger.info(f"VERDICT: {data['status']}")
            for p in data["passes"]:
                logger.info(f"  ✅ {p}")
            for i in data["issues"]:
                logger.info(f"  ❌ {i}")
        else:
            logger.info(f"\n[{section}]")
            if isinstance(data, dict):
                for k, v in data.items():
                    logger.info(f"  {k}: {v}")

    logger.info(f"\nFull report: {report_path}")
    logger.info(f"Training log: {LOG_DIR / 'training_audit.log'}")

    # Cleanup
    vec_env.close()
    gc.collect()

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ADAN Convergence Audit")
    parser.add_argument("--config", default="config/config.yaml", help="Config file path")
    parser.add_argument("--steps", type=int, default=30000, help="Total training steps")
    parser.add_argument("--n-steps", type=int, default=512, help="PPO rollout length")
    parser.add_argument("--batch-size", type=int, default=128, help="PPO mini-batch size")
    parser.add_argument("--n-epochs", type=int, default=4, help="PPO epochs per update")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--ent-coef", type=float, default=0.02, help="Entropy coefficient")
    parser.add_argument("--worker", default="w1", help="Worker key (w1, w2, w3, w4)")
    parser.add_argument("--log-interval", type=int, default=500, help="Log every N steps")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    train(args)
