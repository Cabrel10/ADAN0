"""ADAN Multi-Agent Training with Ray Tune Population-Based Training (PBT).

This script trains multiple PPO agents in parallel using Ray Tune's
PopulationBasedTraining scheduler.  Each trial (worker) instantiates the
real MultiAssetChunkedEnv wrapped in SubprocVecEnv + VecNormalize, with
the TemporalFusionExtractor (which now includes FiLM Meta-RL).

Business-logic components preserved from the original multiprocessing
implementation:
  - CapitalTierTracker
  - MetricsMonitor / UnifiedMetrics
  - PpoStdSafetyCallback
  - VecNormalize (checkpoint save/load)
  - All reward-shaping and risk callbacks

Usage (local 8-core, 16 GB):
    python scripts/train_parallel_agents.py \\
        --config config/config.yaml \\
        --steps 1000000 \\
        --num-cpus 8 \\
        --num-samples 4 \\
        --envs-per-worker 2 \\
        --steps-per-iter 10000
"""

import argparse
import copy
import json
import logging
import os
import signal
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

# Ray Tune imports
import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining

# ADAN imports
from adan_trading_bot.common.config_loader import ConfigLoader

try:
    from adan_trading_bot.common.custom_logger import setup_logging
except ImportError:
    setup_logging = None

from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader
from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv

try:
    from adan_trading_bot.utils.ppo_safety import PpoStdSafetyCallback
except ImportError:
    PpoStdSafetyCallback = None

try:
    from adan_trading_bot.utils.seed_manager import SeedManager
except ImportError:
    SeedManager = None

# Feature extractor (SOTA architecture: FiLM Meta-RL, TemporalFusion)
try:
    from adan_trading_bot.agent.feature_extractors import ContextualTemporalFusionExtractor
except ImportError:
    ContextualTemporalFusionExtractor = None

# Optional imports
try:
    from adan_trading_bot.common.central_logger import logger as central_logger
    from adan_trading_bot.performance.unified_metrics import UnifiedMetrics
    from adan_trading_bot.performance.unified_metrics_db import UnifiedMetricsDB
    UNIFIED_SYSTEM_AVAILABLE = True
except ImportError:
    UNIFIED_SYSTEM_AVAILABLE = False
    central_logger = None
    UnifiedMetrics = None
    UnifiedMetricsDB = None

try:
    import plotly.graph_objects as go
except ImportError:
    go = None


# ---------------------------------------------------------------------------
# Project root resolution
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent  # bot/


logger = logging.getLogger(__name__)


# ===========================================================================
# Business-logic helpers (preserved from original)
# ===========================================================================

def linear_schedule(start_val, end_val, progress):
    return start_val + (end_val - start_val) * progress


def get_adaptive_risk(step: int, total_steps: int,
                      start_cfg: dict, target_cfg: dict,
                      current_drawdown: float = 0.0) -> dict:
    progress = min(1.0, max(0.0, step / max(1, total_steps)))
    pos_size = linear_schedule(start_cfg['position_size_pct'], target_cfg['position_size_pct'], progress)
    sl = linear_schedule(start_cfg['stop_loss_pct'], target_cfg['stop_loss_pct'], progress)
    tp = linear_schedule(start_cfg['take_profit_pct'], target_cfg['take_profit_pct'], progress)

    if current_drawdown >= 0.25:
        safety_mult = 0.4
    elif current_drawdown >= 0.15:
        safety_mult = 0.65
    else:
        safety_mult = 1.0

    return {
        'max_position_size_pct': pos_size * safety_mult,
        'stop_loss_pct': max(0.01, sl),
        'take_profit_pct': tp,
    }


class AdaptiveRiskCallback(BaseCallback):
    def __init__(self, total_fine_tune_steps: int, start_cfg: dict, target_cfg: dict, verbose=0):
        super().__init__(verbose)
        self.total_fine_tune_steps = total_fine_tune_steps
        self.start_cfg = start_cfg
        self.target_cfg = target_cfg
        self.risk_log_samples = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])
        if infos is None:
            infos = [{}]
        for i in range(len(infos)):
            info = infos[i] if i < len(infos) else {}
            current_drawdown = info.get("portfolio", {}).get("max_dd", 0.0)

            risk_params = get_adaptive_risk(
                step=self.num_timesteps,
                total_steps=self.total_fine_tune_steps,
                start_cfg=self.start_cfg,
                target_cfg=self.target_cfg,
                current_drawdown=current_drawdown,
            )
            if hasattr(self.training_env, 'env_method'):
                self.training_env.env_method('set_global_risk', indices=[i], **risk_params)

            if self.num_timesteps % 1000 == 0 and len(self.risk_log_samples) < 10:
                self.risk_log_samples.append(f"Step {self.num_timesteps}: {risk_params}")
        return True


class TimeoutHandler:
    def __init__(self, seconds, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)


# ===========================================================================
# CapitalTierTracker (preserved)
# ===========================================================================

class CapitalTierTracker:
    """Tracks capital tier progression for each worker."""

    TIERS = {
        "Micro": {"min": 0, "max": 100},
        "Small": {"min": 100, "max": 1000},
        "Medium": {"min": 1000, "max": 10000},
        "High": {"min": 10000, "max": 100000},
        "Enterprise": {"min": 100000, "max": float("inf")},
    }

    def __init__(self, initial_balance=20):
        self.initial_balance = initial_balance
        self.current_tier = "Micro"
        self.tier_history = [("Micro", 0, initial_balance)]
        self.progression_log = []

    def get_tier_from_balance(self, balance):
        for tier_name, limits in self.TIERS.items():
            if limits["min"] <= balance < limits["max"]:
                return tier_name
        return "Enterprise"

    def update(self, step, balance, pnl=0.0):
        new_tier = self.get_tier_from_balance(balance)
        if new_tier != self.current_tier:
            self.tier_history.append((new_tier, step, balance))
            self.progression_log.append({
                "step": step,
                "from_tier": self.current_tier,
                "to_tier": new_tier,
                "balance": balance,
                "pnl": pnl,
                "timestamp": datetime.now().isoformat(),
            })
            self.current_tier = new_tier

    def get_progression_summary(self):
        return {
            "current_tier": self.current_tier,
            "tier_history": self.tier_history,
            "total_progressions": len(self.progression_log),
            "progression_log": self.progression_log,
            "reached_enterprise": self.current_tier == "Enterprise",
        }


# ===========================================================================
# MetricsMonitor callback (preserved)
# ===========================================================================

class MetricsMonitor(BaseCallback):
    """Enhanced callback to monitor each worker's performance and capital tier progression."""

    def __init__(self, config, num_workers=4, log_interval=1000):
        super().__init__()
        self.config = config
        self.num_workers = num_workers
        self.log_interval = log_interval
        self.worker_metrics: Dict[int, Dict] = {}
        self.portfolio_curves: Dict[int, list] = {i: [] for i in range(num_workers)}
        self.tier_trackers = {
            i: CapitalTierTracker(config.get("portfolio", {}).get("initial_balance", 20))
            for i in range(num_workers)
        }
        self.step_count = 0
        self.start_time = time.time()

        for i in range(num_workers):
            self.worker_metrics[i] = {
                "total_steps": 0,
                "total_rewards": [],
                "portfolio_values": [],
                "realized_pnls": [],
                "sharpe_ratios": [],
                "drawdowns": [],
                "trade_counts": [],
                "win_rates": [],
            }

    def _on_step(self) -> bool:
        self.step_count += 1
        if self.step_count % self.log_interval == 0:
            self._collect_worker_metrics()
        return True

    def _collect_worker_metrics(self):
        try:
            portfolio_managers = self.training_env.get_attr("portfolio_manager")
            for worker_id, pm in enumerate(portfolio_managers):
                if pm is None:
                    continue
                metrics = pm.metrics.get_metrics_summary()
                try:
                    current_balance = float(pm.get_portfolio_value())
                except Exception:
                    current_balance = float(self.config.get("portfolio", {}).get("initial_balance", 20))
                current_pnl = metrics.get("total_return", 0.0) * current_balance / 100.0

                self.tier_trackers[worker_id].update(self.step_count, current_balance, current_pnl)

                self.worker_metrics[worker_id]["total_steps"] = self.step_count
                self.worker_metrics[worker_id]["portfolio_values"].append(current_balance)
                self.worker_metrics[worker_id]["realized_pnls"].append(current_pnl)
                self.worker_metrics[worker_id]["sharpe_ratios"].append(metrics.get("sharpe_ratio", 0.0))
                self.worker_metrics[worker_id]["drawdowns"].append(metrics.get("max_drawdown", 0.0))
                self.worker_metrics[worker_id]["trade_counts"].append(metrics.get("total_trades", 0))
                self.worker_metrics[worker_id]["win_rates"].append(metrics.get("win_rate", 0.0))

                if worker_id == 0 or self.step_count % (self.log_interval * 5) == 0:
                    self.logger.record(f"worker_{worker_id}/balance", current_balance)
                    self.logger.record(f"worker_{worker_id}/pnl", current_pnl)
                    self.logger.record(f"worker_{worker_id}/tier", self.tier_trackers[worker_id].current_tier)
                    self.logger.record(f"worker_{worker_id}/sharpe", metrics.get("sharpe_ratio", 0.0))
        except Exception as e:
            logging.getLogger(__name__).error(f"Error collecting worker metrics: {e}", exc_info=True)

    def get_final_summary(self):
        summary = {
            "training_duration_minutes": (time.time() - self.start_time) / 60,
            "total_steps": self.step_count,
            "workers": {},
        }
        for worker_id in range(self.num_workers):
            if not self.worker_metrics[worker_id]["portfolio_values"]:
                continue
            final_balance = self.worker_metrics[worker_id]["portfolio_values"][-1]
            initial = self.config.get("portfolio", {}).get("initial_balance", 20)
            tier_summary = self.tier_trackers[worker_id].get_progression_summary()
            summary["workers"][f"w{worker_id + 1}"] = {
                "initial_balance": initial,
                "final_balance": final_balance,
                "total_return_pct": ((final_balance - initial) / max(initial, 1e-8)) * 100,
                "final_sharpe": self.worker_metrics[worker_id]["sharpe_ratios"][-1] if self.worker_metrics[worker_id]["sharpe_ratios"] else 0.0,
                "max_drawdown": max(self.worker_metrics[worker_id]["drawdowns"]) if self.worker_metrics[worker_id]["drawdowns"] else 0.0,
                "total_trades": self.worker_metrics[worker_id]["trade_counts"][-1] if self.worker_metrics[worker_id]["trade_counts"] else 0,
                "tier_progression": tier_summary,
                "reached_enterprise": tier_summary["reached_enterprise"],
            }
        return summary


# ===========================================================================
# OMEGA Worker Profiles
# ===========================================================================

WORKER_PROFILES: Dict[str, Dict[str, Any]] = {
    "scalper": {
        "name": "Scalper",
        "specialization": {"timeframe": "5m"},
        "n_steps": 512,
        "batch_size": 64,
    },
    "intraday": {
        "name": "Intraday",
        "specialization": {"timeframe": "1h"},
        "n_steps": 512,
        "batch_size": 64,
    },
    "swing": {
        "name": "Swing",
        "specialization": {"timeframe": "4h"},
        "n_steps": 512,
        "batch_size": 64,
    },
    "position": {
        "name": "Position",
        "specialization": {"timeframe": "4h"},
        "n_steps": 1024,
        "batch_size": 128,
    },
}


def _inject_worker_profile(worker_config: dict, profile_name: str) -> dict:
    """Merge a profile's defaults into *worker_config* (YAML values take precedence)."""
    profile = WORKER_PROFILES.get(profile_name, {})
    if not profile:
        return worker_config
    merged = copy.deepcopy(profile)
    # Deep-merge: worker_config wins
    for k, v in worker_config.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k].update(v)
        else:
            merged[k] = v
    return merged


# ===========================================================================
# Environment factory
# ===========================================================================

def make_env(
    config: dict,
    worker_idx: int = 0,
    envs_per_worker: int = 1,
    use_subproc: bool = True,
    preloaded_data: Optional[Dict] = None,
    profile: Optional[str] = None,
):
    """Create a vectorised environment wrapped with VecNormalize.

    Bug-fix (v3): Adds *profile* injection (scalper, intraday, swing, position).
    Each sub-env receives the correct ``worker_config``
    (w1/w2/w3/w4) **and** pre-loaded parquet data.
    """
    worker_key = f"w{worker_idx + 1}"
    worker_config = copy.deepcopy(config.get("workers", {}).get(worker_key, {}))
    if profile:
        worker_config = _inject_worker_profile(worker_config, profile)

    def _make_single(env_idx: int):
        def _init():
            wc = copy.deepcopy(worker_config)
            wc["worker_id"] = env_idx
            if profile:
                wc.setdefault("profile", profile)
            return MultiAssetChunkedEnv(
                data=preloaded_data,
                config=config,
                worker_config=wc,
                worker_id=env_idx,
                live_mode=False,
            )
        return _init

    env_fns = [_make_single(worker_idx * envs_per_worker + j) for j in range(envs_per_worker)]

    if use_subproc and envs_per_worker > 1:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)

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


# ===========================================================================
# Ray Tune Trainable – ADAN_PBT_Worker
# ===========================================================================

class ADAN_PBT_Worker(tune.Trainable):
    """Ray Tune Trainable that wraps a single PPO worker.

    Each trial manages:
      * A vectorised environment (SubprocVecEnv + VecNormalize).
      * A PPO model with the real TemporalFusionExtractor.
      * Callbacks: MetricsMonitor, PpoStdSafetyCallback.
      * Checkpoint saving (model.zip + vecnormalize.pkl).
    """

    def setup(self, config: Dict[str, Any]):
        """Initialise env + PPO model from Ray Tune config.

        Bug-fix (v2) – three corrections:
          1. Worker identity: read the correct w<N> section from adan_config.
          2. ChunkedDataLoader: load parquet data to prevent live websockets.
          3. No latency simulator: MultiAssetChunkedEnv used directly.
        """
        self.adan_config = config["adan_config"]
        self.worker_idx = config.get("worker_idx", 0)
        self.envs_per_worker = config.get("envs_per_worker", 2)
        self.use_subproc = config.get("use_subproc", True)
        self.interval_timesteps = config.get("interval_timesteps", 10_000)
        self._total_timesteps = 0
        self.profile = config.get("profile", None)  # scalper / swing / ...

        # Mutable hyper-parameters (PBT will perturb these)
        self.learning_rate = config.get("learning_rate", 3e-4)
        self.ent_coef = config.get("ent_coef", 0.01)
        self.gamma = config.get("gamma", 0.99)

        # 1. Restore worker identity
        worker_key = f"w{self.worker_idx + 1}"
        worker_config = copy.deepcopy(
            self.adan_config.get("workers", {}).get(worker_key, {})
        )
        worker_config["worker_id"] = self.worker_idx
        logger.info(
            f"Worker {self.worker_idx} ({worker_key}): "
            f"assets={worker_config.get('assets', '?')}, "
            f"data_split={worker_config.get('data_split', '?')}"
        )

        # 2. Pre-load parquet data
        preloaded_data = None
        try:
            loader = ChunkedDataLoader(
                config=self.adan_config,
                worker_config=worker_config,
                worker_id=self.worker_idx,
            )
            preloaded_data = loader.load_chunk(0)
            logger.info(
                f"Worker {self.worker_idx}: ChunkedDataLoader loaded chunk 0 "
                f"({type(preloaded_data).__name__})"
            )
        except Exception as exc:
            logger.warning(
                f"Worker {self.worker_idx}: ChunkedDataLoader failed ({exc}); "
                f"env will initialise its own loader."
            )

        # 3. Create env with profile
        self.vec_env = make_env(
            self.adan_config,
            worker_idx=self.worker_idx,
            envs_per_worker=self.envs_per_worker,
            use_subproc=self.use_subproc,
            preloaded_data=preloaded_data,
            profile=self.profile,
        )

        # Policy kwargs + ContextualTemporalFusionExtractor
        agent_cfg = self.adan_config.get("agent", {})
        fe_kwargs = agent_cfg.get("features_extractor_kwargs", {})
        policy_kwargs = copy.deepcopy(fe_kwargs.get("policy_kwargs", {}))

        activation_fn_map = {"ReLU": nn.ReLU, "Tanh": nn.Tanh, "LeakyReLU": nn.LeakyReLU}
        if "activation_fn" in policy_kwargs:
            act_name = str(policy_kwargs["activation_fn"]).split(".")[-1]
            policy_kwargs["activation_fn"] = activation_fn_map.get(act_name, nn.ReLU)

        # OMEGA-4A: only pass valid extractor kwargs
        if ContextualTemporalFusionExtractor is not None:
            policy_kwargs.setdefault(
                "features_extractor_class", ContextualTemporalFusionExtractor
            )
            valid_fe_keys = {"features_dim", "context_dim", "cnn_hidden", "dropout"}
            safe_fe_kwargs = {k: v for k, v in fe_kwargs.items() if k in valid_fe_keys}
            safe_fe_kwargs.setdefault("context_dim", 6)
            policy_kwargs.setdefault("features_extractor_kwargs", safe_fe_kwargs)

        # Seed
        seed = self.adan_config.get("general", {}).get("random_seed", 42) + self.worker_idx
        if SeedManager is not None:
            SeedManager.initialize(seed)

        # PPO model – profile may override n_steps/batch_size
        device = "cuda" if torch.cuda.is_available() else "cpu"
        prof_cfg = WORKER_PROFILES.get(self.profile, {}) if self.profile else {}
        n_steps = prof_cfg.get("n_steps", agent_cfg.get("n_steps", 2048))
        batch_size = prof_cfg.get("batch_size", agent_cfg.get("batch_size", 64))
        # Ensure batch_size divides n_steps * envs_per_worker
        total_rollout = n_steps * self.envs_per_worker
        if total_rollout % batch_size != 0:
            batch_size = max(1, total_rollout // max(1, total_rollout // batch_size))

        self.model = PPO(
            "MultiInputPolicy",
            self.vec_env,
            device=device,
            learning_rate=self.learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=agent_cfg.get("n_epochs", 10),
            gamma=self.gamma,
            gae_lambda=agent_cfg.get("gae_lambda", 0.95),
            clip_range=agent_cfg.get("clip_range", 0.2),
            ent_coef=self.ent_coef,
            vf_coef=agent_cfg.get("vf_coef", 0.5),
            max_grad_norm=agent_cfg.get("max_grad_norm", 0.5),
            policy_kwargs=policy_kwargs if policy_kwargs else None,
            verbose=0,
            seed=seed,
        )

        # Callbacks
        self._callbacks = []
        metrics_monitor = MetricsMonitor(
            config=self.adan_config,
            num_workers=self.envs_per_worker,
            log_interval=max(500, self.interval_timesteps // 10),
        )
        self._callbacks.append(metrics_monitor)

        if PpoStdSafetyCallback is not None:
            ppo_safety = PpoStdSafetyCallback(
                min_log_std=-5.0,
                max_log_std=2.0,
                std_warn_threshold=100.0,
                verbose=0,
            )
            self._callbacks.append(ppo_safety)

        self._metrics_monitor = metrics_monitor

    def step(self):
        """Run one training iteration (interval_timesteps steps of PPO.learn)."""
        # Apply mutable hyperparameters
        self.model.learning_rate = self.learning_rate
        self.model.ent_coef = self.ent_coef
        self.model.gamma = self.gamma

        self.model.learn(
            total_timesteps=self.interval_timesteps,
            callback=self._callbacks,
            reset_num_timesteps=False,
        )
        self._total_timesteps += self.interval_timesteps

        # Collect metrics
        mean_reward = 0.0
        mean_sharpe = 0.0
        mean_balance = 0.0
        try:
            ep_rewards = self.model.ep_info_buffer
            if ep_rewards and len(ep_rewards) > 0:
                mean_reward = float(np.mean([ep["r"] for ep in ep_rewards]))
        except Exception:
            pass

        try:
            wm = self._metrics_monitor.worker_metrics.get(0, {})
            if wm.get("sharpe_ratios"):
                mean_sharpe = wm["sharpe_ratios"][-1]
            if wm.get("portfolio_values"):
                mean_balance = wm["portfolio_values"][-1]
        except Exception:
            pass

        return {
            "mean_reward": mean_reward,
            "mean_sharpe": mean_sharpe,
            "mean_balance": mean_balance,
            "learning_rate": self.learning_rate,
            "ent_coef": self.ent_coef,
            "gamma": self.gamma,
            "timesteps_total": self._total_timesteps,
        }

    def save_checkpoint(self, checkpoint_dir: str) -> str:
        """Save PPO model + VecNormalize stats."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_path = os.path.join(checkpoint_dir, "model.zip")
        self.model.save(model_path)

        vec_path = os.path.join(checkpoint_dir, "vecnormalize.pkl")
        self.vec_env.save(vec_path)

        state = {
            "total_timesteps": self._total_timesteps,
            "learning_rate": self.learning_rate,
            "ent_coef": self.ent_coef,
            "gamma": self.gamma,
        }
        with open(os.path.join(checkpoint_dir, "worker_state.json"), "w") as f:
            json.dump(state, f)

        return checkpoint_dir

    def load_checkpoint(self, checkpoint_dir: str):
        """Restore PPO model + VecNormalize stats."""
        model_path = os.path.join(checkpoint_dir, "model.zip")
        if os.path.exists(model_path):
            self.model = PPO.load(model_path, env=self.vec_env)

        vec_path = os.path.join(checkpoint_dir, "vecnormalize.pkl")
        if os.path.exists(vec_path):
            # Load onto the underlying venv (not the VecNormalize wrapper)
            venv = self.vec_env.venv if hasattr(self.vec_env, "venv") else self.vec_env
            self.vec_env = VecNormalize.load(vec_path, venv)
            self.model.set_env(self.vec_env)

        state_path = os.path.join(checkpoint_dir, "worker_state.json")
        if os.path.exists(state_path):
            with open(state_path) as f:
                state = json.load(f)
            self._total_timesteps = state.get("total_timesteps", 0)
            self.learning_rate = state.get("learning_rate", self.learning_rate)
            self.ent_coef = state.get("ent_coef", self.ent_coef)
            self.gamma = state.get("gamma", self.gamma)

    def cleanup(self):
        """Close environments."""
        try:
            if hasattr(self, "vec_env") and self.vec_env is not None:
                self.vec_env.close()
        except Exception:
            pass


# ===========================================================================
# PBT setup and launch
# ===========================================================================

def run_pbt(
    config: dict,
    num_cpus: int = 8,
    num_samples: int = 4,
    envs_per_worker: int = 2,
    use_subproc: bool = True,
    total_steps: int = 1_000_000,
    interval_timesteps: int = 10_000,
    stop_config: Optional[dict] = None,
    storage_path: Optional[str] = None,
    profiles: Optional[list] = None,
):
    """Launch Ray Tune with Population-Based Training.

    Args:
        config: Full ADAN config dict (from ConfigLoader).
        num_cpus: CPUs available to Ray.
        num_samples: Number of concurrent PBT trials.
        envs_per_worker: Sub-envs per trial (SubprocVecEnv).
        use_subproc: Whether to use SubprocVecEnv.
        total_steps: Total training timesteps per trial.
        interval_timesteps: Timesteps per PBT iteration.
        stop_config: Optional tune stop dict.
        storage_path: Where Ray stores results.
        profiles: Optional list of profile names (e.g. ['scalper', 'swing']).
    """
    if storage_path is None:
        storage_path = str(PROJECT_ROOT / "logs" / "ray_results")

    max_iterations = max(1, total_steps // interval_timesteps)

    # PBT scheduler
    pbt_scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=2,
        metric="mean_reward",
        mode="max",
        hyperparam_mutations={
            "learning_rate": tune.loguniform(1e-6, 1e-3),
            "ent_coef": tune.uniform(0.0, 0.1),
            "gamma": tune.uniform(0.9, 0.999),
        },
    )

    # Build per-trial param space
    # If profiles supplied, cycle through them for each trial
    _profiles = profiles or []
    trial_profiles = [
        _profiles[i % len(_profiles)] if _profiles else None
        for i in range(num_samples)
    ]
    param_space = {
        "adan_config": config,
        "worker_idx": tune.grid_search(list(range(num_samples))),
        "envs_per_worker": envs_per_worker,
        "use_subproc": use_subproc,
        "interval_timesteps": interval_timesteps,
        "learning_rate": tune.loguniform(1e-5, 1e-3),
        "ent_coef": tune.uniform(0.0, 0.05),
        "gamma": tune.uniform(0.95, 0.999),
    }
    # Only add profile to param_space if profiles are provided
    if _profiles:
        param_space["profile"] = tune.grid_search(trial_profiles)

    # Stop criteria
    if stop_config is None:
        stop_config = {"training_iteration": max_iterations}

    # Tuner
    tuner = tune.Tuner(
        ADAN_PBT_Worker,
        tune_config=tune.TuneConfig(
            scheduler=pbt_scheduler,
            num_samples=num_samples,
            max_concurrent_trials=num_samples,
            reuse_actors=False,
        ),
        run_config=ray.train.RunConfig(
            name="adan_pbt_training",
            storage_path=storage_path,
            stop=stop_config,
            verbose=1,
        ),
        param_space=param_space,
    )

    results = tuner.fit()

    # Summary
    summary = {
        "num_trials": len(results),
        "completed": True,
        "best_trial": {},
        "timestamp": datetime.now().isoformat(),
    }

    try:
        best_result = results.get_best_result(metric="mean_reward", mode="max")
        if best_result and best_result.metrics:
            summary["best_trial"] = {
                "mean_reward": best_result.metrics.get("mean_reward", 0.0),
                "mean_sharpe": best_result.metrics.get("mean_sharpe", 0.0),
                "mean_balance": best_result.metrics.get("mean_balance", 0.0),
                "learning_rate": best_result.metrics.get("learning_rate", 0.0),
                "ent_coef": best_result.metrics.get("ent_coef", 0.0),
                "gamma": best_result.metrics.get("gamma", 0.0),
                "timesteps_total": best_result.metrics.get("timesteps_total", 0),
                "training_iteration": best_result.metrics.get("training_iteration", 0),
            }
    except Exception as e:
        logger.warning(f"Could not extract best result: {e}")

    # Write summary
    summary_path = os.path.join(storage_path, "pbt_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"PBT Summary: {json.dumps(summary, indent=2)}")
    return results, summary


# ===========================================================================
# Main entry point
# ===========================================================================

def main(
    config_path: str,
    resume: bool = False,
    num_cpus: int = 8,
    num_samples: int = 4,
    envs_per_worker: int = 2,
    use_subproc: bool = True,
    total_steps: int = 1_000_000,
    interval_timesteps: int = 10_000,
    log_level: str = "INFO",
    checkpoint_dir: Optional[str] = None,
    stop_config: Optional[dict] = None,
    profiles: Optional[list] = None,
):
    """Main entry: load config, init Ray, run PBT."""
    # Logging
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(level=numeric_level)
    logging.getLogger("adan_trading_bot").setLevel(numeric_level)

    # Load config
    config = ConfigLoader.load_config(config_path)
    logger.info(f"Loaded config from {config_path}")

    # Override steps
    if total_steps:
        config.setdefault("training", {})["timesteps_per_instance"] = total_steps

    # Storage path
    storage_path = checkpoint_dir or str(PROJECT_ROOT / "logs" / "ray_results")

    # Init Ray
    ray.init(
        num_cpus=num_cpus,
        include_dashboard=False,
        ignore_reinit_error=True,
        _system_config={
            "object_store_memory": 200 * 1024 * 1024,  # 200 MB
        },
    )

    logger.info("=" * 80)
    logger.info("🔥 ADAN PBT AutoRL Training")
    logger.info(f"   CPUs: {num_cpus}, Samples: {num_samples}, Envs/worker: {envs_per_worker}")
    logger.info(f"   Total steps: {total_steps:,}, Interval: {interval_timesteps:,}")
    logger.info(f"   SubprocVecEnv: {use_subproc}")
    logger.info("=" * 80)

    try:
        results, summary = run_pbt(
            config=config,
            num_cpus=num_cpus,
            num_samples=num_samples,
            envs_per_worker=envs_per_worker,
            use_subproc=use_subproc,
            total_steps=total_steps,
            interval_timesteps=interval_timesteps,
            stop_config=stop_config,
            storage_path=storage_path,
            profiles=profiles,
        )

        logger.info("=" * 80)
        logger.info("🎯 PBT TRAINING COMPLETE")
        logger.info("=" * 80)

        if summary.get("best_trial"):
            bt = summary["best_trial"]
            logger.info(f"Best trial: reward={bt.get('mean_reward', 0):.4f}, "
                        f"sharpe={bt.get('mean_sharpe', 0):.4f}, "
                        f"lr={bt.get('learning_rate', 0):.2e}, "
                        f"ent_coef={bt.get('ent_coef', 0):.4f}")

        print("COMPLETE")

    except Exception as e:
        logger.error(f"PBT training failed: {e}", exc_info=True)
        raise
    finally:
        ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ADAN Agents with Ray Tune PBT")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--num-cpus", type=int, default=8, help="Number of CPUs for Ray")
    parser.add_argument("--num-samples", type=int, default=4, help="Number of concurrent PBT trials")
    parser.add_argument("--envs-per-worker", type=int, default=2, help="Sub-envs per worker (SubprocVecEnv)")
    parser.add_argument("--use-subproc", action="store_true", default=True, help="Use SubprocVecEnv")
    parser.add_argument("--no-subproc", action="store_true", help="Use DummyVecEnv instead")
    parser.add_argument("--steps", type=int, default=1_000_000, help="Total training timesteps")
    parser.add_argument("--steps-per-iter", type=int, default=10_000, help="Timesteps per PBT iteration")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Override checkpoint dir")
    # Legacy args (ignored, kept for CLI compatibility)
    parser.add_argument("--num-envs", type=int, default=4, help="(legacy, ignored)")
    parser.add_argument("--progress-bar", action="store_true", help="(legacy, ignored)")
    parser.add_argument("--timeout", type=int, default=None, help="(legacy, ignored)")
    parser.add_argument("--fine-tune", action="store_true", help="(legacy, ignored)")
    parser.add_argument("--profiles", type=str, nargs="+", default=None,
                        help="Worker profiles, e.g. --profiles scalper swing")

    args = parser.parse_args()

    main(
        config_path=args.config,
        resume=args.resume,
        num_cpus=args.num_cpus,
        num_samples=args.num_samples,
        envs_per_worker=args.envs_per_worker,
        use_subproc=not args.no_subproc,
        total_steps=args.steps,
        interval_timesteps=args.steps_per_iter,
        log_level=args.log_level,
        checkpoint_dir=args.checkpoint_dir,
        profiles=args.profiles,
    )
