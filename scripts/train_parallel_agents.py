"""ADAN Multi-Agent Training with Ray Tune Population-Based Training (PBT).

This script trains multiple PPO agents in parallel using Ray Tune's
PopulationBasedTraining scheduler.  Each trial (worker) instantiates the
real MultiAssetChunkedEnv wrapped in DummyVecEnv + VecNormalize (SubprocVecEnv
disabled by default to avoid Ray/fork conflicts), with the
ContextualTemporalFusionExtractor (which now includes FiLM Meta-RL).

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
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure src/ is in the Python path for adan_trading_bot imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# ── GARDE-FOU ANTI-DEADLOCK (2026-06-27) ──────────────────────────────────
# Le run fa_500k_v4 a gelé à step 12417 (= fin du 6e rollout, 6*2048=12288)
# pendant l'update PPO (n_epochs=20 + CNN+Attention) : thread contention
# OpenMP/MKL sur un VPS 4 cœurs -> deadlock silencieux de PyTorch.
# On borne les threads AVANT d'importer torch/numpy (lus à l'import).
# Surchargeable via ADAN_NUM_THREADS (défaut: nproc-1, min 1).
try:
    _ncpu = os.cpu_count() or 2
    _nthreads = int(os.environ.get("ADAN_NUM_THREADS", max(1, _ncpu - 1)))
except Exception:
    _nthreads = 1
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, str(_nthreads))
# Évite l'oversubscription / les blocages de pool OpenMP imbriqués.
os.environ.setdefault("OMP_DYNAMIC", "FALSE")
os.environ.setdefault("KMP_BLOCKTIME", "0")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Borne aussi le thread-pool interne de PyTorch (intra-op) : la vraie cause
# du deadlock pendant backward(). 1 thread inter-op pour éviter le contention.
try:
    torch.set_num_threads(_nthreads)
    torch.set_num_interop_threads(1)
except Exception:
    pass

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.utils import FloatSchedule, update_learning_rate

# SOTA 2026: WorldModelPPO with auxiliary forward-prediction loss
try:
    from adan_trading_bot.agent.feature_extractors import WorldModelPPO
except ImportError:
    WorldModelPPO = None

# Ray Tune imports (optional — only needed for HPO mode, not sandbox)
try:
    import ray
    from ray import tune
    from ray.tune.schedulers import PopulationBasedTraining
    RAY_AVAILABLE = True
except ImportError:
    ray = None
    tune = None
    PopulationBasedTraining = None
    RAY_AVAILABLE = False

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

# ActionSaturationGuard: anti-collapse par tete (V31-500k FIX).
# Le forensic V31-DIAG (forensics/v31_diag_collapse_20260817/) a demontre un
# boundary-lock tanh: 62-72% des actions a |a0|>=0.999, std=0.136, KL 0.58-0.96.
# Ce callback RELEVE le plancher log_std des tetes saturees (PpoStdSafetyCallback
# ne clampe que la borne HAUTE). Active via ADAN_SATGUARD (defaut ON pour V31).
try:
    from adan_trading_bot.utils.action_saturation_guard import ActionSaturationGuard
except ImportError:
    ActionSaturationGuard = None

# ActionDimMonitor: instrumentation par tête (MESURE SEULE — ne modifie rien).
# Active via ADAN_ACTIONDIM=1 (par défaut OFF pour ne pas alourdir les runs
# de production). Le run diagnostique V2 l'active pour suivre μ(size)/σ(size).
try:
    from adan_trading_bot.utils.action_dim_monitor import ActionDimMonitor
except ImportError:
    ActionDimMonitor = None

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

# Training output directory — use external mount if available, fallback to project logs
_EXTERNAL_TRAIN_DIR = Path(os.environ.get("ADAN_TRAIN_DIR", "./logs/training")).resolve()
TRAIN_OUTPUT_DIR = _EXTERNAL_TRAIN_DIR if _EXTERNAL_TRAIN_DIR.exists() else (PROJECT_ROOT / "logs").resolve()


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
                "realized_pnl_steps": [],
                "realized_pnl_episodes": [],
                "realized_pnl_episode_currents": [],
                "realized_pnl_cumulatives": [],
                "cash_values": [],
                "equity_values": [],
                "realized_equity_values": [],
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
            # FIX: Use env_method to call get_metrics() instead of get_attr("portfolio_manager")
            # get_attr tries to pickle the entire portfolio_manager object (which has module references)
            # env_method just calls a method and returns the result (much safer for pickle)
            metrics_list = self.training_env.env_method("get_portfolio_metrics_dict")
            
            for worker_id, pm_metrics in enumerate(metrics_list):
                if pm_metrics is None:
                    pm_metrics = {}
                
                # SESSION 15: Type check — pm_metrics can be list or dict
                if isinstance(pm_metrics, list):
                    pm_metrics = pm_metrics[0] if pm_metrics else {}
                
                # Ensure pm_metrics is always a dict at this point
                if not isinstance(pm_metrics, dict):
                    pm_metrics = {}
                
                try:
                    total_value = float(pm_metrics.get("total_value", 0.0))
                    realized_pnl = float(pm_metrics.get("total_realized_pnl", 0.0))
                    realized_pnl_step = float(pm_metrics.get("realized_pnl_step", 0.0))
                    realized_pnl_episode = float(pm_metrics.get("realized_pnl_episode", 0.0))
                    realized_pnl_episode_current = float(
                        pm_metrics.get("realized_pnl_episode_current", 0.0)
                    )
                    realized_pnl_cumulative = float(
                        pm_metrics.get("realized_pnl_cumulative", 0.0)
                    )
                    cash = float(pm_metrics.get("cash", 0.0))
                    equity = float(pm_metrics.get("equity", total_value))
                    realized_equity = float(pm_metrics.get("realized_equity", 0.0))
                    initial = float(pm_metrics.get("initial_capital", 20.5))
                    
                    # current_balance = total equity (Cash + Unrealized positions)
                    current_balance = total_value
                    
                    # Sanity: balance should never be negative
                    if current_balance <= 0:
                        current_balance = initial * 0.1  # bankrupt floor
                    elif current_balance > initial * 10:
                        current_balance = initial  # spike detected, reset
                        
                    # For info only: unrealized
                    unrealized = float(pm_metrics.get("unrealized_pnl_total", 0.0))
                    open_count = int(pm_metrics.get("open_positions_count", 0))
                except Exception:
                    current_balance = float(self.config.get("portfolio", {}).get("initial_balance", 20.5))
                    realized_pnl = 0.0
                    realized_pnl_step = 0.0
                    realized_pnl_episode = 0.0
                    realized_pnl_episode_current = 0.0
                    realized_pnl_cumulative = 0.0
                    cash = current_balance
                    equity = current_balance
                    realized_equity = current_balance
                    unrealized = 0.0
                    open_count = 0
                    initial = current_balance

                self.tier_trackers[worker_id].update(self.step_count, current_balance, realized_pnl)

                self.worker_metrics[worker_id]["total_steps"] = self.step_count
                self.worker_metrics[worker_id]["portfolio_values"].append(current_balance)
                self.worker_metrics[worker_id]["realized_pnls"].append(realized_pnl)
                self.worker_metrics[worker_id]["realized_pnl_steps"].append(realized_pnl_step)
                self.worker_metrics[worker_id]["realized_pnl_episodes"].append(realized_pnl_episode)
                self.worker_metrics[worker_id]["realized_pnl_episode_currents"].append(
                    realized_pnl_episode_current
                )
                self.worker_metrics[worker_id]["realized_pnl_cumulatives"].append(
                    realized_pnl_cumulative
                )
                self.worker_metrics[worker_id]["cash_values"].append(cash)
                self.worker_metrics[worker_id]["equity_values"].append(equity)
                self.worker_metrics[worker_id]["realized_equity_values"].append(
                    realized_equity
                )
                # Defensive: ensure pm_metrics is still a dict before calling .get()
                if not isinstance(pm_metrics, dict):
                    pm_metrics = {}
                self.worker_metrics[worker_id]["sharpe_ratios"].append(pm_metrics.get("sharpe_ratio", 0.0))
                self.worker_metrics[worker_id]["drawdowns"].append(pm_metrics.get("max_drawdown", 0.0))
                self.worker_metrics[worker_id]["trade_counts"].append(pm_metrics.get("total_trades", 0))
                self.worker_metrics[worker_id]["win_rates"].append(pm_metrics.get("win_rate", 0.0))

                if worker_id == 0 or self.step_count % (self.log_interval * 5) == 0:
                    self.logger.record(f"worker_{worker_id}/cash_balance", current_balance)
                    self.logger.record(f"worker_{worker_id}/realized_pnl", realized_pnl)
                    self.logger.record(f"worker_{worker_id}/realized_pnl_step", realized_pnl_step)
                    self.logger.record(f"worker_{worker_id}/realized_pnl_episode", realized_pnl_episode)
                    self.logger.record(f"worker_{worker_id}/realized_pnl_cumulative", realized_pnl_cumulative)
                    self.logger.record(f"worker_{worker_id}/cash", cash)
                    self.logger.record(f"worker_{worker_id}/equity", equity)
                    self.logger.record(f"worker_{worker_id}/realized_equity", realized_equity)
                    self.logger.record(f"worker_{worker_id}/unrealized_info", unrealized)
                    self.logger.record(f"worker_{worker_id}/open_positions", open_count)
                    self.logger.record(f"worker_{worker_id}/tier", self.tier_trackers[worker_id].current_tier)
                    # Final defensive check before logging
                    if isinstance(pm_metrics, dict):
                        self.logger.record(f"worker_{worker_id}/sharpe", pm_metrics.get("sharpe_ratio", 0.0))
                    else:
                        self.logger.record(f"worker_{worker_id}/sharpe", 0.0)

                    # ── REWARD COMPONENTS COLLECTOR ──────────────────────────
                    # Read _last_reward_components from the env (not pm)
                    try:
                        envs = self.training_env.get_attr("_last_reward_components")
                        rc = envs[worker_id] if envs and worker_id < len(envs) else None
                        if rc and isinstance(rc, dict):
                            # Core PnL
                            self.logger.record(f"reward_{worker_id}/pnl_net",       float(rc.get("pnl_net", 0.0)))
                            self.logger.record(f"reward_{worker_id}/commission",     float(rc.get("commission", 0.0)))
                            self.logger.record(f"reward_{worker_id}/trade_cost",     float(rc.get("trade_cost", 0.0)))
                            # Risk
                            self.logger.record(f"reward_{worker_id}/drawdown_pen",   float(rc.get("drawdown_penalty", 0.0)))
                            self.logger.record(f"reward_{worker_id}/drawdown_pct",   float(rc.get("drawdown_pct", 0.0)))
                            # Inaction / survival
                            self.logger.record(f"reward_{worker_id}/inaction",       float(rc.get("inaction", 0.0)))
                            self.logger.record(f"reward_{worker_id}/survival",       float(rc.get("survival_bonus", 0.0)))
                            # Outcome
                            self.logger.record(f"reward_{worker_id}/outcome_tp",     float(rc.get("outcome_tp", 0.0)))
                            self.logger.record(f"reward_{worker_id}/outcome_sl",     float(rc.get("outcome_sl", 0.0)))
                            self.logger.record(f"reward_{worker_id}/outcome_passiv", float(rc.get("outcome_passivity", 0.0)))
                            # Frequency
                            self.logger.record(f"reward_{worker_id}/freq_5m",        float(rc.get("frequency_5m", 0.0)))
                            self.logger.record(f"reward_{worker_id}/freq_1h",        float(rc.get("frequency_1h", 0.0)))
                            self.logger.record(f"reward_{worker_id}/freq_4h",        float(rc.get("frequency_4h", 0.0)))
                            self.logger.record(f"reward_{worker_id}/freq_daily",     float(rc.get("frequency_daily", 0.0)))
                            # Position / duration / capacity
                            self.logger.record(f"reward_{worker_id}/pos_limit_pen",  float(rc.get("pos_limit_penalty", 0.0)))
                            self.logger.record(f"reward_{worker_id}/duration_pen",   float(rc.get("duration_penalty", 0.0)))
                            self.logger.record(f"reward_{worker_id}/capacity_rew",   float(rc.get("capacity_reward", 0.0)))
                            self.logger.record(f"reward_{worker_id}/inaction_pen",   float(rc.get("inaction_penalty", 0.0)))
                            # Excellence (Gugu-March)
                            self.logger.record(f"reward_{worker_id}/exc_sharpe",     float(rc.get("excellence_sharpe", 0.0)))
                            self.logger.record(f"reward_{worker_id}/exc_streak",     float(rc.get("excellence_streak", 0.0)))
                            self.logger.record(f"reward_{worker_id}/exc_confluence", float(rc.get("excellence_confluence", 0.0)))
                            self.logger.record(f"reward_{worker_id}/exc_chunk",      float(rc.get("excellence_chunk", 0.0)))
                            # Tier events
                            self.logger.record(f"reward_{worker_id}/promotion",      float(rc.get("promotion_bonus", 0.0)))
                            self.logger.record(f"reward_{worker_id}/demotion",       float(rc.get("demotion_penalty", 0.0)))
                            # Composition
                            self.logger.record(f"reward_{worker_id}/raw",            float(rc.get("raw", 0.0)))
                            self.logger.record(f"reward_{worker_id}/final",          float(rc.get("final_reward", 0.0)))
                            # State context
                            self.logger.record(f"state_{worker_id}/regime",          0.0)  # string — skip scalar
                            self.logger.record(f"state_{worker_id}/open_positions",  float(rc.get("open_positions", 0)))
                            self.logger.record(f"state_{worker_id}/steps_no_trade",  float(rc.get("steps_since_trade", 0)))
                            self.logger.record(f"state_{worker_id}/daily_trades",    float(rc.get("daily_trades", 0)))
                            self.logger.record(f"state_{worker_id}/invalid_attempts",float(rc.get("invalid_attempts", 0)))
                            self.logger.record(f"state_{worker_id}/trade_attempts",  float(rc.get("trade_attempts", 0)))
                    except Exception:
                        pass
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
# DIAGNOSTIC-V3 (2026-06-29) — Entropy-collapse instrumentation
# ===========================================================================
# Measure-only callback. Reads, never writes, the env/policy state. Logs the
# four numbers the decision tree needs, every `log_every` steps, to a CSV:
#   - action0 histogram + mean/std (collapse signature = bimodal at +-1)
#   - HOLD/BUY/SELL share of REQUESTED discrete actions
#   - steps_flat / steps_open share (collapse = 99.7% open)
#   - illegal_ratio (rejected actions / steps)
#   - policy entropy (mean of policy distribution entropy on the rollout batch)
# Activated by env ADAN_DIAG_COLLAPSE=1 so it never perturbs normal CI runs.
# It is purely additive: failures are swallowed so training can never break.
class DiagnosticCollapseCallback(BaseCallback):
    """Per-window collapse telemetry (action0 histo, HOLD%, flat/open, illegal,
    entropy). Measure-only — does NOT touch reward, gradient or env state."""

    def __init__(self, csv_path: str, log_every: int = 10000, verbose: int = 1):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.log_every = max(500, int(log_every))
        self._reset_window()
        self._prev_rej_total = None
        self._header_written = False
        self._next_flush = self.log_every  # first flush at exactly log_every steps
        # Returns/advantages are finalized only after collect_rollouts ends.
        # Defer CSV writes so critic telemetry never reads zero-filled tensors.
        self._flush_pending = False
        # DIAGNOSTIC-V8 circuit-breaker state. Auto-stop training the moment a
        # policy collapse is detected (proven pattern from v8 500k run: pct_buy
        # -> 1.0 and a0_mean -> +inf at ~124k-128k). Stops wasting compute on a
        # dead policy and preserves the last healthy checkpoint.
        self._collapse_tripped = False
        # need N consecutive collapsed windows to avoid a false positive on a
        # transient spike (2 windows = 2*log_every steps of sustained collapse).
        self._collapse_streak = 0
        # V13: require more consecutive windows (env-configurable, default 4 instead
        # of 2) so a transient degenerate window never trips the (opt-in) breaker.
        self._collapse_needed = int(os.environ.get("ADAN_COLLAPSE_WINDOWS", "4") or 4)
        # DIAGNOSTIC-V13 (2026-07-04): the hard-stop breaker is now OPT-IN.
        # Rationale (user): killing a 500k run at first collapse (~40-70k) destroys
        # the wide visual range needed to study the FULL collapse trajectory across
        # sessions. Default OFF = telemetry-only ("measure -> observe, never kill").
        # Set ADAN_COLLAPSE_BREAKER=1 to re-arm the hard stop (e.g. cost-saving runs).
        self._breaker_enabled = (
            os.environ.get("ADAN_COLLAPSE_BREAKER", "0").strip() in ("1", "true", "True")
        )
        # Opt-in critic circuit breaker for long instrumented runs.
        self._critic_breaker_enabled = (
            os.environ.get("ADAN_CRITIC_BREAKER", "0").strip() in ("1", "true", "True")
        )
        self._critic_ev_threshold = float(os.environ.get("ADAN_CRITIC_EV_MIN", "-0.2"))
        self._critic_ev_windows = max(
            1, int(os.environ.get("ADAN_CRITIC_EV_WINDOWS", "10"))
        )
        self._critic_value_loss_max = float(
            os.environ.get("ADAN_CRITIC_VALUE_LOSS_MAX", "1000000")
        )
        self._critic_bad_ev_streak = 0
        self._critic_breaker_reason = ""

    @staticmethod
    def _numeric_stats(values):
        """Finite distribution summary used for critic health telemetry."""
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        finite = array[np.isfinite(array)]
        if not finite.size:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0,
                    "p01": 0.0, "p50": 0.0, "p99": 0.0,
                    "nonfinite_frac": 1.0 if array.size else 0.0}
        return {"min": float(finite.min()), "max": float(finite.max()),
                "mean": float(finite.mean()), "std": float(finite.std()),
                "p01": float(np.quantile(finite, 0.01)),
                "p50": float(np.quantile(finite, 0.50)),
                "p99": float(np.quantile(finite, 0.99)),
                "nonfinite_frac": float(1.0 - finite.size / max(array.size, 1))}

    def _rollout_health(self):
        """Read completed rollout tensors without modifying PPO state."""
        buffer = getattr(self.model, "rollout_buffer", None)
        if buffer is None:
            return {}
        arrays = {name: np.asarray(getattr(buffer, name), dtype=np.float64)
                  for name in ("rewards", "returns", "advantages", "values")}
        returns = arrays["returns"].reshape(-1)
        values = arrays["values"].reshape(-1)
        variance = float(np.var(returns))
        explained_var = (float(1.0 - np.var(returns - values) / variance)
                         if variance > 1e-12 else float("nan"))
        observations = getattr(buffer, "observations", {})
        if not isinstance(observations, dict):
            observations = {"observation": observations}
        obs_stats = {}
        for key, obs in sorted(observations.items()):
            stats = self._numeric_stats(obs)
            flat = np.asarray(obs, dtype=np.float64).reshape(-1)
            finite = flat[np.isfinite(flat)]
            stats["abs_ge_10_frac"] = (
                float(np.mean(np.abs(finite) >= 9.999)) if finite.size else 0.0
            )
            obs_stats[str(key)] = stats
        return {
            "critic_explained_variance": explained_var,
            "reward_stats": json.dumps(self._numeric_stats(arrays["rewards"]), sort_keys=True),
            "return_stats": json.dumps(self._numeric_stats(arrays["returns"]), sort_keys=True),
            "advantage_stats": json.dumps(self._numeric_stats(arrays["advantages"]), sort_keys=True),
            "value_stats": json.dumps(self._numeric_stats(arrays["values"]), sort_keys=True),
            "observation_stats": json.dumps(obs_stats, sort_keys=True),
            "episode_starts": int(np.sum(getattr(buffer, "episode_starts", 0))),
        }

    def _reset_window(self):
        self._a0 = []                 # continuous action0 seen this window
        self._req = {0: 0, 1: 0, 2: 0}  # HOLD / BUY / SELL requested
        self._flat = 0
        self._open = 0
        self._ent = []                # entropy samples this window
        self._rej_delta = 0           # rejections accumulated this window

    @staticmethod
    def _is_open(env) -> bool:
        try:
            pm = getattr(env, "portfolio_manager", None)
            positions = getattr(pm, "positions", None)
            if isinstance(positions, dict):
                for p in positions.values():
                    if bool(getattr(p, "is_open", False)):
                        return True
            elif isinstance(positions, (list, tuple)):
                for p in positions:
                    if bool(getattr(p, "is_open", False)):
                        return True
        except Exception:
            pass
        return False

    def _on_step(self) -> bool:
        try:
            # 1) continuous action0 from the rollout (already sampled by PPO)
            acts = self.locals.get("actions", None)
            if acts is not None:
                arr = np.asarray(acts, dtype=np.float32).reshape(len(acts), -1) \
                    if hasattr(acts, "__len__") else None
                if arr is not None and arr.shape[1] >= 1:
                    for v in arr[:, 0]:
                        self._a0.append(float(v))

            # 2) per-env requested discrete action + position state + rejections
            try:
                reqs = self.training_env.get_attr("_last_discrete_action_requested")
            except Exception:
                reqs = []
            try:
                envs = self.training_env.get_attr("rejection_reasons")
            except Exception:
                envs = []
            # position state via env_method is unsafe (pickling); read per-env attr
            for i in range(len(reqs)):
                r = int(reqs[i] or 0)
                if r in self._req:
                    self._req[r] += 1
                # in-position flag — read via get_attr on the unwrapped env
                try:
                    pm_list = self.training_env.get_attr("portfolio_manager")
                    is_open = False
                    if i < len(pm_list):
                        positions = getattr(pm_list[i], "positions", None)
                        if isinstance(positions, dict):
                            is_open = any(bool(getattr(p, "is_open", False))
                                          for p in positions.values())
                    if is_open:
                        self._open += 1
                    else:
                        self._flat += 1
                except Exception:
                    pass

            # 3) rejection delta (illegal actions) summed across envs
            cur_total = 0
            for rd in envs:
                if isinstance(rd, dict):
                    cur_total += sum(int(v) for v in rd.values())
            if self._prev_rej_total is not None and cur_total >= self._prev_rej_total:
                self._rej_delta += (cur_total - self._prev_rej_total)
            self._prev_rej_total = cur_total

            # 4) policy entropy estimate (cheap: from log_std of the policy)
            try:
                pol = self.model.policy
                if hasattr(pol, "log_std"):
                    log_std = pol.log_std.detach().cpu().numpy().reshape(-1)
                    # diagonal-Gaussian differential entropy per dim
                    ent = float(np.mean(0.5 * np.log(2 * np.pi * np.e) + log_std))
                    self._ent.append(ent)
            except Exception:
                pass

            # Mark the window ready, but do not read returns/advantages here:
            # SB3 computes them only after the final _on_step of the rollout.
            if self.num_timesteps >= self._next_flush:
                self._flush_pending = True
        except Exception:
            pass
        # DIAGNOSTIC-V8: hard stop if collapse confirmed over consecutive windows.
        # V13: only stops when the breaker is explicitly enabled (opt-in). Otherwise
        # the collapse is logged but training continues so the FULL trajectory is
        # captured for cross-session analysis.
        if self._collapse_tripped and self._breaker_enabled:
            logging.getLogger(__name__).critical(
                "[COLLAPSE-BREAKER] Training STOPPED at %d timesteps — policy "
                "collapse confirmed. Inspect the diagnostic CSV; resume from the "
                "last healthy checkpoint, NOT this one.", int(self.num_timesteps))
            return False  # SB3 stops learning when a callback returns False
        if self._critic_breaker_reason and self._critic_breaker_enabled:
            logging.getLogger(__name__).critical(
                "[CRITIC-BREAKER] Training STOPPED at %d timesteps — %s. "
                "Inspect the diagnostic CSV and resume only from a healthy checkpoint.",
                int(self.num_timesteps), self._critic_breaker_reason,
            )
            return False
        return True

    def _on_rollout_end(self) -> None:
        """Flush only after SB3 has finalized returns and advantages."""
        if not self._flush_pending and self.num_timesteps < self._next_flush:
            return
        self._flush()
        self._flush_pending = False
        while self._next_flush <= self.num_timesteps:
            self._next_flush += self.log_every

    def _flush(self):
        try:
            import csv
            a0 = np.asarray(self._a0, dtype=np.float32) if self._a0 else np.zeros(1)
            total_req = sum(self._req.values()) or 1
            total_state = (self._flat + self._open) or 1
            bins = np.linspace(-1.0, 1.0, 11)
            histo, _ = np.histogram(np.clip(a0, -1, 1), bins=bins)
            row = {
                "timesteps": int(self.num_timesteps),
                "a0_mean": round(float(a0.mean()), 4),
                "a0_std": round(float(a0.std()), 4),
                "a0_pct_buy": round(float((a0 > 0.01).mean()), 4),
                "a0_pct_sell": round(float((a0 < -0.01).mean()), 4),
                "a0_pct_hold_band": round(float((np.abs(a0) <= 0.01).mean()), 4),
                "req_HOLD_pct": round(self._req[0] / total_req, 4),
                "req_BUY_pct": round(self._req[1] / total_req, 4),
                "req_SELL_pct": round(self._req[2] / total_req, 4),
                "steps_flat_pct": round(self._flat / total_state, 4),
                "steps_open_pct": round(self._open / total_state, 4),
                "illegal_ratio": round(self._rej_delta / total_state, 4),
                "policy_entropy": round(float(np.mean(self._ent)), 4) if self._ent else 0.0,
                "a0_histo": "|".join(str(int(x)) for x in histo),
            }
            row.update(self._rollout_health())

            # Surface the previous completed PPO update beside the current finalized
            # rollout. SB3 exposes `value_loss` (the critic loss) under train/*.
            logger_values = getattr(
                getattr(self.model, "logger", None), "name_to_value", {}
            ) or {}
            ppo_metrics_available = False
            for metric in (
                "approx_kl",
                "clip_fraction",
                "entropy_loss",
                "policy_gradient_loss",
                "value_loss",
            ):
                raw_value = logger_values.get(
                    f"train/{metric}", logger_values.get(metric)
                )
                ppo_metrics_available = ppo_metrics_available or raw_value is not None
                try:
                    row[metric] = float(raw_value)
                except (TypeError, ValueError):
                    row[metric] = float("nan")
            row["critic_loss"] = row["value_loss"]
            row["policy_loss"] = row["policy_gradient_loss"]
            row["ppo_metrics_available"] = ppo_metrics_available
            row["step"] = row["timesteps"]
            row["explained_variance"] = row.get(
                "critic_explained_variance", float("nan")
            )
            row["value_std"] = json.loads(row.get("value_stats", "{}")).get(
                "std", float("nan")
            )
            row["advantages_std"] = json.loads(
                row.get("advantage_stats", "{}")
            ).get("std", float("nan"))

            # Long-run critic breaker: sustained EV failure, any non-finite rollout
            # tensor, non-finite PPO metrics once available, or exploding value loss.
            ev = float(row.get("critic_explained_variance", float("nan")))
            if np.isfinite(ev) and ev < self._critic_ev_threshold:
                self._critic_bad_ev_streak += 1
            else:
                self._critic_bad_ev_streak = 0
            reasons = []
            for stats_name in (
                "reward_stats", "return_stats", "advantage_stats", "value_stats"
            ):
                stats = json.loads(row.get(stats_name, "{}"))
                if float(stats.get("nonfinite_frac", 0.0)) > 0.0:
                    reasons.append(f"non-finite {stats_name.removesuffix('_stats')}")
            if not np.isfinite(ev):
                reasons.append("non-finite explained variance")
            if ppo_metrics_available:
                for metric in (
                    "approx_kl", "clip_fraction", "entropy_loss",
                    "policy_gradient_loss", "value_loss",
                ):
                    if not np.isfinite(row[metric]):
                        reasons.append(f"non-finite {metric}")
                if row["value_loss"] > self._critic_value_loss_max:
                    reasons.append(
                        f"value_loss {row['value_loss']:.6g} > "
                        f"{self._critic_value_loss_max:.6g}"
                    )
            if self._critic_bad_ev_streak >= self._critic_ev_windows:
                reasons.append(
                    f"explained_variance < {self._critic_ev_threshold:g} for "
                    f"{self._critic_bad_ev_streak} rollouts"
                )
            if reasons and not self._critic_breaker_reason:
                self._critic_breaker_reason = "; ".join(reasons)
                logging.getLogger(__name__).critical(
                    "[CRITIC-DETECT %d] %s -> BREAKER ARMED=%s",
                    row["timesteps"], self._critic_breaker_reason,
                    self._critic_breaker_enabled,
                )
            row["critic_bad_ev_streak"] = self._critic_bad_ev_streak
            row["critic_breaker_reason"] = self._critic_breaker_reason

            # DIAGNOSTIC-V8 collapse detection (measure -> decide, no reward touch).
            # Trip if ANY of the proven collapse signatures holds for this window:
            #   - a0_pct_buy >= 0.97  (near-total BUY, histo degenerate)
            #   - a0_pct_sell >= 0.97 (symmetric guard for the SELL attractor)
            #   - |a0_mean| >= 5.0    (continuous output diverging to +/-inf)
            # Require self._collapse_needed consecutive windows before stopping.
            try:
                _pb = row["a0_pct_buy"]; _ps = row["a0_pct_sell"]
                _am = abs(row["a0_mean"])
                # V13: detector relaxed (user). Thresholds env-configurable and set
                # to NEAR-TOTAL degeneracy so a merely-drifting policy is NOT flagged;
                # only a truly dead one is. Combined with the opt-in breaker (default
                # OFF), the run is never killed prematurely.
                _thr_pb = float(os.environ.get("ADAN_COLLAPSE_PCT", "0.99") or 0.99)
                _thr_am = float(os.environ.get("ADAN_COLLAPSE_A0", "8.0") or 8.0)
                _collapsed = (_pb >= _thr_pb) or (_ps >= _thr_pb) or (_am >= _thr_am)
                if _collapsed:
                    self._collapse_streak += 1
                else:
                    self._collapse_streak = 0
                if self._collapse_streak >= self._collapse_needed:
                    self._collapse_tripped = True
                    logging.getLogger(__name__).critical(
                        "[COLLAPSE-DETECT %d] pct_buy=%.3f pct_sell=%.3f "
                        "a0_mean=%.3f streak=%d -> BREAKER ARMED",
                        row["timesteps"], _pb, _ps, row["a0_mean"],
                        self._collapse_streak)
                elif _collapsed:
                    logging.getLogger(__name__).warning(
                        "[COLLAPSE-WARN %d] pct_buy=%.3f a0_mean=%.3f streak=%d/%d",
                        row["timesteps"], _pb, row["a0_mean"],
                        self._collapse_streak, self._collapse_needed)
            except Exception:
                pass

            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            write_header = not self._header_written and not os.path.exists(self.csv_path)
            with open(self.csv_path, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(row.keys()))
                if write_header:
                    w.writeheader()
                w.writerow(row)
            self._header_written = True
            if self.verbose:
                logging.getLogger(__name__).info(
                    "[DIAG-V3 %d] HOLD=%.1f%% BUY=%.1f%% SELL=%.1f%% | "
                    "flat=%.1f%% open=%.1f%% | illegal=%.3f | a0 mu=%.3f sd=%.3f | "
                    "ent=%.3f | histo=%s",
                    row["timesteps"], row["req_HOLD_pct"] * 100,
                    row["req_BUY_pct"] * 100, row["req_SELL_pct"] * 100,
                    row["steps_flat_pct"] * 100, row["steps_open_pct"] * 100,
                    row["illegal_ratio"], row["a0_mean"], row["a0_std"],
                    row["policy_entropy"], row["a0_histo"],
                )
        except Exception as e:
            logging.getLogger(__name__).warning(f"[DIAG-V3] flush failed: {e}")
        finally:
            self._reset_window()


# ===========================================================================
# OMEGA Worker Profiles
# ===========================================================================

WORKER_PROFILES: Dict[str, Dict[str, Any]] = {
    # ── W0 Scalper 5m ────────────────────────────────────────────────────────
    # Horizon: gamma=0.95 -> ~20 steps = ~1.7h of 5m candles
    # n_steps=512: small rollout, fast learning on noisy 5m signal
    # ent_coef=0.03 (DIAGNOSTIC-V3 2026-06-29): was 0.01. Forensic confusion
    # matrix (430k/480k/500k) proved ENTROPY COLLAPSE: action0 is bimodal at
    # +-1 (std=0.995), HOLD=0%, agent in-position 99.7% of steps. 0.01 was too
    # low to keep exploration alive once the policy found the "always max long"
    # attractor. Tripling ent_coef is the #1 anti-collapse lever (fees held at
    # 0.5% by user decision, so entropy + sterile penalty must carry the fix).
    "scalper": {
        "name": "Scalper",
        "specialization": {"timeframe": "5m"},
        "n_steps": 512,
        "batch_size": 64,
        "learning_rate": 3e-5,
        "ent_coef": 0.03,
        "gamma": 0.95,
        "clip_range": 0.15,
    },
    # ── W1 Intraday 1h ───────────────────────────────────────────────────────
    # Horizon: gamma=0.99 -> ~100 steps = ~4 days of 1h candles
    # n_steps=2048: large enough to capture intraday patterns
    # ent_coef=0.015: higher exploration for diverse intraday regimes
    "intraday": {
        "name": "Intraday",
        "specialization": {"timeframe": "1h"},
        "n_steps": 2048,
        "batch_size": 128,
        "learning_rate": 1e-4,
        "ent_coef": 0.015,
        "gamma": 0.99,
        "clip_range": 0.20,
    },
    # ── W2 Swing 4h ──────────────────────────────────────────────────────────
    # Horizon: gamma=0.995 -> ~200 steps = ~33 days of 4h candles
    # n_steps=8192: very large rollout for long-horizon swing trades
    # ent_coef=0.025: high exploration -- swing must discover rare setups
    "swing": {
        "name": "Swing",
        "specialization": {"timeframe": "4h"},
        "n_steps": 8192,
        "batch_size": 256,
        "learning_rate": 3e-4,
        "ent_coef": 0.025,
        "gamma": 0.995,
        "clip_range": 0.25,
    },
    # ── W3 Position 4h ───────────────────────────────────────────────────────
    # Horizon: gamma=0.999 -> ~1000 steps = ~166 days of 4h candles
    # n_steps=16384: ultra-long rollout for macro trend following
    # ent_coef=0.04: maximum exploration for rare multi-month signals
    "position": {
        "name": "Position",
        "specialization": {"timeframe": "4h"},
        "n_steps": 16384,
        "batch_size": 512,
        "learning_rate": 5e-4,
        "ent_coef": 0.04,
        "gamma": 0.999,
        "clip_range": 0.30,
    },
}


def _resolve_exact_rollout_steps(preferred_steps: int, iteration_steps: int) -> int:
    """Choose the largest profile-compatible rollout that divides an iteration.

    Stable-Baselines3 always completes a full rollout. If ``n_steps`` exceeds or
    does not divide Ray's per-iteration budget, ``learn()`` silently overshoots
    (the 8,192-step smoke produced 16,384 steps for the Position profile).
    Keeping a divisor makes Ray, SB3, checkpoints, and progress.csv agree.
    """
    preferred = max(1, int(preferred_steps))
    budget = max(1, int(iteration_steps))
    upper = min(preferred, budget)
    for candidate in range(upper, 0, -1):
        if budget % candidate == 0:
            return candidate
    return 1


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
    use_subproc: bool = False,
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
    # =========================================================================
    # DOUBLE NORMALIZATION FIX (Session 15+)
    # =========================================================================
    # MultiAssetChunkedEnv internally uses StateBuilder(normalize=True) which
    # applies per-timeframe scalers (MinMax for 5m, Standard for 1h, Robust
    # for 4h) AND clips to [-10, +10].  Wrapping with VecNormalize(norm_obs=True)
    # on top applies a SECOND running z-score, compressing already-normalized
    # values and destroying the signal calibration.
    #
    # Evidence: The sandbox path (line ~1620) deliberately DISABLED VecNormalize
    # with the comment "observations are already normalized in StateBuilder".
    # The PBT path must be consistent.
    #
    # norm_obs=False: StateBuilder already normalizes observations
    # norm_reward=False: symlog in reward pipeline already compresses outliers
    # clip_obs kept at 10.0: safety net matching StateBuilder's own clip range
    #
    # Audit anomaly #5 (preserved): DO NOT z-score the reward — see original
    # reasoning about symlog + time_decay calibration above.
    vec_env = VecNormalize(
        vec_env,
        norm_obs=False,
        norm_reward=False,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=gamma,
        training=True,
    )
    return vec_env


# ===========================================================================
# Ray Tune Trainable – ADAN_PBT_Worker
# ===========================================================================

# Base class depends on whether ray[tune] is available
_TrainableBase = tune.Trainable if RAY_AVAILABLE else object


class ADAN_PBT_Worker(_TrainableBase):
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
        # Support paired worker_config dict (avoids cartesian product bug)
        wc = config.get("worker_config", {})
        self.worker_idx = wc.get("worker_idx", config.get("worker_idx", 0))
        self.profile = wc.get("profile", config.get("profile", None))  # scalper / swing / ...
        self.envs_per_worker = config.get("envs_per_worker", 2)
        self.use_subproc = config.get("use_subproc", False)
        self.interval_timesteps = config.get("interval_timesteps", 5_000)  # Reduced from 15k to avoid hangs
        self._total_timesteps = 0
        self._completed_iterations = 0
        self._max_iterations = config.get("_max_iterations", 100)

        # Checkpoint directory: use Ray's trial logdir for per-worker checkpoints
        # Ray Trainable provides self.logdir as the trial-specific directory
        self.checkpoint_dir = os.path.join(
            getattr(self, "logdir", str(TRAIN_OUTPUT_DIR / "checkpoints")),
            "adan_checkpoints"
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Mutable hyper-parameters (PBT will perturb these)
        # IMPORTANT: Use profile-specific values as initial seeds if available,
        # falling back to PBT's random sample. This ensures each worker starts
        # with its research-calibrated hyperparams before PBT explores.
        _prof = WORKER_PROFILES.get(self.profile, {}) if self.profile else {}
        self.learning_rate = _prof.get("learning_rate", config.get("learning_rate", 3e-4))
        self.ent_coef = _prof.get("ent_coef", config.get("ent_coef", 0.01))
        self.gamma = _prof.get("gamma", config.get("gamma", 0.99))
        
        # Trading hyperparams (Ray PBT auto-evolves these)
        self.sl_pct = config.get("sl_pct", 0.02)  # Stop-Loss percentage
        self.tp_pct = config.get("tp_pct", 0.04)  # Take-Profit percentage

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
        
        # **NEW**: Inject PBT-evolved SL/TP into trading_parameters
        # This allows Ray to optimize these critical trading parameters
        if "trading_parameters" not in worker_config:
            worker_config["trading_parameters"] = {}
        
        worker_config["trading_parameters"]["stop_loss_pct"] = self.sl_pct
        worker_config["trading_parameters"]["take_profit_pct"] = self.tp_pct
        
        logger.info(
            f"Worker {self.worker_idx}: PBT trading params: SL={self.sl_pct:.2%}, TP={self.tp_pct:.2%}"
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
            safe_fe_kwargs.setdefault("context_dim", 14)
            policy_kwargs.setdefault("features_extractor_kwargs", safe_fe_kwargs)

        # CRITICAL: share the feature extractor between actor and critic.
        # Without this, SB3 instantiates it 3x (shared + pi + vf),
        # tripling memory usage and compute for no benefit.
        policy_kwargs["share_features_extractor"] = True

        # gSDE STABILITY (V2 execution audit, 2026-06-24 — MEASURED):
        # σ_eff ≈ ||features||_2 * exp(log_std_init). With the real extractor
        # (features_dim=256) ||features||_2≈11.4 (scripts/diag_gsde_latent.py),
        # so the historical log_std_init=-0.5 gave σ_eff≈6.9 AT INIT -> gSDE
        # diverged and the net defended by saturating size to μ=-7. We now
        # default to -2.0 (σ_eff≈1.5) + use_expln=True (bounds growth). This is
        # the SAME fix as sandbox so a 500k run does not repeat the collapse.
        # Override via ADAN_LOG_STD_INIT / ADAN_USE_EXPLN if needed.
        _use_sde = os.environ.get("ADAN_USE_SDE", "1") == "1"
        _default_log_std = "-2.0" if _use_sde else "-1.0"
        _log_std_init = float(os.environ.get("ADAN_LOG_STD_INIT", _default_log_std))
        policy_kwargs["log_std_init"] = _log_std_init
        if _use_sde and os.environ.get("ADAN_USE_EXPLN", "1") == "1":
            policy_kwargs["use_expln"] = True
        logger.info(
            f"Worker {self.worker_idx}: exploration="
            f"{'gSDE' if _use_sde else 'DiagGaussian'} "
            f"log_std_init={_log_std_init:+.3f} "
            f"(std0≈{float(np.exp(_log_std_init)):.3f}) use_expln="
            f"{policy_kwargs.get('use_expln', False)}"
            + (f" -> σ_eff≈{11.4*float(np.exp(_log_std_init)):.2f} at init."
               if _use_sde else " -> variance decoupled from feature norm.")
        )

        # Seed
        seed = self.adan_config.get("general", {}).get("random_seed", 42) + self.worker_idx
        if SeedManager is not None:
            SeedManager.initialize(seed)

        # PPO model – profile may override n_steps/batch_size
        device = "cuda" if torch.cuda.is_available() else "cpu"
        prof_cfg = WORKER_PROFILES.get(self.profile, {}) if self.profile else {}
        preferred_n_steps = int(
            prof_cfg.get("n_steps", agent_cfg.get("n_steps", 2048))
        )
        n_steps = _resolve_exact_rollout_steps(
            preferred_n_steps,
            self.interval_timesteps,
        )
        if n_steps != preferred_n_steps:
            logger.warning(
                "Worker %s (%s): n_steps adjusted %s -> %s so the %s-step "
                "Ray iteration is exact",
                self.worker_idx,
                self.profile,
                preferred_n_steps,
                n_steps,
                self.interval_timesteps,
            )
        batch_size = prof_cfg.get("batch_size", agent_cfg.get("batch_size", 64))
        # Ensure batch_size divides n_steps * envs_per_worker
        total_rollout = n_steps * self.envs_per_worker
        if total_rollout % batch_size != 0:
            batch_size = max(1, total_rollout // max(1, total_rollout // batch_size))

        # Profile overrides for gamma and clip_range (research-calibrated per trading style)
        gamma_final     = prof_cfg.get("gamma",      self.gamma)
        clip_range_final= prof_cfg.get("clip_range", agent_cfg.get("clip_range", 0.2))
        ent_coef_final  = prof_cfg.get("ent_coef",   self.ent_coef)
        lr_final        = prof_cfg.get("learning_rate", self.learning_rate)
        # Override V2: ADAN_ENT_COEF force l'entropie pour TOUS les profils (run
        # diagnostique). Pousse l'agent hors du plateau ; ne réveille pas SIZE à
        # lui seul (cause μ) mais aide TP/SL et augmente la variance des rollouts.
        _ent_override = os.environ.get("ADAN_ENT_COEF")
        if _ent_override is not None:
            ent_coef_final = float(_ent_override)
            logger.info(
                f"Worker {self.worker_idx}: ent_coef={ent_coef_final:.4f} "
                f"(override V2 via ADAN_ENT_COEF)."
            )

        logger.info(
            f"Worker {self.worker_idx} ({self.profile}): "
            f"n_steps={n_steps} batch={batch_size} gamma={gamma_final:.4f} "
            f"clip={clip_range_final} ent={ent_coef_final:.4f} lr={lr_final:.2e}"
        )

        # Each worker gets its own TB log dir so curves are separate
        profile_tag = self.profile or f"w{self.worker_idx}"
        tb_log_dir = str(TRAIN_OUTPUT_DIR / "tb_workers" / f"worker_{self.worker_idx}_{profile_tag}")

        # SOTA 2026: Use WorldModelPPO for auxiliary forward-prediction loss
        PPOClass = WorldModelPPO if WorldModelPPO is not None else PPO
        ppo_kwargs = dict(
            policy="MultiInputPolicy",
            env=self.vec_env,
            device=device,
            learning_rate=lr_final,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=agent_cfg.get("n_epochs", 10),
            gamma=gamma_final,
            gae_lambda=agent_cfg.get("gae_lambda", 0.95),
            clip_range=clip_range_final,
            ent_coef=ent_coef_final,
            vf_coef=agent_cfg.get("vf_coef", 0.5),
            max_grad_norm=agent_cfg.get("max_grad_norm", 0.5),
            policy_kwargs=policy_kwargs if policy_kwargs else None,
            tensorboard_log=tb_log_dir,
            # Heavy mode must honor the same measured exploration switch as
            # sandbox mode. ADAN_USE_SDE=0 selects the stable DiagGaussian path
            # recommended for the production 500k run.
            use_sde=_use_sde,
            sde_sample_freq=4,  # Resample exploration noise every 4 steps
            verbose=1,
            seed=seed,
        )
        if PPOClass is WorldModelPPO:
            ppo_kwargs["aux_loss_coef"] = 0.1
            logger.info(f"Worker {self.worker_idx}: Using WorldModelPPO with aux_loss_coef=0.1")
        self.model = PPOClass(**ppo_kwargs)

        # Callbacks
        self._callbacks = []
        metrics_monitor = MetricsMonitor(
            config=self.adan_config,
            num_workers=self.envs_per_worker,
            log_interval=max(500, self.interval_timesteps // 10),
        )
        self._callbacks.append(metrics_monitor)

        if PpoStdSafetyCallback is not None:
            # V29 PATCH 1 (2026-08-12): log_std borne dans [-2, 0] au lieu de
            # [-5, +2]. Le clamp precedent autorisait sigma jusqu'a exp(2)=7.39
            # -> rupture mesuree V28 a step 90112 (a0_std median 0.62 -> 8.64,
            # x14, direction_sat_frac 1.00). Avec [-2,0], sigma <= 1.0 par
            # construction (gate smoke 2048: a0_std < 1.0). Overridable via
            # ADAN_LOG_STD_MIN / ADAN_LOG_STD_MAX pour diagnostics.
            ppo_safety = PpoStdSafetyCallback(
                min_log_std=float(os.environ.get("ADAN_LOG_STD_MIN", "-2.0")),
                max_log_std=float(os.environ.get("ADAN_LOG_STD_MAX", "0.0")),
                std_warn_threshold=1.0,
                verbose=0,
            )
            self._callbacks.append(ppo_safety)

        # V31-500k FIX: ActionSaturationGuard — anti-collapse par tete.
        # PpoStdSafetyCallback ne clampe que la borne HAUTE de log_std; ce guard
        # RELEVE le plancher quand une tete se verrouille a +-1 (boundary lock).
        # Defaut ON pour V31; desactivable via ADAN_SATGUARD=0.
        if ActionSaturationGuard is not None and \
                os.environ.get("ADAN_SATGUARD", "1") == "1":
            sat_guard = ActionSaturationGuard(
                sat_edge=0.98,
                sat_threshold=float(os.environ.get("ADAN_SATGUARD_THR", "0.95")),
                patience=int(os.environ.get("ADAN_SATGUARD_PATIENCE", "2")),
                bump_log_std=float(os.environ.get("ADAN_SATGUARD_BUMP", "0.5")),
                max_log_std=float(os.environ.get("ADAN_SATGUARD_MAXLS", "2.0")),
                intervene=os.environ.get("ADAN_SATGUARD_INTERVENE", "1") == "1",
                verbose=1,
            )
            self._callbacks.append(sat_guard)
            logger.info(
                f"Worker {self.worker_idx}: ActionSaturationGuard ACTIF "
                f"(thr={sat_guard.sat_threshold}, bump={sat_guard.bump_log_std}, "
                f"intervene={sat_guard.intervene})"
            )

        # ActionDimMonitor (MESURE SEULE) — suit μ/σ pré-tanh + post-tanh par tête.
        # Activé seulement si ADAN_ACTIONDIM=1 (run diagnostique V2). NE MODIFIE
        # RIEN ; permet d'observer si μ(size)=-7.2 remonte au fil de l'entraînement.
        if ActionDimMonitor is not None and os.environ.get("ADAN_ACTIONDIM", "0") == "1":
            _ad_csv = os.environ.get(
                "ADAN_ACTIONDIM_CSV",
                str(TRAIN_OUTPUT_DIR / f"actiondim_worker_{self.worker_idx}_{profile_tag}.csv"),
            )
            _ad_every = int(os.environ.get("ADAN_ACTIONDIM_EVERY", "1"))
            action_dim_monitor = ActionDimMonitor(
                log_every=_ad_every,
                pre_tanh_batch=int(os.environ.get("ADAN_ACTIONDIM_BATCH", "256")),
                csv_path=_ad_csv,
                verbose=1,
            )
            self._callbacks.append(action_dim_monitor)
            logger.info(
                f"Worker {self.worker_idx}: ActionDimMonitor ACTIF "
                f"(every={_ad_every}, csv={_ad_csv}) — mesure seule, ne modifie rien."
            )

        self._metrics_monitor = metrics_monitor
        
        # Initialize checkpoint tracking for robust 2500-step saves
        self._last_checkpoint_step = 0

    def _sync_mutable_hyperparameters(self) -> None:
        """Synchronize PBT values across every SB3/VecNormalize consumer."""
        self.model.learning_rate = self.learning_rate
        self.model.lr_schedule = FloatSchedule(self.learning_rate)
        optimizer = getattr(getattr(self.model, "policy", None), "optimizer", None)
        if optimizer is not None:
            update_learning_rate(optimizer, self.learning_rate)
        self.model.ent_coef = self.ent_coef
        self.model.gamma = self.gamma
        if hasattr(self.model, "rollout_buffer"):
            self.model.rollout_buffer.gamma = self.gamma
        if hasattr(self.vec_env, "gamma"):
            self.vec_env.gamma = self.gamma

    def reset_config(self, new_config: Dict[str, Any]) -> bool:
        """Apply PBT mutations while safely reusing this trial actor.

        Ray calls this method before restoring the exploited checkpoint.  Worker
        identity and the environment stay attached to the target actor: PBT may
        clone a source trial whose ``worker_config`` names another temporal
        profile, but rebuilding or silently switching the target environment
        here would invalidate its data/profile assignment.  Only the parameters
        declared in ``hyperparam_mutations`` are updated.

        Capital tiers are immutable runtime configuration.  Actor reuse is
        rejected loudly if a future scheduler ever attempts to alter them.
        """
        incoming_adan_config = new_config.get("adan_config", self.adan_config)
        if incoming_adan_config.get("capital_tiers") != self.adan_config.get("capital_tiers"):
            raise ValueError("PBT actor reuse cannot mutate capital_tiers")

        old_worker_idx = self.worker_idx
        old_profile = self.profile
        incoming_worker_config = new_config.get("worker_config", {})
        if incoming_worker_config:
            new_config["worker_config"] = {
                **incoming_worker_config,
                "worker_idx": old_worker_idx,
                "profile": old_profile,
            }
        else:
            new_config["worker_idx"] = old_worker_idx
            new_config["profile"] = old_profile

        self.learning_rate = float(new_config.get("learning_rate", self.learning_rate))
        self.ent_coef = float(new_config.get("ent_coef", self.ent_coef))
        self.gamma = float(new_config.get("gamma", self.gamma))
        self.sl_pct = float(new_config.get("sl_pct", self.sl_pct))
        self.tp_pct = float(new_config.get("tp_pct", self.tp_pct))
        self.interval_timesteps = int(
            new_config.get("interval_timesteps", self.interval_timesteps)
        )
        self._max_iterations = int(new_config.get("_max_iterations", self._max_iterations))
        # Ray restores the exploited source checkpoint after reset_config(). Keep
        # the target mutations explicitly so source metadata cannot overwrite them.
        self._pbt_pending_mutations = {
            "learning_rate": self.learning_rate,
            "ent_coef": self.ent_coef,
            "gamma": self.gamma,
            "sl_pct": self.sl_pct,
            "tp_pct": self.tp_pct,
            "worker_idx": old_worker_idx,
            "profile": old_profile,
        }

        # SB3 consumes lr_schedule during train(), not model.learning_rate.
        self._sync_mutable_hyperparameters()

        logger.info(
            "Worker %s (%s): PBT actor reset applied: lr=%.2e ent=%.4f "
            "gamma=%.4f SL=%.2f%% TP=%.2f%%; capital tiers unchanged",
            self.worker_idx,
            self.profile,
            self.learning_rate,
            self.ent_coef,
            self.gamma,
            self.sl_pct * 100.0,
            self.tp_pct * 100.0,
        )
        return True

    def step(self):
        """Run one training iteration (interval_timesteps steps of PPO.learn)."""
        # Apply mutable hyperparameters (PBT perturbs these between iterations).
        self._sync_mutable_hyperparameters()

        model_steps_before = int(getattr(self.model, "num_timesteps", 0))
        self.model.learn(
            total_timesteps=self.interval_timesteps,
            callback=self._callbacks,
            reset_num_timesteps=False,
        )
        model_steps_after = int(getattr(self.model, "num_timesteps", 0))
        learned_steps = model_steps_after - model_steps_before
        if learned_steps <= 0:
            raise RuntimeError(
                f"PPO learned no steps: before={model_steps_before}, "
                f"after={model_steps_after}"
            )
        self._total_timesteps += learned_steps
        if learned_steps != self.interval_timesteps:
            raise RuntimeError(
                "PPO/Ray timestep contract violated: requested "
                f"{self.interval_timesteps}, learned {learned_steps}"
            )

        # EXPERT FIX: Explicit GC after each iteration to prevent memory accumulation
        # that causes Ray GCS crashes after ~4000 steps (ObjectRef retention + metadata growth)
        import gc
        gc.collect()

        # Ray results must describe the environment *after* this learn() call,
        # never the last periodic callback sample. On the final iteration, close
        # positions before taking that snapshot so terminal receipts, episode/run
        # PnL, cash and equity all reach result.json/progress.csv atomically.
        final_iteration = (
            self._completed_iterations + 1 >= self._max_iterations
        )
        if final_iteration:
            terminal_receipts_by_env = self.vec_env.env_method(
                "finalize_open_positions",
                reason="TRAINING_END",
            )
            for env_index, terminal_receipts in enumerate(
                terminal_receipts_by_env
            ):
                terminal_pnl = sum(
                    float(
                        receipt.get(
                            "pnl_net",
                            receipt.get("pnl", 0.0),
                        )
                        or 0.0
                    )
                    for receipt in (terminal_receipts or [])
                    if isinstance(receipt, dict)
                )
                self.vec_env.env_method(
                    "_finalize_episode_financial_telemetry",
                    reset_close_pnl=terminal_pnl,
                    indices=env_index,
                )

        self._metrics_monitor._collect_worker_metrics()

        # CHECKPOINT: Save every 15k steps for crash recovery
        # Verify vec_env obs_rms consistency (detect NaN corruption from divergence)
        if hasattr(self.vec_env, 'obs_rms') and self.vec_env.obs_rms is not None:
            _obs_rms = self.vec_env.obs_rms
            _rms_items = _obs_rms.items() if isinstance(_obs_rms, dict) else [("obs", _obs_rms)]
            for _key, _rms in _rms_items:
                if hasattr(_rms, 'mean') and np.any(np.isnan(_rms.mean)):
                    logger.error(f"❌ VecNormalize obs_rms[{_key}] has NaN mean! Resetting stats.")
                    _rms.mean[:] = 0.0
                    _rms.var[:] = 1.0
                    _rms.count = 1e-4

        # ROBUST CHECKPOINT: Save every 2500 steps (not modulo to avoid missed crossings)
        # Track last saved checkpoint to ensure we save AT LEAST every 2500 steps
        checkpoint_interval = 2_500
        if not hasattr(self, '_last_checkpoint_step'):
            self._last_checkpoint_step = 0
        
        steps_since_last_checkpoint = self._total_timesteps - self._last_checkpoint_step
        
        # Save if we've accumulated >= checkpoint_interval steps since last save
        if steps_since_last_checkpoint >= checkpoint_interval:
            try:
                checkpoint_dir = os.path.join(
                    self.checkpoint_dir,
                    f"checkpoint_{self._total_timesteps:08d}"
                )
                self.save_checkpoint(checkpoint_dir)
                self._last_checkpoint_step = self._total_timesteps
                logger.info(f"✅ Checkpoint saved at {self._total_timesteps} steps (interval: {checkpoint_interval})")
            except Exception as e:
                logger.error(f"❌ Checkpoint save failed at {self._total_timesteps} steps: {e}")

        # Collect metrics
        mean_reward = 0.0
        mean_sharpe = 0.0
        mean_balance = 0.0
        open_positions = 0
        realized_pnl = 0.0
        realized_pnl_step = 0.0
        realized_pnl_episode = 0.0
        realized_pnl_episode_current = 0.0
        realized_pnl_cumulative = 0.0
        cash = 0.0
        equity = 0.0
        realized_equity = 0.0
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
                mean_balance = wm["portfolio_values"][-1]  # Latest total equity snapshot
            if wm.get("realized_pnls"):
                realized_pnl = wm["realized_pnls"][-1]
            if wm.get("realized_pnl_steps"):
                realized_pnl_step = wm["realized_pnl_steps"][-1]
            if wm.get("realized_pnl_episodes"):
                realized_pnl_episode = wm["realized_pnl_episodes"][-1]
            if wm.get("realized_pnl_episode_currents"):
                realized_pnl_episode_current = wm["realized_pnl_episode_currents"][-1]
            if wm.get("realized_pnl_cumulatives"):
                realized_pnl_cumulative = wm["realized_pnl_cumulatives"][-1]
            if wm.get("cash_values"):
                cash = wm["cash_values"][-1]
            if wm.get("equity_values"):
                equity = wm["equity_values"][-1]
            if wm.get("realized_equity_values"):
                realized_equity = wm["realized_equity_values"][-1]
        except Exception:
            pass

        # Surface PPO health in Ray result.json/progress.csv. SB3 already
        # computes these values; exporting them is measurement-only and lets the
        # supervisor detect critic/policy drift per trial instead of relying on
        # a population average or scraping terminal tables.
        _logger_values = getattr(getattr(self.model, "logger", None), "name_to_value", {}) or {}
        _ppo_health = {}
        for _metric in (
            "approx_kl",
            "clip_fraction",
            "entropy_loss",
            "explained_variance",
            "policy_gradient_loss",
            "value_loss",
            "std",
        ):
            _value = _logger_values.get(f"train/{_metric}", _logger_values.get(_metric))
            try:
                _ppo_health[_metric] = float(_value)
            except (TypeError, ValueError):
                _ppo_health[_metric] = float("nan")

        # Ray's Trainable.training_iteration is owned by Tune and is incremented
        # only after step() returns, so reading it here is always one iteration
        # behind (and may be absent). Track completed learn() calls locally to
        # stop exactly at the requested per-trial budget.
        self._completed_iterations += 1
        done = self._completed_iterations >= self._max_iterations

        return {
            "mean_reward": mean_reward,
            **_ppo_health,
            "mean_sharpe": mean_sharpe,
            # Keep historical semantics: latest portfolio total-value snapshot.
            "mean_balance": mean_balance,
            "realized_pnl": realized_pnl,
            "realized_pnl_step": realized_pnl_step,
            # Most recently completed episode; current episode is explicit too.
            "realized_pnl_episode": realized_pnl_episode,
            "realized_pnl_episode_current": realized_pnl_episode_current,
            "realized_pnl_cumulative": realized_pnl_cumulative,
            "cash": cash,
            "equity": equity,
            "realized_equity": realized_equity,
            "learning_rate": self.learning_rate,
            "ent_coef": self.ent_coef,
            "gamma": self.gamma,
            "sl_pct": self.sl_pct,              # Stop-Loss % (PBT optimizes)
            "tp_pct": self.tp_pct,              # Take-Profit % (PBT optimizes)
            "timesteps_total": self._total_timesteps,
            "done": done,
        }

    def save_checkpoint(self, checkpoint_dir: str) -> str:
        """Save PPO model + VecNormalize stats atomically with integrity checks."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save to temp files first (atomic write pattern)
        model_path_tmp = os.path.join(checkpoint_dir, "model.zip.tmp")
        vec_path_tmp = os.path.join(checkpoint_dir, "vecnormalize.pkl.tmp")
        
        try:
            # Write to temp
            self.model.save(model_path_tmp)
            self.vec_env.save(vec_path_tmp)
            
            # Atomic rename
            model_path = os.path.join(checkpoint_dir, "model.zip")
            vec_path = os.path.join(checkpoint_dir, "vecnormalize.pkl")
            os.replace(model_path_tmp, model_path)
            os.replace(vec_path_tmp, vec_path)
            
            # Integrity check: verify files are non-empty after rename
            _model_size = os.path.getsize(model_path)
            _vec_size = os.path.getsize(vec_path)
            if _model_size < 1024:
                raise RuntimeError(f"Model checkpoint suspiciously small: {_model_size} bytes")
            if _vec_size < 64:
                raise RuntimeError(f"VecNormalize checkpoint suspiciously small: {_vec_size} bytes")
            
            # Save state with metadata
            state = {
                "total_timesteps": self._total_timesteps,
                "learning_rate": self.learning_rate,
                "ent_coef": self.ent_coef,
                "gamma": self.gamma,
                "worker_idx": self.worker_idx,
                "profile": self.profile,
                "timestamp": datetime.now().isoformat(),
                "model_size_bytes": os.path.getsize(model_path),
                "vecnorm_size_bytes": os.path.getsize(vec_path),
            }
            state_path = os.path.join(checkpoint_dir, "worker_state.json")
            with open(state_path, "w") as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"✅ Checkpoint saved: {checkpoint_dir} "
                       f"(steps={self._total_timesteps}, lr={self.learning_rate:.2e})")
            return checkpoint_dir
            
        except Exception as e:
            logger.error(f"❌ Checkpoint save failed: {e}", exc_info=True)
            # Cleanup temp files
            for tmp_file in [model_path_tmp, vec_path_tmp]:
                if os.path.exists(tmp_file):
                    try:
                        os.remove(tmp_file)
                    except:
                        pass
            raise

    def load_checkpoint(
        self,
        checkpoint_dir: str,
        *,
        restore_hyperparameters: bool = True,
    ):
        """Restore model state, optionally preserving current PBT mutations."""
        model_path = os.path.join(checkpoint_dir, "model.zip")
        vec_path = os.path.join(checkpoint_dir, "vecnormalize.pkl")
        state_path = os.path.join(checkpoint_dir, "worker_state.json")
        
        try:
            # Verify files exist
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found: {model_path}")
            if not os.path.exists(vec_path):
                raise FileNotFoundError(f"VecNormalize not found: {vec_path}")
            
            # Load model — use GPU if available (checkpoint resume should match training device)
            PPOClass = WorldModelPPO if WorldModelPPO is not None else PPO
            _load_device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = PPOClass.load(model_path, env=self.vec_env, device=_load_device)
            logger.info(f"✅ Model loaded: {model_path} (device={_load_device})")
            
            # Load VecNormalize onto the underlying venv
            venv = self.vec_env.venv if hasattr(self.vec_env, "venv") else self.vec_env
            self.vec_env = VecNormalize.load(vec_path, venv)
            # CRITICAL: Sync gamma from current PBT state (not stale checkpoint value)
            self.vec_env.gamma = self.gamma
            # DOUBLE NORMALIZATION FIX: Force norm_obs=False even when loading
            # from old checkpoints that had norm_obs=True. StateBuilder already
            # normalizes observations internally.
            self.vec_env.norm_obs = False
            self.vec_env.norm_reward = False
            self.model.set_env(self.vec_env)
            logger.info(f"✅ VecNormalize loaded: {vec_path} (gamma synced to {self.gamma:.4f}, norm_obs=False)")
            
            # Load temporal state. During PBT exploit, source hyperparameters are
            # deliberately excluded: reset_config() already applied target values.
            if os.path.exists(state_path):
                with open(state_path) as f:
                    state = json.load(f)
                self._total_timesteps = state.get("total_timesteps", 0)
                if restore_hyperparameters:
                    self.learning_rate = state.get("learning_rate", self.learning_rate)
                    self.ent_coef = state.get("ent_coef", self.ent_coef)
                    self.gamma = state.get("gamma", self.gamma)
                logger.info(f"✅ State restored: steps={self._total_timesteps}, "
                           f"lr={self.learning_rate:.2e}, ent_coef={self.ent_coef:.4f}")
            else:
                logger.warning(f"⚠️  State file not found: {state_path}")
                
        except Exception as e:
            logger.error(f"❌ Checkpoint load failed: {e}", exc_info=True)
            raise

    def _save(self, checkpoint_dir: str) -> str:
        """Ray Tune protocol: save ONLY serializable state.
        
        CRITICAL: Ray Tune WILL call this after each step().
        We MUST NOT try to pickle the env or model directly.
        
        Strategy:
          1. Store metadata (hyperparams) in JSON
          2. Delegate model save to save_checkpoint() → creates model.zip
          3. Ray tracks only the metadata; model lives in our checkpoint dir
        """
        try:
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save metadata (non-serializable env/model not included)
            metadata = {
                "total_timesteps": self._total_timesteps,
                "learning_rate": self.learning_rate,
                "ent_coef": self.ent_coef,
                "gamma": self.gamma,
                "worker_idx": self.worker_idx,
                "profile": self.profile,
                "sl_pct": self.sl_pct,
                "tp_pct": self.tp_pct,
                "timestamp": datetime.now().isoformat(),
                "checkpoint_dir": self.checkpoint_dir,  # Reference for restore
            }
            
            meta_path = os.path.join(checkpoint_dir, "ray_metadata.json")
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            # Actually save model to our custom checkpoint location
            # (Ray doesn't know about it, but it persists on disk independently)
            checkpoint_subdir = os.path.join(
                self.checkpoint_dir,
                f"checkpoint_{self._total_timesteps:08d}"
            )
            self.save_checkpoint(checkpoint_subdir)
            
            logger.info(f"✅ Ray Tune checkpoint saved (metadata+model): {checkpoint_dir}")
            return checkpoint_dir
            
        except Exception as e:
            logger.error(f"❌ Ray Tune _save() failed: {e}", exc_info=True)
            raise

    def _restore(self, checkpoint_path: str):
        """Ray Tune protocol: restore from metadata + reload model from disk.
        
        Ray calls this to restore from a checkpoint_dir.
        We load the metadata and then locate the actual model on disk.
        """
        try:
            import json
            
            meta_path = os.path.join(checkpoint_path, "ray_metadata.json")
            if not os.path.exists(meta_path):
                logger.warning(f"⚠️  No Ray metadata found at {meta_path}; skipping restore")
                return
            
            with open(meta_path) as f:
                metadata = json.load(f)
            
            source_timesteps = int(metadata.get("total_timesteps", 0))
            pending = getattr(self, "_pbt_pending_mutations", None)
            if pending is None:
                # Normal resume (not PBT exploit): checkpoint hyperparameters apply.
                self.learning_rate = metadata.get("learning_rate", self.learning_rate)
                self.ent_coef = metadata.get("ent_coef", self.ent_coef)
                self.gamma = metadata.get("gamma", self.gamma)
                self.sl_pct = metadata.get("sl_pct", self.sl_pct)
                self.tp_pct = metadata.get("tp_pct", self.tp_pct)

            model_checkpoint_dir = metadata.get(
                "checkpoint_dir", getattr(self, "checkpoint_dir", "")
            )
            if os.path.exists(model_checkpoint_dir):
                exact_checkpoint = os.path.join(
                    model_checkpoint_dir,
                    f"checkpoint_{source_timesteps:08d}",
                )
                if os.path.isdir(exact_checkpoint):
                    self.load_checkpoint(
                        exact_checkpoint,
                        restore_hyperparameters=pending is None,
                    )
                    if pending is not None:
                        self.learning_rate = pending["learning_rate"]
                        self.ent_coef = pending["ent_coef"]
                        self.gamma = pending["gamma"]
                        self.sl_pct = pending["sl_pct"]
                        self.tp_pct = pending["tp_pct"]
                        self.worker_idx = pending["worker_idx"]
                        self.profile = pending["profile"]
                    self._total_timesteps = source_timesteps
                    self._sync_mutable_hyperparameters()
                    self._pbt_pending_mutations = None
                    logger.info(f"✅ Ray Tune restore: loaded model from {exact_checkpoint}")
                    logger.info(f"   Restored state: steps={self._total_timesteps}, "
                               f"lr={self.learning_rate:.2e}, "
                               f"SL%={self.sl_pct:.2%}, TP%={self.tp_pct:.2%}")
                else:
                    raise FileNotFoundError(
                        "Exact PBT checkpoint not found for "
                        f"total_timesteps={source_timesteps}: {exact_checkpoint}"
                    )
            else:
                logger.warning(f"⚠️  Checkpoint dir not found: {model_checkpoint_dir}")
                
        except Exception as e:
            logger.error(f"❌ Ray Tune _restore() failed: {e}", exc_info=True)
            raise

    def cleanup(self):
        """Finalize auditable positions before closing vector environments."""
        vec_env = getattr(self, "vec_env", None)
        if vec_env is None:
            return
        try:
            vec_env.env_method(
                "finalize_open_positions",
                reason="TRAINING_END",
            )
        except Exception as exc:
            logger.error(
                "Worker %s failed to finalize positions during cleanup: %s",
                getattr(self, "worker_idx", "unknown"),
                exc,
                exc_info=True,
            )
        finally:
            vec_env.close()


# ===========================================================================
# PBT setup and launch
# ===========================================================================

def run_pbt(
    config: dict,
    num_cpus: int = 8,
    num_samples: int = 4,
    resume: bool = False,
    envs_per_worker: int = 1,   # OOM FIX: 1 env per worker to minimize RAM
    use_subproc: bool = False,
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
        envs_per_worker: Sub-envs per trial (1 = minimal RAM, DummyVecEnv).
        use_subproc: Whether to use SubprocVecEnv (default False to avoid Ray/fork conflicts).
        total_steps: Total training timesteps per trial.
        interval_timesteps: Timesteps per PBT iteration.
        stop_config: Optional tune stop dict.
        storage_path: Where Ray stores results.
        profiles: Optional list of profile names (e.g. ['scalper', 'swing']).
    """
    if storage_path is None:
        storage_path = str((TRAIN_OUTPUT_DIR / "ray_results").resolve())
    else:
        # Ray 2.56 routes storage_path through pyarrow.fs and rejects relative
        # paths as URIs with an empty scheme.
        storage_path = str(Path(storage_path).expanduser().resolve())

    max_iterations = max(1, total_steps // interval_timesteps)

    # PBT scheduler
    pbt_scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=2,
        metric="mean_reward",
        mode="max",
        hyperparam_mutations={
            # PPO hyperparams
            "learning_rate": tune.loguniform(1e-6, 1e-3),
            "ent_coef": tune.uniform(0.02, 0.1),   # V31 FIX: floor 0.02 anti-collapse
            "gamma": tune.uniform(0.9, 0.999),
            # Trading hyperparams (Ray PBT will auto-evolve these)
            "sl_pct": tune.uniform(0.01, 0.08),   # Stop-Loss: 1% to 8%
            "tp_pct": tune.uniform(0.02, 0.15),   # Take-Profit: 2% to 15%
        },
    )

    # Build per-trial param space
    # CRITICAL FIX: tune.grid_search creates N deterministic trials from the list.
    # If we also pass num_samples=N, Ray creates N × N = N² trials (cartesian product).
    # Solution: when using grid_search, set num_samples=1 — grid_search already
    # defines the exact trial count. PBT scheduler handles the rest.
    _profiles = profiles or []

    if _profiles:
        # Build paired worker_idx + profile configs — one trial per profile
        worker_configs = [
            {"worker_idx": i, "profile": _profiles[i % len(_profiles)]}
            for i in range(num_samples)
        ]
        param_space = {
            "adan_config": config,
            "worker_config": tune.grid_search(worker_configs),
            "envs_per_worker": envs_per_worker,
            "use_subproc": use_subproc,
            "interval_timesteps": interval_timesteps,
            # PPO hyperparams
            "learning_rate": tune.loguniform(1e-4, 1e-3),
            "ent_coef": tune.uniform(0.02, 0.05),  # V31 FIX: floor 0.02 anti-collapse
            "gamma": tune.uniform(0.95, 0.999),
            # Trading hyperparams (PBT auto-evolves these)
            "sl_pct": tune.uniform(0.01, 0.08),
            "tp_pct": tune.uniform(0.02, 0.15),
        }
        # CRITICAL: grid_search already defines len(worker_configs) trials
        # Set num_samples=1 to avoid len×num_samples trial explosion
        _actual_num_samples = 1
    else:
        param_space = {
            "adan_config": config,
            "worker_idx": tune.grid_search(list(range(num_samples))),
            "envs_per_worker": envs_per_worker,
            "use_subproc": use_subproc,
            "interval_timesteps": interval_timesteps,
            # PPO hyperparams
            "learning_rate": tune.loguniform(1e-4, 1e-3),
            "ent_coef": tune.uniform(0.02, 0.05),  # V31 FIX: floor 0.02 anti-collapse
            "gamma": tune.uniform(0.95, 0.999),
            # Trading hyperparams (PBT auto-evolves these)
            "sl_pct": tune.uniform(0.01, 0.08),
            "tp_pct": tune.uniform(0.02, 0.15),
        }
        # Same fix: grid_search defines the trial count
        _actual_num_samples = 1

    # Stop criteria
    if stop_config is None:
        stop_config = {"training_iteration": max_iterations}

    # Tuner – Ray >= 2.54 removed stop/verbose from RunConfig.
    # We handle stopping via the Trainable's own iteration counter.
    # Pass max_iterations through param_space so the worker can self-stop.
    param_space["_max_iterations"] = max_iterations

    tuner = None
    if resume:
        # Try to restore from exact storage_path first (it should contain experiment_state.json)
        restore_path = Path(storage_path)
        if not restore_path.exists():
            restore_path = Path(storage_path).parent / "adan_pbt_training"
        
        # FIX: Find the most recent experiment_state-*.json instead of hardcoded path
        import glob as _glob
        exp_states = sorted(_glob.glob(str(restore_path / "experiment_state-*.json")))
        
        if restore_path.exists() and exp_states:
            exp_state_file = exp_states[-1]  # Use the most recent one
            try:
                logger.info(f"Attempting restore from: {restore_path}")
                logger.info(f"Using experiment state: {Path(exp_state_file).name}")
                tuner = tune.Tuner.restore(
                    str(restore_path),
                    trainable=ADAN_PBT_Worker,
                    resume_errored=True,
                    restart_errored=False,
                )
                logger.info(f"✅ Successfully resumed training from {restore_path}")
            except Exception as e:
                logger.warning(f"Could not resume from {restore_path}: {e}. Starting fresh.")
                tuner = None
        else:
            logger.warning(f"No valid experiment state found at {restore_path}. Starting fresh.")
            tuner = None

    if tuner is None:
        # GPU sharing: each trial gets fraction of GPU
        _is_colab = False
        try:
            import google.colab  # noqa: F401
            _is_colab = True
        except ImportError:
            pass

        tune_config_kwargs = dict(
            scheduler=pbt_scheduler,
            num_samples=_actual_num_samples,  # FIXED: 1 because grid_search defines trial count
            max_concurrent_trials=num_samples,  # Allow all trials to run in parallel
            reuse_actors=True,  # EXPERT FIX: Must be True for PBT to prevent GCS crashes
        )
        # FIX: Increase timeouts to prevent GCS disconnects during long training
        # Limit concurrent trials for stability (user wants 2 simultaneous out of 4 total)
        _max_concurrent = min(num_samples, 2)
        tune_config_kwargs["max_concurrent_trials"] = _max_concurrent

        # Resource allocation must use the capacity explicitly exposed to Ray,
        # not os.cpu_count(): the host may have more CPUs than this run owns.
        _avail_cpus = max(1, int(num_cpus))
        trial_resources = None
        if _is_colab:
            import torch as _torch
            if _torch.cuda.is_available():
                # Colab: distribute GPU evenly across ALL trials (4 trials → 0.25 each)
                _cpu_per_trial = max(0.5, (_avail_cpus - 1) / max(num_samples, 1))
                _gpu_per_trial = 1.0 / max(num_samples, 1)
                trial_resources = {
                    "cpu": _cpu_per_trial,
                    "gpu": _gpu_per_trial,
                }
                logger.info(f"[COLAB] Trial resources: cpu={_cpu_per_trial:.1f}, gpu={_gpu_per_trial:.2f}")
        else:
            # VPS/local: no GPU, distribute CPUs across concurrent trials
            # Reserve 1 CPU for Ray overhead, split rest among concurrent trials
            # Example: 8 cores, 2 concurrent → (8-1)/2 = 3.5 CPUs per trial
            _cpu_per_trial = max(1.0, (_avail_cpus - 1) / max(_max_concurrent, 1))
            trial_resources = {
                "cpu": _cpu_per_trial,
                "gpu": 0,
            }
            logger.info(
                f"[VPS] Trial resources: cpu={_cpu_per_trial:.1f}/trial, "
                f"concurrent={_max_concurrent}/{num_samples}, "
                f"total_cpus={_avail_cpus}"
            )

        # Configure checkpointing: the Train and Tune variants are no longer
        # interchangeable in Ray 2.56.  ray.train.CheckpointConfig uses the
        # string sentinel "DEPRECATED" for checkpoint_frequency; Tune copies
        # that value into Trial.checkpoint_freq and later evaluates
        # training_iteration % checkpoint_freq, which raises TypeError.
        # Always use Tune's public config and pin the disabled iteration-based
        # frequency to an integer. ADAN's step-based save_checkpoint() remains
        # the sole checkpoint trigger.
        checkpoint_config = tune.CheckpointConfig(
            num_to_keep=10,  # Keep 10 most recent checkpoints (covers ~25k steps at 2500-step interval)
            checkpoint_score_attribute="timesteps_total",
            checkpoint_score_order="max",
            checkpoint_frequency=0,
        )
        if not isinstance(checkpoint_config.checkpoint_frequency, int):
            raise TypeError(
                "Ray Tune checkpoint_frequency must be int, got "
                f"{type(checkpoint_config.checkpoint_frequency).__name__}: "
                f"{checkpoint_config.checkpoint_frequency!r}"
            )

        # Ray's Tuner API does not accept trial_resources/resources_per_trial
        # (Ray 2.56), and TuneConfig never owned that parameter. Annotating the
        # trainable is the public cross-version scheduling API.
        scheduled_trainable = (
            tune.with_resources(ADAN_PBT_Worker, resources=trial_resources)
            if trial_resources
            else ADAN_PBT_Worker
        )
        tuner = tune.Tuner(
            scheduled_trainable,
            tune_config=tune.TuneConfig(**tune_config_kwargs),
            run_config=tune.RunConfig(
                name="adan_pbt_training",
                storage_path=str(storage_path),
                checkpoint_config=checkpoint_config,
                verbose=0,  # Disable verbose logging
                failure_config=tune.FailureConfig(max_failures=3),  # Retry on failure
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
    use_subproc: bool = False,
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
    storage_path = checkpoint_dir or str((TRAIN_OUTPUT_DIR / "ray_results").resolve())

    # ── Google Colab Detection & Auto-Configuration ───────────────────────
    IS_COLAB = False
    try:
        import google.colab  # noqa: F401
        IS_COLAB = True
        # Auto-detect Colab resources instead of hardcoding
        _available_cpus = os.cpu_count() or 2
        # Reserve 1 CPU for system/Ray overhead, use the rest
        _colab_cpus = max(2, _available_cpus - 1)
        logger.info(f"[COLAB] Google Colab detected — {_available_cpus} CPUs, using {_colab_cpus}")
        num_cpus = _colab_cpus
        envs_per_worker = 1  # 1 env per worker to minimize memory
        use_subproc = False   # Avoid fork conflicts with Ray
    except ImportError:
        IS_COLAB = False

    # ============================================================================
    # 🔧 SESSION 15: ULTIMATE RAY CONFIG - HARDENED FOR 16GB RAM + SSD SPILLING
    # ============================================================================
    # Hardware Profile: Intel 8-core + 16GB RAM + 16GB Swap + 11GB SSD free
    # Strategy: Bridge RAM with fast M.2 NVMe spilling to prevent GCS asphyxiation
    # ============================================================================

    # 1. Paths — portable and workspace-local by default. The historical
    # /mnt/new_data path does not exist on the production VPS and made Ray fail
    # before initialization. Operators may still override both paths explicitly.
    # Keep Ray's AF_UNIX socket path below Linux's 107-byte limit. A runtime
    # directory nested under PROJECT_ROOT produced a 120+ byte plasma socket on
    # this VPS. The workspace-level default is short, persistent and writable.
    _default_ray_root = PROJECT_ROOT.parents[1] / ".adan_ray"
    _ray_runtime_root = os.environ.get("ADAN_RAY_RUNTIME_ROOT", str(_default_ray_root))
    _ray_spill_dir = os.environ.get(
        "ADAN_RAY_SPILL_DIR", os.path.join(_ray_runtime_root, "spill")
    )
    _ray_tmp = os.environ.get("RAY_TMPDIR", os.path.join(_ray_runtime_root, "tmp"))
    os.makedirs(_ray_spill_dir, exist_ok=True)
    os.makedirs(_ray_tmp, exist_ok=True)

    # Build PYTHONPATH so Ray workers find adan_trading_bot without pip install
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _src_dir = os.path.join(_project_root, "src")
    _pythonpath = _src_dir + ":" + os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = _pythonpath
    _runtime_env = {"env_vars": {"PYTHONPATH": _pythonpath}}

    # 2. Object Store: Dynamic Config Based on Available RAM
    # RESTORED (29 mai 8153b72): Calculate based on system RAM for scalability
    # Original 2GB was too small for 16GB machine (caused crashes at 200 steps)
    # Use 25% of total RAM (conservative, scales across 4GB Colab to 16GB+ production)
    total_memory = int(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES'))
    OBJECT_STORE_GB = max(1_000_000_000, int(total_memory * 0.25))  # Min 1GB, 25% of RAM

    # 3. Spilling Config: Aggressive SSD spilling to prevent OOM
    # The SSD acts as "extended RAM" — writes at 3000+ MB/s so negligible latency
    spilling_config = {
        "type": "filesystem",
        "params": {
            "directory_path": _ray_spill_dir,
        }
    }

    # 4. System Config: Minimal but stable GCS settings
    system_config = {
        # Spilling
        "object_spilling_config": json.dumps(spilling_config),
        "automatic_object_spilling_enabled": True,

        # Memory Safety
        "memory_usage_threshold": 0.88,  # Kill workers at 88% RAM (safety margin)
    }

    # 5. Ray Init: Production-Grade
    # CRITICAL FIX (Session 18): Use loopback IP (127.0.0.1) instead of auto-detected
    # network IP to prevent crashes when Internet is disconnected.
    # Ray uses this for internal cluster communication, not external network.
    ray_init_kwargs = dict(
        num_cpus=num_cpus,
        object_store_memory=OBJECT_STORE_GB,
        include_dashboard=False,
        ignore_reinit_error=True,
        _temp_dir=_ray_tmp,
        runtime_env=_runtime_env,
        _system_config=system_config,
        _node_ip_address="127.0.0.1",  # Loopback - immune to network disconnects
    )

    # 6. Environment Variables: Additional Safety Layer
    # CRITICAL FIX (Session 18): Force loopback IP to prevent network dependency
    os.environ.update({
        "RAY_NODE_IP_ADDRESS": "127.0.0.1",  # Loopback - immune to Wi-Fi disconnects
        "RAY_memory_monitor_refresh_ms": "0",  # Disable aggressive killer
        "RAY_memory_usage_threshold": "0.88",
        "RAY_gcs_rpc_server_reconnect_timeout_s": "600",
        "RAY_health_check_failure_threshold": "10",
        "RAY_health_check_initial_delay_ms": "1000",
        "RAY_TMPDIR": _ray_tmp,
    })

    if IS_COLAB and torch.cuda.is_available():
        ray_init_kwargs["num_gpus"] = 1

    ray.init(**ray_init_kwargs)

    # 7. Validation Log
    logger.info("=" * 90)
    logger.info("🔥 ADAN PBT ULTIMATE CONFIG (SESSION 15 + FIX)")
    logger.info(f"   💾 Object Store: {OBJECT_STORE_GB // (1024**3):.1f}GB (25% of {total_memory // (1024**3):.1f}GB RAM) + SSD Spilling")
    logger.info(f"   📁 Spill Dir: {_ray_spill_dir}")
    logger.info(f"   🛡️  Memory Threshold: 88% (Kill workers before GCS asphyxiation)")
    logger.info(f"   ⏱️  GCS Reconnect: 600s (10 min patience for network hiccups)")
    logger.info(f"   📊 CPUs: {num_cpus}, Samples: {num_samples}, Envs/worker: {envs_per_worker}")
    logger.info("=" * 90)

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
            resume=resume,
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


# ===========================================================================
# SANDBOX MODE — Real architecture training (no Ray, no GPU)
# CPU-only, CI/sandbox/GitHub Actions compatible.
# Uses the REAL MultiAssetChunkedEnv with all production components:
#   HMM (3 regimes), DBE, FiLM, Oracle, PortfolioManager, RewardCalculator
# ===========================================================================
def sandbox_train(steps: int = None, initial_capital: float = None,
                  config_path: str = None, resume_ckpt: str = None,
                  checkpoint_out: str = None, worker_key: str = None):
    """Run training in sandbox/CI mode — no Ray, no GPU, single-process.

    Uses the REAL MultiAssetChunkedEnv from src/adan_trading_bot with all
    production modules (HMM, DBE, FiLM, Oracle, RewardCalculator, etc.).

    Session 9 — supports CI relay:
        resume_ckpt: path to a previously saved zip (and matching _vecnorm.pkl).
                     When provided, the PPO model + VecNormalize stats are
                     restored, and training continues from that checkpoint.
                     The total step count is reset_num_timesteps=False so
                     PPO's internal counter accumulates across relays.
        checkpoint_out: explicit output path for the new checkpoint zip
                        (default: checkpoints/ppo_adan0_sandbox_{steps}steps).

    Parameters are read from config.yaml [sandbox] section. CLI args override.
    """
    import copy as _copy

    # Load config from YAML (single source of truth)
    if config_path is None:
        config_path = str(PROJECT_ROOT / "config" / "config.yaml")
    config = ConfigLoader.load_config(config_path)

    # Read sandbox-specific parameters FROM config (Rule #16: no hardcoding)
    sandbox_cfg = config.get("sandbox", {})
    if initial_capital is None:
        initial_capital = float(sandbox_cfg.get("initial_capital", 20.50))
    if steps is None:
        steps = int(sandbox_cfg.get("max_training_steps", 5000))

    # Set initial capital in environment config
    if "environment" not in config:
        config["environment"] = {}
    config["environment"]["initial_capital"] = initial_capital
    logger.info(f"[SANDBOX] Config: steps={steps}, initial_capital={initial_capital}")

    # Worker config: use w1 (scalper) as default sandbox worker.
    # worker_key allows validating other profiles (w2=intraday, w3=swing,
    # w4=position) — accepts either the worker key ("w2") or a profile name
    # ("intraday"). Falls back to w1 when unknown.
    _workers = config.get("workers", {})
    _wkey = "w1"
    if worker_key:
        wk = str(worker_key).strip().lower()
        if wk in _workers:
            _wkey = wk
        else:
            # match by profile name
            _profile_map = {
                str(cfg.get("profile", "")).strip().lower(): k
                for k, cfg in _workers.items() if isinstance(cfg, dict)
            }
            if wk in _profile_map:
                _wkey = _profile_map[wk]
            else:
                logger.warning(
                    f"[SANDBOX] worker_key='{worker_key}' unknown — falling back to w1"
                )
    logger.info(f"[SANDBOX] Using worker_config key='{_wkey}' "
                f"(profile={_workers.get(_wkey, {}).get('profile', '?')})")
    worker_config = _copy.deepcopy(_workers.get(_wkey, {}))
    worker_config["worker_id"] = 0
    worker_config.setdefault("data_split_override", "train")
    worker_config.setdefault("timeframes", config.get("data", {}).get("timeframes", ["5m", "1h", "4h"]))
    worker_config.setdefault("assets", config.get("environment", {}).get("assets", ["BTCUSDT"]))

    # Load data using the real ChunkedDataLoader
    loader = ChunkedDataLoader(config=config, worker_config=worker_config, worker_id=0)
    data = loader.load_chunk(0)
    logger.info(f"[SANDBOX] Data loaded: {list(data.keys())} assets, "
                f"timeframes={list(list(data.values())[0].keys()) if data else 'none'}")

    # Create the REAL MultiAssetChunkedEnv
    env = MultiAssetChunkedEnv(
        data=data,
        config=config,
        worker_config=worker_config,
        worker_id=0,
        live_mode=False,
    )
    logger.info(f"[SANDBOX] MultiAssetChunkedEnv created — "
                f"obs_space keys: {list(env.observation_space.spaces.keys())}")

    # MEMORY FIX: Disable VecNormalize to prevent OOM crashes
    # VecNormalize maintains running statistics that accumulate in Ray's object store
    # causing progressive OOM after ~6900 steps. Using raw observations instead.
    vec_env = DummyVecEnv([lambda: env])
    gamma = config.get("agent", {}).get("gamma", 0.99)
    
    # Skip VecNormalize entirely - observations are already normalized in StateBuilder
    logger.info("[SANDBOX] VecNormalize DISABLED to prevent OOM crashes (using raw observations)")

    # Read hyperparams from config (NEVER hardcode)
    agent_cfg = config.get("agent", {})

    # Session 8 fix: Force exploration in 25D action space.
    # With standard Gaussian policy, Xavier init produces actions ≈ N(0, 0.01)
    # which NEVER cross action_threshold (0.05-0.10). Two fixes:
    # 1. use_sde=True: State-Dependent Exploration (gSDE) — exploration noise
    #    is learned as a function of state, producing coherent large actions.
    # 2. log_std_init=-0.5: Initial std ≈ exp(-0.5) ≈ 0.61, so actions
    #    start with nuanced variance inside [-1, 1], learning fine position sizing
    #    instead of saturating at extremes (previous 0.5 gave std≈1.65 → epilepsy).
    # V2 override : ADAN_LOG_STD_INIT (def -0.5, compat checkpoints). Le run
    # diagnostique le relève (0.0 -> std0≈1.0) pour rouvrir l'exploration.
    # gSDE STABILITY FIX (V2 execution audit, 2026-06-24 — MEASURED, not guessed):
    # gSDE variance = (latent_sde**2) @ (get_std(log_std)**2), i.e. for ~uniform
    # std,  σ_eff ≈ ||features||_2 * exp(log_std_init).
    # scripts/diag_gsde_latent.py MEASURED ||features||_2 ≈ 11.4 with the real
    # ContextualTemporalFusionExtractor (features_dim=256). So log_std_init=-0.5
    # gives σ_eff ≈ 6.9 AT INIT (chaotic) and PPO then drives log_std up further
    # -> σ explodes (3.4->13->41->110 observed). The old "frozen size μ=-7" was
    # the network DEFENDING against this chaos by saturating tanh.
    # Fixes (both SB3-documented, no architecture surgery):
    #   - log_std_init=-2.0  -> std≈0.135 -> σ_eff ≈ 1.5 (sane exploration)
    #   - use_expln=True     -> SB3: "keeps variance above zero and prevents it
    #                           from growing too fast" (bounds the blow-up)
    # NOTE: VecNormalize(norm_obs) is NOT the fix here — StateBuilder already
    # normalizes+clips obs to [-10,10] (measured), and heavy/500k_FIXED used
    # norm_obs=False too. LayerNorm on features makes it WORSE (||.||_2->16).
    _sb_log_std_init = float(os.environ.get("ADAN_LOG_STD_INIT", "-2.0"))
    _sb_use_expln = os.environ.get("ADAN_USE_EXPLN", "1") == "1"
    _sb_use_sde = os.environ.get("ADAN_USE_SDE", "1") == "1"
    policy_kwargs = {
        "share_features_extractor": True,
        "log_std_init": _sb_log_std_init,
    }
    if _sb_use_sde:
        # use_expln only matters for gSDE
        policy_kwargs["use_expln"] = _sb_use_expln

    # ------------------------------------------------------------------
    # CRITICAL FIX (V2 execution audit, 2026-06-24):
    # The sandbox mode previously built policy_kwargs WITHOUT
    # features_extractor_class, so SB3 silently fell back to its default
    # CombinedExtractor (a 0-parameter flatten of the Dict obs). That means
    # the CNN / cross-attention / FiLM context / aux forward-predictor NEVER
    # ran in sandbox training — only a bare MLP was trained. This made
    # sandbox checkpoints architecturally DIFFERENT from heavy-mode (Ray)
    # checkpoints and invalidated any μ/σ comparison against the 500K model.
    # We now wire the SAME ContextualTemporalFusionExtractor as heavy mode so
    # sandbox trains the real architecture (proof: scripts/audit_execution.py).
    # ------------------------------------------------------------------
    fe_kwargs = agent_cfg.get("features_extractor_kwargs", {})
    _cfg_pk = copy.deepcopy(fe_kwargs.get("policy_kwargs", {}))
    _activation_fn_map = {"ReLU": nn.ReLU, "Tanh": nn.Tanh, "LeakyReLU": nn.LeakyReLU}
    if "activation_fn" in _cfg_pk:
        _act_name = str(_cfg_pk["activation_fn"]).split(".")[-1]
        _cfg_pk["activation_fn"] = _activation_fn_map.get(_act_name, nn.ReLU)
    # carry over net_arch / activation_fn from config policy_kwargs (if any)
    for _k, _v in _cfg_pk.items():
        policy_kwargs.setdefault(_k, _v)

    if ContextualTemporalFusionExtractor is not None:
        policy_kwargs["features_extractor_class"] = ContextualTemporalFusionExtractor
        _valid_fe_keys = {"features_dim", "context_dim", "cnn_hidden", "dropout"}
        _safe_fe_kwargs = {k: v for k, v in fe_kwargs.items() if k in _valid_fe_keys}
        _safe_fe_kwargs.setdefault("context_dim", 14)
        policy_kwargs["features_extractor_kwargs"] = _safe_fe_kwargs
        logger.info(
            "[SANDBOX] features_extractor=ContextualTemporalFusionExtractor "
            f"(CNN+cross-attn+FiLM+aux) | fe_kwargs={_safe_fe_kwargs}"
        )
    else:
        logger.warning(
            "[SANDBOX] ContextualTemporalFusionExtractor UNAVAILABLE — falling "
            "back to SB3 CombinedExtractor (bare MLP). Architecture will NOT "
            "match heavy mode. Check the import at the top of this file."
        )

    logger.info(
        f"[SANDBOX] gSDE: use_sde={_sb_use_sde} use_expln={_sb_use_expln} "
        f"log_std_init={_sb_log_std_init:+.3f} (std0≈{float(np.exp(_sb_log_std_init)):.3f}) "
        f"-> expected σ_eff≈{11.4*float(np.exp(_sb_log_std_init)):.2f} at init "
        f"(target <~1.5)."
    )

    # S15 HARD RESET: Use config values (512/64/10) — safe for 7GB CI
    sandbox_n_steps = int(sandbox_cfg.get("n_steps", 512))
    sandbox_batch_size = int(sandbox_cfg.get("batch_size", 64))
    # GARDE-FOU (2026-06-27): n_epochs surchargeable via ADAN_N_EPOCHS.
    # Le gel a step 12417 s'est produit pendant un update PPO ; reduire
    # n_epochs (20->10) raccourcit la fenetre de backward intensif ou le
    # deadlock OpenMP/CPU se manifeste (test recommande utilisateur).
    sandbox_n_epochs = int(os.environ.get("ADAN_N_EPOCHS",
                                          sandbox_cfg.get("n_epochs", 10)))
    logger.info(f"[SANDBOX] PPO: n_steps={sandbox_n_steps}, batch_size={sandbox_batch_size}, "
                f"n_epochs={sandbox_n_epochs}")

    # Session 9 — relay support: load PPO from disk if --resume was given
    if resume_ckpt:
        logger.info(f"[SANDBOX] Resuming PPO from {resume_ckpt}")
        model = PPO.load(resume_ckpt, env=vec_env, device="cpu")
        # The PPO model loaded from disk keeps its own num_timesteps;
        # we tell .learn() to NOT reset it so we accumulate across relays.
        reset_num_timesteps = False
        prior_steps = int(getattr(model, "num_timesteps", 0))
        logger.info(f"[SANDBOX] Prior cumulative timesteps: {prior_steps}")
    else:
        # DIAGNOSTIC-V6.1: learning-rate WARMUP schedule. The FIRST PPO updates on
        # random init produced approx_kl 0.17 (>1.5*target_kl) -> "Early stopping
        # at step 0", almost no gradient applied. A warmup ramps lr from 10% ->
        # 100% of target over the first 20% of training, so the violent first
        # updates are tiny and KL stays under control; full lr afterwards.
        _lr_target = float(sandbox_cfg.get("learning_rate",
                           agent_cfg.get("learning_rate", 3e-4)))
        _warmup_frac = float(sandbox_cfg.get("lr_warmup_frac", 0.20))
        def _lr_schedule(progress_remaining: float) -> float:
            # SB3 passes progress_remaining: 1.0 at start -> 0.0 at end.
            done = 1.0 - float(progress_remaining)
            if _warmup_frac > 0 and done < _warmup_frac:
                ramp = 0.10 + 0.90 * (done / _warmup_frac)  # 0.10 -> 1.0
                return _lr_target * ramp
            return _lr_target
        logger.info(f"[SANDBOX] LR warmup schedule: target={_lr_target:.2e}, "
                    f"warmup_frac={_warmup_frac} (start={_lr_target*0.10:.2e})")
        # V15 (2026-07-07): when the L2 action anchor is requested
        # (ADAN_L2_ANCHOR_LAMBDA>0), the sandbox path MUST instantiate
        # WorldModelPPO — its overridden train() is the ONLY place the anchor
        # loss + Critic probes live. Vanilla PPO silently ignores the anchor.
        # aux_loss_coef defaults to 0.0 here so the diagnosis isolates the
        # anchor effect (no forward-prediction MSE confounding a0_mean).
        _anchor_lambda_env = float(os.environ.get("ADAN_L2_ANCHOR_LAMBDA", "0.0") or 0.0)
        _use_wmppo = (_anchor_lambda_env > 0.0) and (WorldModelPPO is not None)
        _SandboxPPOClass = WorldModelPPO if _use_wmppo else PPO
        _extra_ppo_kwargs = {}
        if _use_wmppo:
            _aux_coef = float(os.environ.get("ADAN_AUX_LOSS_COEF", "0.0") or 0.0)
            _extra_ppo_kwargs["aux_loss_coef"] = _aux_coef
            logger.warning(
                "[SANDBOX][V15] Using WorldModelPPO (anchor lambda=%.4f, "
                "aux_loss_coef=%.3f) — L2 action anchor + Critic probes ACTIVE.",
                _anchor_lambda_env, _aux_coef)
        else:
            logger.info("[SANDBOX] Using vanilla PPO (no L2 anchor requested).")
        model = _SandboxPPOClass(
            "MultiInputPolicy",
            vec_env,
            learning_rate=_lr_schedule,
            n_steps=sandbox_n_steps,
            batch_size=sandbox_batch_size,
            n_epochs=sandbox_n_epochs,
            gamma=float(agent_cfg.get("gamma", 0.99)),
            gae_lambda=float(agent_cfg.get("gae_lambda", 0.95)),
            # DIAGNOSTIC-V5: read clip_range/target_kl/max_grad_norm from the
            # sandbox block FIRST (that is the path that runs) so the PPO
            # stabilisation knobs actually take effect. clip_fraction was 0.73
            # and approx_kl 0.58 on the V4 run -> stricter trust region + KL
            # early-stop are required.
            clip_range=float(sandbox_cfg.get("clip_range",
                             agent_cfg.get("clip_range", 0.2))),
            target_kl=float(sandbox_cfg.get("target_kl",
                            agent_cfg.get("target_kl", 0.035))),
            ent_coef=float(os.environ.get(
                "ADAN_ENT_COEF",
                sandbox_cfg.get("ent_coef", agent_cfg.get("ent_coef", 0.01)))),
            vf_coef=float(agent_cfg.get("vf_coef", 0.5)),
            max_grad_norm=float(sandbox_cfg.get("max_grad_norm",
                                agent_cfg.get("max_grad_norm", 0.5))),
            use_sde=_sb_use_sde,       # gSDE (set ADAN_USE_SDE=0 to fall back to
                                       # plain DiagGaussian — σ then independent of
                                       # features, cannot diverge).
            sde_sample_freq=int(os.environ.get("ADAN_SDE_SAMPLE_FREQ", "4")),
            verbose=1,
            device="cpu",
            policy_kwargs=policy_kwargs,
            **_extra_ppo_kwargs,
        )
        reset_num_timesteps = True
        prior_steps = 0

    # Checkpoints directory
    ckpt_dir = PROJECT_ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # MEMORY FIX: Add frequent checkpoints to recover from OOM crashes
    # GARDE-FOU (2026-06-27): fréquence pilotable par ADAN_CKPT_FREQ (défaut 10k)
    # pour récupérer si re-gel à 12k-20k sans tout perdre.
    try:
        _ckpt_freq = int(os.environ.get("ADAN_CKPT_FREQ", "10000"))
    except Exception:
        _ckpt_freq = 10000
    _ckpt_freq = max(1000, min(_ckpt_freq, max(1000, steps)))
    _ckpt_prefix = os.environ.get(
        "ADAN_CKPT_PREFIX", "ppo_adan0_sandbox_checkpoint"
    ).strip() or "ppo_adan0_sandbox_checkpoint"
    checkpoint_callback = CheckpointCallback(
        save_freq=_ckpt_freq,  # Save every ADAN_CKPT_FREQ steps (default 10k)
        save_path=str(ckpt_dir),
        name_prefix=_ckpt_prefix,
        save_replay_buffer=False,  # Don't save replay buffer to save memory
        save_vecnormalize=False,  # VecNormalize is disabled, don't save it
    )
    logger.info(
        "[SANDBOX] Checkpoints every %d steps with prefix=%s",
        _ckpt_freq,
        _ckpt_prefix,
    )

    # V2 instrumentation (MESURE SEULE) — suit μ/σ pré-tanh par tête pour voir si
    # μ(size)=-7.2 remonte. Activé via ADAN_ACTIONDIM=1. NE MODIFIE RIEN.
    _sb_callbacks = [checkpoint_callback]

    # DIAGNOSTIC-V3 (2026-06-29): entropy-collapse telemetry. Activated by
    # ADAN_DIAG_COLLAPSE=1. Measure-only — logs action0 histo / HOLD% /
    # flat-open / illegal_ratio / entropy every ADAN_DIAG_EVERY (default 10k)
    # steps to a CSV. Feeds the post-50k decision tree.
    if os.environ.get("ADAN_DIAG_COLLAPSE", "0") == "1":
        _diag_csv = os.environ.get(
            "ADAN_DIAG_CSV",
            str(PROJECT_ROOT / "logs" / "training" / "diagnostic_collapse_v3.csv"),
        )
        _diag_every = int(os.environ.get("ADAN_DIAG_EVERY", "10000"))
        _sb_callbacks.append(DiagnosticCollapseCallback(
            csv_path=_diag_csv, log_every=_diag_every, verbose=1,
        ))
        logger.info(
            "[SANDBOX] DiagnosticCollapseCallback ACTIF "
            f"(every={_diag_every}, csv={_diag_csv}, "
            f"critic_breaker={os.environ.get('ADAN_CRITIC_BREAKER', '0')})"
        )

    if ActionDimMonitor is not None and os.environ.get("ADAN_ACTIONDIM", "0") == "1":
        _sb_ad_csv = os.environ.get(
            "ADAN_ACTIONDIM_CSV",
            str(ckpt_dir.parent / "logs" / "training" / "actiondim_sandbox.csv"),
        )
        _sb_callbacks.append(ActionDimMonitor(
            log_every=int(os.environ.get("ADAN_ACTIONDIM_EVERY", "1")),
            pre_tanh_batch=int(os.environ.get("ADAN_ACTIONDIM_BATCH", "256")),
            csv_path=_sb_ad_csv,
            verbose=1,
        ))
        logger.info(f"[SANDBOX] ActionDimMonitor ACTIF (csv={_sb_ad_csv}) — "
                    f"mesure seule, ne modifie rien.")

    t0 = time.time()
    model.learn(
        total_timesteps=steps,
        reset_num_timesteps=reset_num_timesteps,
        callback=_sb_callbacks,  # checkpoints + instrumentation V2
    )
    elapsed = time.time() - t0

    # Training is a lifecycle boundary. Close financially first, publish the
    # receipt-backed CLOSE, and include terminal PnL in run telemetry before
    # saving or producing the sandbox summary.
    final_receipts = env.finalize_open_positions(reason="TRAINING_END")
    terminal_pnl = sum(
        float(receipt.get("pnl_net", receipt.get("pnl", 0.0)) or 0.0)
        for receipt in final_receipts
    )
    env._finalize_episode_financial_telemetry(reset_close_pnl=terminal_pnl)

    # EXPERT FIX: Explicit GC after training to prevent memory accumulation
    import gc
    gc.collect()
    cumulative_steps = int(getattr(model, "num_timesteps", prior_steps + steps))

    # Save with cumulative-step naming so successive relays don't overwrite
    if checkpoint_out:
        ckpt_path = checkpoint_out.replace(".zip", "")
    else:
        ckpt_path = str(ckpt_dir / f"ppo_adan0_sandbox_{cumulative_steps}steps")
    model.save(ckpt_path)
    # Save VecNormalize stats if VecNormalize is active (may be disabled for OOM prevention)
    if hasattr(vec_env, 'save') and hasattr(vec_env, 'obs_rms'):
        vec_env.save(ckpt_path + "_vecnorm.pkl")
        vec_env.save(str(ckpt_dir / "vecnormalize_sandbox.pkl"))
    else:
        logger.info("[SANDBOX] VecNormalize disabled — skipping vecnorm save")

    # Persist scalers only when explicitly allowed. Production runs load the
    # already-fitted prod_scalers and must not rewrite their pickles/manifest at
    # shutdown (even a metadata-only timestamp rewrite obscures provenance).
    _save_scalers = os.environ.get("ADAN_SAVE_SCALERS", "1").strip().lower() in (
        "1", "true", "yes"
    )
    if _save_scalers:
        try:
            if hasattr(env, 'state_builder') and env.state_builder is not None:
                scalers_dir = str(PROJECT_ROOT / "prod_scalers")
                env.state_builder.save_scalers(scalers_dir)
                logger.info(f"[SANDBOX] Training scalers saved to {scalers_dir}")
        except Exception as e:
            logger.warning(f"[SANDBOX] Could not save training scalers: {e}")
    else:
        logger.info(
            "[SANDBOX] Persisted scalers kept read-only "
            "(ADAN_SAVE_SCALERS=0; no refit/save at shutdown)"
        )

    size = os.path.getsize(ckpt_path + ".zip")
    logger.info(f"[SANDBOX] Training done: +{steps} steps (cum={cumulative_steps}) "
                f"in {elapsed:.0f}s, checkpoint={ckpt_path}.zip ({size:,} bytes)")

    # Log key training and terminal financial stats after forced liquidation.
    info = env.get_info() if hasattr(env, 'get_info') else {}
    metric_rows = env.get_portfolio_metrics_dict() if hasattr(env, "get_portfolio_metrics_dict") else []
    financial_metrics = metric_rows[0] if metric_rows and isinstance(metric_rows[0], dict) else {}
    n_trades = info.get("run_completed_cycles", info.get("total_trades", "unknown"))
    logger.info(
        "[SANDBOX] Completed trade cycles: %s "
        "(opens=%s closes=%s open_positions=%s)",
        n_trades,
        info.get("run_opens", "unknown"),
        info.get("run_closes", "unknown"),
        info.get("run_open_positions", "unknown"),
    )

    return {
        "steps": steps,
        "cumulative_steps": cumulative_steps,
        "elapsed": elapsed,
        "checkpoint": ckpt_path + ".zip",
        "vecnorm": ckpt_path + "_vecnorm.pkl",
        "size": size,
        "trades": n_trades,
        "terminal_cash": float(financial_metrics.get("cash", info.get("cash", 0.0))),
        "terminal_equity": float(financial_metrics.get("equity", info.get("portfolio_value", 0.0))),
        "terminal_realized_pnl": float(financial_metrics.get(
            "realized_pnl_cumulative", info.get("total_realized_pnl", 0.0)
        )),
        "run_opens": int(financial_metrics.get("run_opens", info.get("run_opens", 0))),
        "run_closes": int(financial_metrics.get("run_closes", info.get("run_closes", 0))),
        "run_open_positions": int(financial_metrics.get(
            "run_open_positions", info.get("run_open_positions", 0)
        )),
        "resumed_from": resume_ckpt,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ADAN Agents — Dual-Mode")
    parser.add_argument("--mode", type=str, default="heavy",
                        choices=["sandbox", "heavy"],
                        help="sandbox: CI/sandbox/GH-Actions compatible (CPU, <400s, ~2GB RAM). "
                             "heavy: Full Ray Tune PBT (12GB+ RAM, GPU optional, Colab/Kaggle).")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config")
    parser.add_argument("--resume", action="store_true", help="Resume training (heavy mode)")
    # Session 9 — sandbox relay support: explicit checkpoint paths for CI
    parser.add_argument("--resume-from", type=str, default=None,
                        help="(sandbox) Path to a previous checkpoint .zip to resume from. "
                             "Looks for <ckpt>_vecnorm.pkl next to it.")
    parser.add_argument("--checkpoint-out", type=str, default=None,
                        help="(sandbox) Explicit output path for the new checkpoint .zip")
    parser.add_argument("--num-cpus", type=int, default=8, help="Number of CPUs for Ray")
    parser.add_argument("--num-samples", type=int, default=4, help="Number of concurrent PBT trials")
    parser.add_argument("--envs-per-worker", type=int, default=2, help="Sub-envs per worker (SubprocVecEnv)")
    parser.add_argument("--use-subproc", action="store_true", default=False, help="Use SubprocVecEnv (default: off)")
    parser.add_argument("--no-subproc", action="store_true", help="Use DummyVecEnv (default behaviour)")
    parser.add_argument("--steps", type=int, default=None,
                        help="Total training timesteps. In sandbox mode, if omitted "
                             "falls back to config [sandbox.max_training_steps]. "
                             "In heavy mode, defaults to 1_000_000. An explicit value "
                             "(incl. 1000000) is ALWAYS honored (bugfix v13: the old "
                             "default 1_000_000 collided with the 'use config' sentinel, "
                             "silently truncating explicit 1M runs to 10k).")
    parser.add_argument("--steps-per-iter", type=int, default=10_000, help="Timesteps per PBT iteration")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Override checkpoint dir")
    # Legacy args (ignored, kept for CLI compatibility)
    parser.add_argument("--num-envs", type=int, default=4, help="(legacy, ignored)")
    parser.add_argument("--progress-bar", action="store_true", help="(legacy, ignored)")
    parser.add_argument("--timeout", type=int, default=None, help="(legacy, ignored)")
    parser.add_argument("--fine-tune", action="store_true", help="(legacy, ignored)")
    parser.add_argument("--profiles", type=str, nargs="+", default=None,
                        help="Worker profiles, e.g. --profiles scalper swing "
                             "or --profiles 'scalper,intraday,swing,position'")

    args = parser.parse_args()

    # Normalize profiles: handle both comma-separated and space-separated
    if args.profiles:
        normalized = []
        for p in args.profiles:
            normalized.extend(p.split(","))
        args.profiles = [x.strip() for x in normalized if x.strip()]

    if args.mode == "sandbox":
        # ─── SANDBOX MODE ───
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        # steps=None means "read from config.yaml [sandbox.max_training_steps]".
        # v13 bugfix: pass args.steps VERBATIM. --steps default is now None (not
        # 1_000_000), so an explicit --steps 1000000 is honored instead of being
        # silently reinterpreted as "unspecified -> use config (10000)".
        result = sandbox_train(
            steps=args.steps,
            config_path=args.config if args.config != "config/config.yaml" else None,
            resume_ckpt=args.resume_from,
            checkpoint_out=args.checkpoint_out,
            worker_key=(args.profiles[0] if args.profiles else None),
        )
        print(json.dumps(result, indent=2, default=str))
    else:
        # ─── HEAVY MODE (default) ───
        # Full Ray Tune PBT — requires 12GB+ RAM, optional GPU
        # Compatible with Colab (T4/A100), Kaggle (P100), local multi-GPU
        main(
            config_path=args.config,
            resume=args.resume,
            num_cpus=args.num_cpus,
            num_samples=args.num_samples,
            envs_per_worker=args.envs_per_worker,
            use_subproc=not args.no_subproc,
            total_steps=(args.steps if args.steps is not None else 1_000_000),
            interval_timesteps=args.steps_per_iter,
            log_level=args.log_level,
            checkpoint_dir=args.checkpoint_dir,
            profiles=args.profiles,
        )
